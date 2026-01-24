import os
import glob
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from PIL import Image
import numpy as np
from model import UNet
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
batch_size = 32
lr = 2e-4
steps = 10_000


# --- METRICS ---
def calculate_psnr(pred, target, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio
    Higher is better (typically 20-50 dB for good quality)
    """
    mse = mx.mean((pred - target) ** 2)
    if mse == 0:
        return mx.array(mx.inf)
    psnr = 20 * mx.log10(max_val / mx.sqrt(mse))
    return psnr


def calculate_ssim(pred, target, data_range=1.0, k1=0.01, k2=0.03):
    """
    Simplified SSIM calculation for image quality
    Returns value between -1 and 1 (1 is perfect match)
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_pred = mx.mean(pred)
    mu_target = mx.mean(target)

    sigma_pred = mx.mean((pred - mu_pred) ** 2)
    sigma_target = mx.mean((target - mu_target) ** 2)
    sigma_pred_target = mx.mean((pred - mu_pred) * (target - mu_target))

    ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred**2 + mu_target**2 + c1) * (sigma_pred + sigma_target + c2)
    )

    return ssim


def plot_metrics(steps, train_losses, val_losses, val_psnrs, val_ssims):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Loss plot
    axes[0].plot(steps, train_losses, label="Train Loss", marker="o")
    val_steps = [s for s in steps if s % 500 == 0 and s > 0]
    axes[0].plot(val_steps, val_losses, label="Val Loss", marker="s")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("L1 Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # PSNR plot
    axes[1].plot(val_steps, val_psnrs, label="Val PSNR", color="green", marker="s")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation PSNR")
    axes[1].legend()
    axes[1].grid(True)

    # SSIM plot
    axes[2].plot(val_steps, val_ssims, label="Val SSIM", color="orange", marker="s")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("Validation SSIM")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("colorizer_metrics.png", dpi=150)
    plt.close()


# --- DATA LOADER ---
def prepare_datasets(dataset_path, val_split=0.1):
    dataset = mx.load(dataset_path)
    X = dataset["X"]
    y = dataset["y"]

    # Shuffle indices
    n_samples = X.shape[0]
    indices = mx.random.permutation(n_samples)

    # Split
    val_size = int(n_samples * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_data = {"X": X[train_indices], "y": y[train_indices]}
    val_data = {"X": X[val_indices], "y": y[val_indices]}

    return train_data, val_data


# --- VALIDATION ---
def validate(model, val_data, batch_size=32):
    """
    Run validation and compute metrics
    """
    model.eval()  # Set to evaluation mode if needed

    X_val = val_data["X"]
    y_val = val_data["y"]

    # Use a subset for faster validation (e.g., 10 batches)
    n_val_batches = min(10, X_val.shape[0] // batch_size)

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    for i in range(n_val_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_val[start_idx:end_idx]
        y_batch = y_val[start_idx:end_idx]

        # Forward pass (no gradients needed)
        predicted_ab = model(X_batch)

        # Calculate metrics
        loss = mx.mean(mx.abs(y_batch - predicted_ab))
        psnr = calculate_psnr(predicted_ab, y_batch)
        ssim = calculate_ssim(predicted_ab, y_batch)

        total_loss += loss.item()
        total_psnr += psnr.item()
        total_ssim += ssim.item()

    # Average over batches
    avg_loss = total_loss / n_val_batches
    avg_psnr = total_psnr / n_val_batches
    avg_ssim = total_ssim / n_val_batches

    return avg_loss, avg_psnr, avg_ssim


# --- TRAINING ---
def train():
    # Load and split data
    train_data, val_data = prepare_datasets("preprocess_data.npz", val_split=0.1)
    print(f"Training samples: {train_data['X'].shape[0]}")
    print(f"Validation samples: {val_data['X'].shape[0]}")

    model = UNet()
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=lr)

    # Tracking metrics
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    steps_logged = []

    print("Starting training...")

    for step in range(steps):
        # Training step
        idx = mx.random.randint(
            low=0,
            high=train_data["X"].shape[0],
            shape=(batch_size,),
        )
        X_batch = train_data["X"][idx]
        y_batch = train_data["y"][idx]

        # Forward and backward pass
        def loss_fn(model, X, y):
            predicted_ab = model(X)
            return mx.mean(mx.abs(y - predicted_ab))

        loss, grads = nn.value_and_grad(model, loss_fn)(model, X_batch, y_batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # Log training progress
        if step % 100 == 0:
            mx.eval(loss)
            train_losses.append(loss.item())
            steps_logged.append(step)
            print(f"Step {step}/{steps} | Train Loss: {loss.item():.6f}")

        # Run validation
        if step % 500 == 0 and step > 0:
            val_loss, val_psnr, val_ssim = validate(model, val_data, batch_size)
            val_losses.append(val_loss)
            val_psnrs.append(val_psnr)
            val_ssims.append(val_ssim)

            print(
                f"  Validation | Loss: {val_loss:.6f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}"
            )
            model.save_weights(f"colorizer_step_{step}.safetensors")

    # Plot metrics
    plot_metrics(steps_logged, train_losses, val_losses, val_psnrs, val_ssims)


if __name__ == "__main__":
    train()
