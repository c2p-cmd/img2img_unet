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
dataset_path = "../landscape-pictures/"
img_size = 64
batch_size = 32
lr = 1e-3
epochs = 10


# --- DATA LOADER ---
def load_images(path):
    # Find all JPGs recursively
    files = glob.glob(os.path.join(path, "**/*.jpg"), recursive=True)
    if not files:
        print("No images found! Check your path.")
        return []
    print(f"Found {len(files)} images.")
    return files


def get_batch(files, batch_size):
    # Randomly select files
    indices = np.random.choice(len(files), batch_size)
    batch_files = [files[i] for i in indices]

    images = []
    for f in batch_files:
        try:
            img = Image.open(f).convert("RGB")
            img = img.resize((img_size, img_size))
            arr = np.array(img).astype(np.float32) / 255.0
            images.append(arr)
        except:
            continue

    # Stack into (B, H, W, C)
    return mx.array(np.stack(images))


# --- NOISE FUNCTION ---
def add_noise(images, noise_factor=0.3):
    noise = mx.random.normal(images.shape)
    noisy = images + noise_factor * noise
    return mx.clip(noisy, 0.0, 1.0)


# --- TRAINING ---
def train():
    files = load_images(dataset_path)
    model = UNet()
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=lr)

    state = [model.state, optimizer.state]

    # Define Loss Function
    def loss_fn(model, clean_images):
        noisy_images = add_noise(clean_images)
        predicted_clean = model(noisy_images)
        diff = predicted_clean - clean_images
        diff_sq = mx.power(diff, 2)
        return mx.mean(diff_sq)

    # Compile the training step for speed
    @partial(mx.compile, inputs=state, outputs=state)
    def step(clean_images):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, clean_images)
        optimizer.update(model, grads)
        return loss

    print("Starting training...")

    # Simple training loop
    steps_per_epoch = len(files) // batch_size
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(steps_per_epoch):
            clean_batch = get_batch(files, batch_size)
            loss = step(clean_batch)
            mx.eval(loss)
            _loss_value = loss.item()
            epoch_loss += _loss_value

            if i % 10 == 0:
                print(
                    f"Epoch {epoch+1} | Step {i}/{steps_per_epoch} | Loss: {_loss_value:.4f}",
                    end="\r",
                )

        _avg_loss = epoch_loss / steps_per_epoch
        losses.append(_avg_loss)

        print(f"\nEpoch {epoch+1} Complete. Avg Loss: {_avg_loss:.4f}")

        # Save model every epoch
        model.save_weights(f"denoiser_epoch_{epoch}.safetensors")

    plt.figure(figsize=(10, 4))
    sns.lineplot(np.array(losses), label="Training Loss")
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("denoiser_loss.jpeg")
    plt.close()


if __name__ == "__main__":
    train()
