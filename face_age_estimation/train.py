import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torchvision import transforms
from model import FaceCNN

torch.set_float32_matmul_precision("medium")


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Weights based on the data distribution
        self.age_weights = {
            (0, 15): 4.0,  # Very sparse - highest penalty
            (15, 20): 2.5,  # Sparse
            (20, 30): 1.0,  # Most common - baseline
            (30, 40): 1.5,  # Moderate
            (40, 50): 2.0,  # Sparse
            (50, 60): 3.0,  # Very sparse
            (60, 70): 4.0,  # Minimal data
            (70, 120): 5.0,  # Extremely sparse - highest penalty
        }

    def forward(self, pred, target):
        weights = torch.ones_like(target)

        for (age_min, age_max), weight in self.age_weights.items():
            mask = (target >= age_min) & (target < age_max)
            weights[mask] = weight

        mse = (pred - target) ** 2
        weighted_mse = (mse * weights).mean()
        return weighted_mse


class FaceDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"]  # Shape: (B, 200, 200, 3)
        self.y = data["y"].astype(float)  # Shape: (B, 2) [age, gender]

        print(f"Size of data X:{self.X.shape}, y:{self.y.shape}")

        # ImageNet normalization for EfficientNet
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    size=(224, 224),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]

        # Apply ImageNet normalization
        image = self.transform(image)

        age = torch.FloatTensor([self.y[idx, 0]])
        gender = torch.FloatTensor([self.y[idx, 1]])  # 0 or 1

        return image, age, gender


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, npz_path, batch_size=64, train_split=0.8, val_split=0.1):
        super().__init__()
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split

    def setup(self, stage=None):
        dataset = FaceDataset(self.npz_path)

        total = len(dataset)
        train_size = int(total * self.train_split)
        val_size = int(total * self.val_split)
        test_size = total - train_size - val_size

        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )


class FaceLightningModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, age_weight=1.0, gender_weight=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.model = FaceCNN()
        self.backbone_frozen = True
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        self.learning_rate = learning_rate
        self.age_weight = age_weight
        self.gender_weight = gender_weight

        # Loss functions
        # self.age_loss_fn = nn.MSELoss()
        self.age_loss_fn = WeightedMSELoss()
        self.gender_loss_fn = nn.BCEWithLogitsLoss()

    def on_train_epoch_start(self):
        if self.current_epoch == 10 and self.backbone_frozen:
            print(f"\n🔓 Epoch {self.current_epoch}: Unfreezing backbone for fine-tuning...")
            
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            self.backbone_frozen = False
            
            # Manually update optimizer learning rates
            optimizer = self.trainer.optimizers[0]
            
            # Create new parameter groups with different learning rates
            backbone_params = list(self.model.backbone.parameters())
            head_params = (
                list(self.model.age_head.parameters()) + 
                list(self.model.gender_head.parameters())
            )
            
            # Clear existing param groups
            optimizer.param_groups.clear()
            
            # Add new param groups with different LRs
            optimizer.add_param_group({'params': backbone_params, 'lr': 1e-5})
            optimizer.add_param_group({'params': head_params, 'lr': self.learning_rate})
            
            print(f"✅ Backbone LR: 1e-5, Heads LR: {self.learning_rate}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, age_true, gender_true = batch
        age_pred, gender_pred = self(images)

        age_loss = self.age_loss_fn(age_pred, age_true)
        gender_loss = self.gender_loss_fn(gender_pred, gender_true)

        total_loss = self.age_weight * age_loss + self.gender_weight * gender_loss

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_age_loss", age_loss)
        self.log("train_gender_loss", gender_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, age_true, gender_true = batch
        age_pred, gender_pred = self(images)

        age_loss = self.age_loss_fn(age_pred, age_true)
        gender_loss = self.gender_loss_fn(gender_pred, gender_true)
        total_loss = self.age_weight * age_loss + self.gender_weight * gender_loss

        # Metrics
        gender_acc = ((gender_pred > 0.5).float() == gender_true).float().mean()
        age_mae = torch.abs(age_pred - age_true).mean()

        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_age_loss", age_loss)
        self.log("val_gender_loss", gender_loss)
        self.log("val_gender_acc", gender_acc, prog_bar=True)
        self.log("val_age_mae", age_mae, prog_bar=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        images, age_true, gender_true = batch
        age_pred, gender_pred = self(images)

        age_loss = self.age_loss_fn(age_pred, age_true)
        gender_loss = self.gender_loss_fn(gender_pred, gender_true)

        gender_acc = ((gender_pred > 0.5).float() == gender_true).float().mean()
        age_mae = torch.abs(age_pred - age_true).mean()

        self.log("test_age_loss", age_loss)
        self.log("test_gender_loss", gender_loss)
        self.log("test_gender_acc", gender_acc)
        self.log("test_age_mae", age_mae)

    def configure_optimizers(self):
        # Different learning rates for backbone vs heads
        if self.backbone_frozen:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.model.backbone.parameters(),
                        "lr": 1e-5,
                    },  # Lower LR
                    {
                        "params": self.model.age_head.parameters(),
                        "lr": self.learning_rate,
                    },
                    {
                        "params": self.model.gender_head.parameters(),
                        "lr": self.learning_rate,
                    },
                ]
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }


MAX_EPOCHS = 75

# Data
data_module = FaceDataModule("data.npz")

# Model
model = FaceLightningModel(
    learning_rate=1e-3,
    age_weight=3.0,
    gender_weight=1.0,
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val_age_mae",
    dirpath="checkpoints",
    filename="face-efficientnet-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)

early_stop_callback = EarlyStopping(monitor="val_loss", patience=15, mode="min")

logger = CSVLogger("logs", name="face-efficientnet")

# Trainer
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu",
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback],
    log_every_n_steps=10,
    logger=logger,
    precision="16-mixed",
)

# Train
trainer.fit(
    model,
    datamodule=data_module,
)

print("Validation Metrics")
trainer.validate(
    model=model,
    datamodule=data_module,
    ckpt_path="best",
)

print("Test Metrics")
trainer.test(
    model=model,
    datamodule=data_module,
    ckpt_path="best",
)

print(
    "Best Model",
    checkpoint_callback.best_model_path,
    "\nBest Score",
    checkpoint_callback.best_model_score,
)
