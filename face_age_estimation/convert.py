import torch
from torch import nn
from model import FaceCNN
import pytorch_lightning as pl


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
            print(
                f"\n🔓 Epoch {self.current_epoch}: Unfreezing backbone for fine-tuning..."
            )

            for param in self.model.backbone.parameters():
                param.requires_grad = True
            self.backbone_frozen = False

            # Manually update optimizer learning rates
            optimizer = self.trainer.optimizers[0]

            # Create new parameter groups with different learning rates
            backbone_params = list(self.model.backbone.parameters())
            head_params = list(self.model.age_head.parameters()) + list(
                self.model.gender_head.parameters()
            )

            # Clear existing param groups
            optimizer.param_groups.clear()

            # Add new param groups with different LRs
            optimizer.add_param_group({"params": backbone_params, "lr": 1e-5})
            optimizer.add_param_group({"params": head_params, "lr": self.learning_rate})

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


face_lightning_model = FaceLightningModel.load_from_checkpoint(
    "checkpoints/face-efficientnet-epoch=74-val_loss=662.11.ckpt"
)
torch_model = face_lightning_model.model

torch.save(torch_model.state_dict(), "face_model.pth")
print("Done...")
