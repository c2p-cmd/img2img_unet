import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FaceCNN(nn.Module):
    def __init__(self):
        super().__init__()               
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            efficientnet.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.age_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)    
        )
        self.gender_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Returns:
            age:   tensor of shape (batch, 1)
            gender_logits: tensor of shape (batch, 1)
        """
        features = self.backbone(x)          # shape (B, 1280, 1, 1)
        age = self.age_head(features)
        gender_logits = self.gender_head(features)
        return age, gender_logits