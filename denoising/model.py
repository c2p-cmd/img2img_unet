import mlx.core as mx
import mlx.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (Downsampling)
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 32 -> 16

        # Bottleneck
        self.bottleneck = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=4,
            stride=2,  # upsample
            padding=1,
        )  # 16 -> 32
        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=4,
            stride=2,  # upsample
            padding=1,
        )  # 32 -> 64

        # Output mapping
        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def __call__(self, x):
        # Encode
        e1 = nn.relu(self.enc1(x))  # (B, 64, 64, 64)
        e2 = nn.relu(self.enc2(e1))  # (B, 32, 32, 128)
        e3 = nn.relu(self.enc3(e2))  # (B, 16, 16, 256)

        # Bottleneck
        b = nn.relu(self.bottleneck(e3))  # (B, 16, 16, 256)

        # Decode with Skip Connections
        d3 = nn.relu(self.up3(b))
        d2 = d3 + e2

        d2 = nn.relu(self.up2(d2))
        d1 = d2 + e1

        # Output
        out = nn.sigmoid(self.out(d1))
        return out
