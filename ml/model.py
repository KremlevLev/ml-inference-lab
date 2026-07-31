import torch
import torch.nn as nn

class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()

        self.pointwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.pointwise(out)
        out = self.bn2(out)

        out = out + residual
        out = self.activation(out)

        return out

class ModernLightCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            )

        self.stage1 = nn.Sequential(
            ResidualDepthwiseBlock(32),
            ResidualDepthwiseBlock(32),
        )

        self.downsample1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.stage2 = nn.Sequential(
            ResidualDepthwiseBlock(64),
            ResidualDepthwiseBlock(64)
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        self.stage3 = nn.Sequential(
            ResidualDepthwiseBlock(128),
            ResidualDepthwiseBlock(128)
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(
            in_features=128,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)

        x = self.downsample1(x)
        x = self.stage2(x)

        x = self.downsample2(x)
        x = self.stage3(x)

        x = self.pool(x)
        x = self.flatten(x)
        logits = self.classifier(x)

        return logits

model = ModernLightCNN(num_classes=10)

x = torch.randn(8, 3, 32, 32)
logits = model(x)

print(logits.shape)