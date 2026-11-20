import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Standard Residual Block for DroNet (ResNet-8 backbone).
    Uses basic PyTorch operators compatible with Android PyTorch Mobile 1.13.1:
    - Conv2d, BatchNorm2d, ReLU, Add
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual  # Basic Add operator
        out = self.relu2(out)
        return out


class DroNet(nn.Module):
    """
    DroNet (ResNet-8 based Architecture) for End-to-End Drone Control.

    Inputs:
        x: Tensor of shape [Batch, 3, 224, 224] (RGB image with ImageNet normalization)
    Outputs:
        Tensor of shape [Batch, 3] -> [Roll, Pitch, Yaw]
    """
    def __init__(self, num_outputs: int = 3):
        super(DroNet, self).__init__()

        # Initial Convolutional Layer
        # Input: [B, 3, 224, 224] -> Output: [B, 32, 112, 112]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=False)
        # Output: [B, 32, 56, 56]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Blocks (ResNet-8 Backbone)
        # Layer 1: [B, 32, 56, 56] -> [B, 32, 28, 28]
        self.layer1 = ResBlock(32, 32, stride=2)
        # Layer 2: [B, 32, 28, 28] -> [B, 64, 14, 14]
        self.layer2 = ResBlock(32, 64, stride=2)
        # Layer 3: [B, 64, 14, 14] -> [B, 128, 7, 7]
        self.layer3 = ResBlock(64, 128, stride=2)

        # Average Pooling with fixed kernel (kernel_size=7 for 224x224 input)
        # Replaces AdaptiveAvgPool2d for strict PyTorch Mobile 1.13.1 compatibility
        # Output: [B, 128, 1, 1]
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        # Regression Head for Roll, Pitch, Yaw (Throttle fixed on drone side)
        self.fc1 = nn.Linear(128, 64)
        self.relu_fc = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(64, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = self.relu_fc(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    # Self-test model initialization and forward pass
    net = DroNet(num_outputs=3)
    dummy_x = torch.randn(1, 3, 224, 224)
    out = net(dummy_x)
    print(f"Model successfully initialized. Input shape: {dummy_x.shape}, Output shape: {out.shape}")
