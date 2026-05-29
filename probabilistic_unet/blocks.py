from __future__ import annotations

import torch
import torch.nn as nn

from .utils import init_weights


class DownConvBlock(nn.Module):
    """Three convolutional layers with ReLU activations and optional pooling."""

    def __init__(self, input_dim: int, output_dim: int, initializers: dict[str, str], padding: bool, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = []

        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """Upsampling block used by the U-Net decoder."""

    def __init__(self, input_dim: int, output_dim: int, initializers: dict[str, str], padding: bool, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, initializers, padding, pool=False)

    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        if self.bilinear:
            up = nn.functional.interpolate(x, mode="bilinear", scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)

        if up.shape[3] != bridge.shape[3]:
            raise ValueError("Upsampled tensor and skip connection have incompatible shapes")

        out = torch.cat([up, bridge], dim=1)
        return self.conv_block(out)
