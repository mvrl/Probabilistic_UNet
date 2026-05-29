from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DownConvBlock, UpConvBlock


class Unet(nn.Module):
    """A U-Net backbone."""

    def __init__(self, input_channels: int, num_classes: int, num_filters: list[int], initializers: dict[str, str], apply_last_layer: bool = True, padding: bool = True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps: list[torch.Tensor] = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            block_input = self.input_channels if i == 0 else block_output
            block_output = self.num_filters[i]
            pool = i != 0
            self.contracting_path.append(DownConvBlock(block_input, block_output, initializers, padding, pool=pool))

        self.upsampling_path = nn.ModuleList()
        for i in range(len(self.num_filters) - 2, -1, -1):
            block_input = block_output + self.num_filters[i]
            block_output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(block_input, block_output, initializers, padding))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(block_output, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, record_activations: bool) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i - 1])

        if record_activations:
            self.activation_maps.append(x)

        if self.apply_last_layer:
            x = self.last_layer(x)

        return x
