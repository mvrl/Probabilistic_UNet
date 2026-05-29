from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            truncated_normal_(module.bias, mean=0.0, std=0.001)


def init_weights_orthogonal_normal(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            truncated_normal_(module.bias, mean=0.0, std=0.001)


def l2_regularisation(module: nn.Module) -> torch.Tensor:
    l2_reg = None
    for parameter in module.parameters():
        if l2_reg is None:
            l2_reg = parameter.norm(2)
        else:
            l2_reg = l2_reg + parameter.norm(2)
    if l2_reg is None:
        return torch.tensor(0.0)
    return l2_reg


def save_mask_prediction_example(mask: torch.Tensor, pred: torch.Tensor, iteration: int) -> None:
    plt.imshow(pred[0, :, :], cmap="Greys")
    plt.savefig(f"images/{iteration}_prediction.png")
    plt.imshow(mask[0, :, :], cmap="Greys")
    plt.savefig(f"images/{iteration}_mask.png")
