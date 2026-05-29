from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, kl

from .unet import Unet
from .utils import init_weights, init_weights_orthogonal_normal


class Encoder(nn.Module):
    """Convolutional encoder used for the prior and posterior networks."""

    def __init__(self, input_channels: int, num_filters: list[int], no_convs_per_block: int, initializers: dict[str, str], padding: bool = True, posterior: bool = False):
        super().__init__()
        self.input_channels = input_channels + int(posterior)
        self.num_filters = num_filters

        layers: list[nn.Module] = []
        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class AxisAlignedConvGaussian(nn.Module):
    """Gaussian parametrization network with diagonal covariance."""

    def __init__(self, input_channels: int, num_filters: list[int], no_convs_per_block: int, latent_dim: int, initializers: dict[str, str], posterior: bool = False):
        super().__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.name = "Posterior" if posterior else "Prior"
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, kernel_size=1, stride=1)

        nn.init.kaiming_normal_(self.conv_layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, patch: torch.Tensor, segm: torch.Tensor | None = None) -> Independent:
        if segm is not None:
            patch = torch.cat((patch, segm), dim=1)

        encoding = self.encoder(patch)
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        mu_log_sigma = self.conv_layer(encoding).squeeze(dim=(2, 3))
        mu = mu_log_sigma[:, : self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim :]
        return Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)


class Fcomb(nn.Module):
    """Combines latent samples with U-Net feature maps."""

    def __init__(self, num_filters: list[int], latent_dim: int, num_output_channels: int, num_classes: int, no_convs_fcomb: int, initializers: dict[str, str], use_tile: bool = True):
        super().__init__()
        self.num_channels = num_output_channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = "Fcomb"

        if self.use_tile:
            layers: list[nn.Module] = [
                nn.Conv2d(self.num_filters[0] + self.latent_dim, self.num_filters[0], kernel_size=1),
                nn.ReLU(inplace=True),
            ]

            for _ in range(no_convs_fcomb - 2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)
            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers["w"] == "orthogonal":
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, tensor: torch.Tensor, dim: int, n_tile: int) -> torch.Tensor:
        init_dim = tensor.size(dim)
        repeat_idx = [1] * tensor.dim()
        repeat_idx[dim] = n_tile
        tensor = tensor.repeat(*repeat_idx)
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(tensor.device)
        return torch.index_select(tensor, dim, order_index)

    def forward(self, feature_map: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)
        raise RuntimeError("Fcomb requires tiled latent features")


class ProbabilisticUnet(nn.Module):
    """A probabilistic U-Net implementation."""

    def __init__(self, input_channels: int = 1, num_classes: int = 1, num_filters: list[int] | None = None, latent_dim: int = 6, no_convs_fcomb: int = 4, beta: float = 10.0):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters or [32, 64, 128, 192]
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {"w": "he_normal", "b": "normal"}
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers, apply_last_layer=False, padding=True)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {"w": "orthogonal", "b": "normal"}, use_tile=True)

    def forward(self, patch: torch.Tensor, segm: torch.Tensor | None = None, training: bool = True) -> None:
        if training:
            if segm is None:
                raise ValueError("segm must be provided when training=True")
            self.posterior_latent_space = self.posterior(patch, segm)
        self.prior_latent_space = self.prior(patch)
        self.unet_features = self.unet(patch, False)

    def sample(self, testing: bool = False) -> torch.Tensor:
        if not testing:
            z_prior = self.prior_latent_space.rsample()
        else:
            z_prior = self.prior_latent_space.sample()
        self.z_prior_sample = z_prior
        return self.fcomb(self.unet_features, z_prior)

    def reconstruct(self, use_posterior_mean: bool = False, calculate_posterior: bool = False, z_posterior: torch.Tensor | None = None) -> torch.Tensor:
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.base_dist.loc
        elif calculate_posterior or z_posterior is None:
            z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb(self.unet_features, z_posterior)

    def kl_divergence(self, analytic: bool = True, calculate_posterior: bool = False, z_posterior: torch.Tensor | None = None) -> torch.Tensor:
        if analytic:
            return kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)

        if calculate_posterior or z_posterior is None:
            z_posterior = self.posterior_latent_space.rsample()
        log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
        log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
        return log_posterior_prob - log_prior_prob

    def elbo(self, segm: torch.Tensor, analytic_kl: bool = True, reconstruct_posterior_mean: bool = False) -> torch.Tensor:
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        z_posterior = self.posterior_latent_space.rsample()
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)
        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)
        return -(self.reconstruction_loss + self.beta * self.kl)
