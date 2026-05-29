from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .data import LIDC_IDRI
from .model import ProbabilisticUnet


@dataclass
class ModelConfig:
    input_channels: int = 1
    num_classes: int = 1
    num_filters: tuple[int, ...] = (32, 64, 128, 192)
    latent_dim: int = 2
    no_convs_fcomb: int = 4
    beta: float = 10.0


@dataclass
class TrainingConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs/1")
    lr: float = 1e-5
    weight_decay: float = 1e-6
    lr_decay_every: int = 5
    lr_decay: float = 0.95
    batch_size_train: int = 20
    batch_size_val: int = 1
    epochs: int = 35
    val_split: float = 0.1
    num_workers: int = 0
    device: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)


def resolve_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_jsonable(config: TrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["data_dir"] = str(config.data_dir)
    payload["output_dir"] = str(config.output_dir)
    return payload


def create_data_loaders(config: TrainingConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    dataset = LIDC_IDRI(dataset_location=config.data_dir)
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Dataset must contain at least two samples to create train/validation splits")

    indices = list(range(dataset_size))
    split = max(1, math.floor(config.val_split * dataset_size))
    if split >= dataset_size:
        split = dataset_size - 1

    print("There is no random shuffle: initial portion of the dataset is used for validation and the rest for training")
    train_indices, val_indices = indices[split:], indices[:split]

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=config.batch_size_val,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    print(f"Number of training/validation patches: {(len(train_indices), len(val_indices))}")
    return train_loader, val_loader


def train_model(config: TrainingConfig) -> Path:
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Writing outputs to: {config.output_dir}")

    train_loader, val_loader = create_data_loaders(config, device)
    model = ProbabilisticUnet(
        input_channels=config.model.input_channels,
        num_classes=config.model.num_classes,
        num_filters=list(config.model.num_filters),
        latent_dim=config.model.latent_dim,
        no_convs_fcomb=config.model.no_convs_fcomb,
        beta=config.model.beta,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=config.lr_decay)

    train_loss: list[float] = []
    val_loss: list[float] = []
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        epoch_train_loss = 0.0
        for step, (patch, mask, _) in enumerate(train_loader):
            patch = patch.to(device)
            mask = mask.to(device).unsqueeze(1)
            model(patch, mask, training=True)
            loss = -model.elbo(mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.detach().cpu().item()

            if step % 100 == 0:
                print(f"[Ep {epoch + 1} {step + 1} of {len(train_loader)}] train loss: {epoch_train_loss / (step + 1)}")

        epoch_train_loss /= len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for patch, mask, _ in val_loader:
                patch = patch.to(device)
                mask = mask.to(device).unsqueeze(1)
                model(patch, mask, training=True)
                loss = -model.elbo(mask)
                epoch_val_loss += loss.detach().cpu().item()

        epoch_val_loss /= len(val_loader)
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        print(f"End of epoch {epoch + 1}, train loss: {epoch_train_loss}, val loss: {epoch_val_loss}")

        scheduler.step()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint_path = config.output_dir / "model_dict.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at epoch: {epoch + 1}")

    print("Finished training")

    plt.figure()
    plt.plot(train_loss)
    plt.title("train loss")
    plt.savefig(config.output_dir / "loss_train.png")
    plt.close()

    plt.figure()
    plt.plot(val_loss)
    plt.title("val loss")
    plt.savefig(config.output_dir / "loss_val.png")
    plt.close()

    (config.output_dir / "logging.txt").write_text(
        "Logging...\n"
        f"Validation loss {val_loss}\n"
        f"Training loss {train_loss}\n",
        encoding="utf-8",
    )
    (config.output_dir / "training_config.json").write_text(json.dumps(_to_jsonable(config), indent=2), encoding="utf-8")
    return config.output_dir
