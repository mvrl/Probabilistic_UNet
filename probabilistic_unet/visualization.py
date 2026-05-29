from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .data import LIDC_IDRI
from .model import ProbabilisticUnet
from .training import ModelConfig, resolve_device


@dataclass
class VisualizationConfig:
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("trained_model")
    output_dir: Path = Path("outputs/1")
    batch_size_val: int = 10
    save_batches: int = 3
    samples_per_example: int = 4
    val_split: float = 0.1
    num_workers: int = 0
    device: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)


def create_visualization_loader(config: VisualizationConfig, device: torch.device) -> DataLoader:
    dataset = LIDC_IDRI(dataset_location=config.data_dir)
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Dataset must contain at least two samples to create a validation split")

    indices = list(range(dataset_size))
    split = max(1, math.floor(config.val_split * dataset_size))
    if split >= dataset_size:
        split = dataset_size - 1

    test_indices = indices[:split]
    pin_memory = device.type == "cuda"
    test_loader = DataLoader(
        dataset,
        batch_size=config.batch_size_val,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    print(f"Number of test patches: {len(test_indices)}")
    return test_loader


def visualize_predictions(config: VisualizationConfig) -> Path:
    device = resolve_device(config.device)
    print(f"Using device: {device}")
    print(f"Using trained model from directory: {config.checkpoint_dir}")

    checkpoint_path = config.checkpoint_dir / "model_dict.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    save_dir = config.output_dir / "visual_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    test_loader = create_visualization_loader(config, device)
    model = ProbabilisticUnet(
        input_channels=config.model.input_channels,
        num_classes=config.model.num_classes,
        num_filters=list(config.model.num_filters),
        latent_dim=config.model.latent_dim,
        no_convs_fcomb=config.model.no_convs_fcomb,
        beta=config.model.beta,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    with torch.no_grad():
        for step, (patch, mask, _) in enumerate(test_loader):
            if step >= config.save_batches:
                break
            patch = patch.to(device)
            mask = mask.to(device).unsqueeze(1)
            output_samples = []
            for _ in range(config.samples_per_example):
                model(patch, training=False)
                output_samples.append(torch.sigmoid(model.sample()).detach().cpu().numpy())

            for item_index in range(patch.shape[0]):
                patch_out = patch[item_index, 0, :, :].detach().cpu().numpy()
                mask_out = mask[item_index, 0, :, :].detach().cpu().numpy()

                plt.figure()
                plt.subplot(3, 2, 1)
                plt.imshow(patch_out)
                plt.title("patch")
                plt.axis("off")
                plt.subplot(3, 2, 2)
                plt.imshow(mask_out)
                plt.title("GT Mask")
                plt.axis("off")

                for sample_index, sample in enumerate(output_samples):
                    plt.subplot(3, 2, sample_index + 3)
                    plt.imshow(sample[item_index, 0, :, :])
                    plt.title(f"prediction #{sample_index + 1}")
                    plt.axis("off")

                plt.savefig(save_dir / f"result_{step}_{item_index}.png", bbox_inches="tight")
                plt.close()

    print("Finished saving images")
    return save_dir
