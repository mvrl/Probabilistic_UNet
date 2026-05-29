"""
ADE20k multi-class segmentation example for the Probabilistic U-Net.

Motivation
----------
The original Probabilistic U-Net paper models ambiguity that arises from
*annotator disagreement*: multiple radiologists independently annotate the
same lung CT patch and their segmentations differ.

This example demonstrates an analogous but structurally different source of
ambiguity: *semantic class ambiguity*.  Given a photograph of a living room,
all of the following are valid segmentation tasks on the same image:

    - "segment the chair"
    - "segment the table"
    - "segment the floor"

If we pair each image with a randomly chosen single-class binary mask at every
training step, the model must learn a latent space whose samples correspond to
different semantic classes.  At inference time, drawing several z ~ p(z | x)
from the learned prior should yield masks that highlight different objects in
the scene.

Setup
-----
    pip install transformers Pillow

Usage
-----
    python examples/ade20k_multiclass.py \\
        --image-dir /path/to/images \\
        --output-dir outputs/ade20k \\
        --epochs 50 \\
        --latent-dim 6
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
except ImportError as e:
    raise ImportError(
        "This example requires the 'transformers' and 'Pillow' packages.\n"
        "Install them with:  pip install transformers Pillow"
    ) from e

from probabilistic_unet.model import ProbabilisticUnet
from probabilistic_unet.training import resolve_device


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ADE20kMultiMaskDataset(Dataset):
    """
    Derives per-class binary masks from natural images using a SegFormer model
    pre-trained on ADE20k.

    Each call to ``__getitem__`` returns the *same* RGB image paired with the
    binary mask for one randomly chosen semantic class that is present in the
    scene.  The Probabilistic U-Net is therefore trained to produce any of the
    valid masks; the latent variable z must learn to encode *which* class is
    being asked for.

    Parameters
    ----------
    image_paths:
        Paths to the input images.
    processor:
        HuggingFace ``SegformerImageProcessor`` instance.
    seg_model:
        HuggingFace ``SegformerForSemanticSegmentation`` instance.
    device:
        Torch device on which the segmentation model runs.
    image_size:
        Spatial size (H, W) to which all images are resized.
    min_mask_fraction:
        Classes whose mask covers less than this fraction of the image are
        discarded (avoids near-empty masks for tiny objects).
    """

    def __init__(
        self,
        image_paths: list[Path],
        processor: SegformerImageProcessor,
        seg_model: SegformerForSemanticSegmentation,
        device: torch.device,
        image_size: tuple[int, int] = (128, 128),
        min_mask_fraction: float = 0.01,
    ) -> None:
        self.image_size = image_size
        self.images: list[np.ndarray] = []      # (3, H, W) float32 in [0, 1]
        self.class_masks: list[dict[int, np.ndarray]] = []  # {class_id: (H, W)}

        seg_model = seg_model.to(device).eval()

        for path in image_paths:
            pil_img = Image.open(path).convert("RGB").resize(
                (image_size[1], image_size[0]), Image.BILINEAR
            )
            img_np = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)

            seg_map = _segment(pil_img, processor, seg_model, device, image_size)

            pixel_total = seg_map.size
            masks: dict[int, np.ndarray] = {}
            for class_id in np.unique(seg_map):
                binary = (seg_map == class_id).astype(np.float32)
                if binary.mean() >= min_mask_fraction:
                    masks[int(class_id)] = binary

            if not masks:
                continue  # skip images where every class is too small

            self.images.append(img_np.transpose(2, 0, 1))  # (3, H, W)
            self.class_masks.append(masks)

        if not self.images:
            raise ValueError(
                "No usable images found.  Make sure the images contain objects "
                "that cover at least min_mask_fraction of the image."
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]                           # (3, H, W)
        masks = self.class_masks[idx]
        class_id = int(np.random.choice(list(masks.keys())))
        mask = masks[class_id]                           # (H, W)
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(mask).float(),
            f"img{idx}_class{class_id}",
        )


def _segment(
    pil_img: Image.Image,
    processor: SegformerImageProcessor,
    seg_model: SegformerForSemanticSegmentation,
    device: torch.device,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Return an (H, W) integer array of ADE20k class indices."""
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = seg_model(**inputs).logits  # (1, num_classes, H/4, W/4)
    seg_map = (
        F.interpolate(logits, size=image_size, mode="bilinear", align_corners=False)
        .argmax(dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
    )
    return seg_map


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    dataset: ADE20kMultiMaskDataset,
    output_dir: Path,
    *,
    epochs: int = 50,
    batch_size: int = 4,
    latent_dim: int = 6,
    beta: float = 10.0,
    lr: float = 1e-4,
    val_fraction: float = 0.15,
    device: torch.device,
) -> ProbabilisticUnet:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_val = max(1, math.floor(val_fraction * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = ProbabilisticUnet(
        input_channels=3,
        num_classes=1,
        latent_dim=latent_dim,
        beta=beta,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for patch, mask, _ in train_loader:
            patch = patch.to(device)
            mask = mask.to(device).unsqueeze(1)
            model(patch, mask, training=True)
            loss = -model.elbo(mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for patch, mask, _ in val_loader:
                patch = patch.to(device)
                mask = mask.to(device).unsqueeze(1)
                model(patch, mask, training=True)
                epoch_val_loss += (-model.elbo(mask)).item()
        epoch_val_loss /= len(val_loader)

        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}  train={epoch_loss:.2f}  val={epoch_val_loss:.2f}")

        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            torch.save(model.state_dict(), output_dir / "model_dict.pth")

    _plot_losses(train_losses, val_losses, output_dir)
    return model


def _plot_losses(train: list[float], val: list[float], out: Path) -> None:
    plt.figure()
    plt.plot(train, label="train")
    plt.plot(val, label="val")
    plt.xlabel("epoch")
    plt.title("ELBO loss")
    plt.legend()
    plt.savefig(out / "loss.png")
    plt.close()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(
    dataset: ADE20kMultiMaskDataset,
    model: ProbabilisticUnet,
    output_dir: Path,
    device: torch.device,
    n_images: int = 4,
    n_samples: int = 6,
) -> None:
    """
    For each of the first ``n_images`` images, draw ``n_samples`` independent
    samples from the learned prior p(z | x) and save a figure showing:

    - the input RGB image
    - the ADE20k-derived class masks that were used during training
    - the sampled segmentation hypotheses

    The diversity of the samples reveals what the model has learned about
    which objects are plausible to segment in the scene.
    """
    save_dir = output_dir / "visual_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for img_idx in range(min(n_images, len(dataset))):
            img_np = dataset.images[img_idx]            # (3, H, W)
            masks = dataset.class_masks[img_idx]

            patch = torch.from_numpy(img_np).unsqueeze(0).to(device)  # (1, 3, H, W)
            model(patch, training=False)

            samples = [
                torch.sigmoid(model.sample(testing=True)).squeeze().cpu().numpy()
                for _ in range(n_samples)
            ]

            n_gt = len(masks)
            n_cols = max(n_gt, n_samples)
            fig, axes = plt.subplots(3, n_cols, figsize=(2.5 * n_cols, 7))

            # Row 0: input image (repeated across columns)
            for ax in axes[0]:
                ax.imshow(img_np.transpose(1, 2, 0))
                ax.axis("off")
            axes[0][0].set_title("input image", fontsize=8)

            # Row 1: ground-truth class masks derived from SegFormer
            for col, (class_id, mask) in enumerate(masks.items()):
                if col >= n_cols:
                    break
                axes[1][col].imshow(mask, cmap="gray", vmin=0, vmax=1)
                axes[1][col].set_title(f"GT class {class_id}", fontsize=8)
                axes[1][col].axis("off")
            for col in range(len(masks), n_cols):
                axes[1][col].axis("off")

            # Row 2: prior samples from the model
            for col, sample in enumerate(samples):
                if col >= n_cols:
                    break
                axes[2][col].imshow(sample, cmap="gray", vmin=0, vmax=1)
                axes[2][col].set_title(f"sample {col + 1}", fontsize=8)
                axes[2][col].axis("off")
            for col in range(len(samples), n_cols):
                axes[2][col].axis("off")

            axes[0][0].set_ylabel("input", fontsize=8)
            axes[1][0].set_ylabel("GT masks", fontsize=8)
            axes[2][0].set_ylabel("prior samples", fontsize=8)

            plt.tight_layout()
            fig.savefig(save_dir / f"result_{img_idx}.png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"Saved result_{img_idx}.png")

    print(f"Results written to {save_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", required=True, type=Path, help="Directory containing input images (JPEG, PNG, …)")
    p.add_argument("--output-dir", default=Path("outputs/ade20k"), type=Path)
    p.add_argument("--image-size", default=128, type=int, help="Images are resized to this square resolution")
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--batch-size", default=4, type=int)
    p.add_argument("--latent-dim", default=6, type=int)
    p.add_argument("--beta", default=10.0, type=float, help="KL weight in the ELBO")
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--n-vis-images", default=4, type=int, help="How many images to visualise")
    p.add_argument("--n-samples", default=6, type=int, help="Prior samples per visualised image")
    p.add_argument(
        "--segformer-model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="HuggingFace model ID for the ADE20k SegFormer",
    )
    p.add_argument("--device", default=None, help="pytorch device string (default: auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    image_paths = sorted(
        p for p in args.image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")
    print(f"Found {len(image_paths)} images")

    print(f"Loading SegFormer ({args.segformer_model}) …")
    processor = SegformerImageProcessor.from_pretrained(args.segformer_model)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(args.segformer_model)

    print("Building dataset …")
    dataset = ADE20kMultiMaskDataset(
        image_paths=image_paths,
        processor=processor,
        seg_model=seg_model,
        device=device,
        image_size=(args.image_size, args.image_size),
    )
    print(f"Dataset contains {len(dataset)} images with usable masks")

    print("Training …")
    model = train(
        dataset,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        beta=args.beta,
        lr=args.lr,
        device=device,
    )

    print("Visualising …")
    visualize(dataset, model, args.output_dir, device, n_images=args.n_vis_images, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
