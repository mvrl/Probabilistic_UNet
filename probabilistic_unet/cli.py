from __future__ import annotations

from pathlib import Path

import typer

from .training import ModelConfig, TrainingConfig, train_model
from .visualization import VisualizationConfig, visualize_predictions

app = typer.Typer(help="Train and visualize the Probabilistic U-Net.")


def parse_num_filters(value: str) -> tuple[int, ...]:
    try:
        filters = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter("num-filters must be a comma-separated list of integers") from exc
    if not filters:
        raise typer.BadParameter("num-filters must contain at least one filter size")
    return filters


@app.command()
def train(
    data_dir: Path = typer.Option(Path("data"), exists=False, file_okay=False, dir_okay=True, help="Directory containing preprocessed .pickle LIDC files."),
    output_dir: Path = typer.Option(Path("outputs/1"), file_okay=False, dir_okay=True, help="Directory for checkpoints and training artifacts."),
    epochs: int = typer.Option(35, min=1, help="Number of training epochs."),
    batch_size_train: int = typer.Option(20, min=1, help="Training batch size."),
    batch_size_val: int = typer.Option(1, min=1, help="Validation batch size."),
    lr: float = typer.Option(1e-5, help="Learning rate."),
    weight_decay: float = typer.Option(1e-6, help="Adam weight decay."),
    lr_decay_every: int = typer.Option(5, min=1, help="StepLR step size."),
    lr_decay: float = typer.Option(0.95, min=0.0, help="StepLR gamma."),
    val_split: float = typer.Option(0.1, min=0.0, max=0.9, help="Fraction of examples reserved for validation."),
    num_workers: int = typer.Option(0, min=0, help="DataLoader worker count."),
    device: str | None = typer.Option(None, help="Explicit torch device, for example cpu, cuda, or mps."),
    input_channels: int = typer.Option(1, min=1, help="Number of image channels."),
    num_classes: int = typer.Option(1, min=1, help="Number of output classes."),
    num_filters: str = typer.Option("32,64,128,192", help="Comma-separated filter sizes for the U-Net backbone."),
    latent_dim: int = typer.Option(2, min=1, help="Latent space dimension."),
    no_convs_fcomb: int = typer.Option(4, min=1, help="Number of 1x1 convolutions in the fcomb head."),
    beta: float = typer.Option(10.0, help="KL weight used in the ELBO."),
) -> None:
    config = TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        lr=lr,
        weight_decay=weight_decay,
        lr_decay_every=lr_decay_every,
        lr_decay=lr_decay,
        val_split=val_split,
        num_workers=num_workers,
        device=device,
        model=ModelConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            num_filters=parse_num_filters(num_filters),
            latent_dim=latent_dim,
            no_convs_fcomb=no_convs_fcomb,
            beta=beta,
        ),
    )
    train_model(config)
    typer.echo(f"Training artifacts saved to {output_dir}")


@app.command()
def visualize(
    data_dir: Path = typer.Option(Path("data"), exists=False, file_okay=False, dir_okay=True, help="Directory containing preprocessed .pickle LIDC files."),
    checkpoint_dir: Path = typer.Option(Path("trained_model"), exists=False, file_okay=False, dir_okay=True, help="Directory containing model_dict.pth."),
    output_dir: Path = typer.Option(Path("outputs/1"), file_okay=False, dir_okay=True, help="Directory where visualizations are written."),
    batch_size_val: int = typer.Option(10, min=1, help="Visualization batch size."),
    save_batches: int = typer.Option(3, min=1, help="Number of validation batches to render."),
    samples_per_example: int = typer.Option(4, min=1, help="Number of sampled masks per example."),
    val_split: float = typer.Option(0.1, min=0.0, max=0.9, help="Fraction of examples reserved for validation."),
    num_workers: int = typer.Option(0, min=0, help="DataLoader worker count."),
    device: str | None = typer.Option(None, help="Explicit torch device, for example cpu, cuda, or mps."),
    input_channels: int = typer.Option(1, min=1, help="Number of image channels."),
    num_classes: int = typer.Option(1, min=1, help="Number of output classes."),
    num_filters: str = typer.Option("32,64,128,192", help="Comma-separated filter sizes for the U-Net backbone."),
    latent_dim: int = typer.Option(2, min=1, help="Latent space dimension."),
    no_convs_fcomb: int = typer.Option(4, min=1, help="Number of 1x1 convolutions in the fcomb head."),
    beta: float = typer.Option(10.0, help="KL weight used in the ELBO."),
) -> None:
    config = VisualizationConfig(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        batch_size_val=batch_size_val,
        save_batches=save_batches,
        samples_per_example=samples_per_example,
        val_split=val_split,
        num_workers=num_workers,
        device=device,
        model=ModelConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            num_filters=parse_num_filters(num_filters),
            latent_dim=latent_dim,
            no_convs_fcomb=no_convs_fcomb,
            beta=beta,
        ),
    )
    save_dir = visualize_predictions(config)
    typer.echo(f"Visualization artifacts saved to {save_dir}")


def main() -> None:
    app()
