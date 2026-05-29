# Probabilistic U-Net (PyTorch + Typer CLI)

This repository contains a packaged Probabilistic U-Net implementation for ambiguous image segmentation, updated to work cleanly with modern PyTorch and a Typer-based command-line interface.

- Paper: https://arxiv.org/abs/1806.05034
- Original TensorFlow/PyTorch lineage: https://github.com/SimonKohl/probabilistic_unet and https://github.com/stefanknegt/Probabilistic-Unet-Pytorch

## What changed

- packaged the code as an installable `probabilistic_unet` Python module
- added a Typer CLI for training and visualization workflows
- replaced hard-coded `.cuda()` calls with modern device selection (`cuda`, `mps`, or `cpu`)
- updated checkpoint loading and loss usage for current PyTorch APIs
- kept `train_model.py` and `visualize.py` as thin compatibility entrypoints

## Repository layout

```text
probabilistic_unet/
  __init__.py
  __main__.py
  blocks.py
  cli.py
  data.py
  model.py
  training.py
  unet.py
  utils.py
  visualization.py
train_model.py
visualize.py
pyproject.toml
environment.yml
```

## Environment setup

The project now installs like a normal Python package.

### Option 1: `venv` + pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Option 2: Conda or Mamba

```bash
conda env create -f environment.yml
conda activate probabilistic-unet
```

### PyTorch note

If you need a platform-specific CUDA or ROCm build of PyTorch, install that build first using the selector on https://pytorch.org/get-started/locally/ and then run:

```bash
python -m pip install -e .
```

## Data setup

This code expects the preprocessed LIDC `.pickle` files used by prior versions of the project.

1. Download the preprocessed LIDC data from the link referenced by the upstream repository.
2. Create a `data/` directory in the repository root.
3. Place the `.pickle` files inside `data/`.

The loader will read every `*.pickle` file in that directory.

## CLI usage

After installation, use either the module form or the console script:

```bash
python -m probabilistic_unet --help
probabilistic-unet --help
```

### Train

```bash
probabilistic-unet train \
  --data-dir data \
  --output-dir outputs/run-01 \
  --epochs 35 \
  --batch-size-train 20 \
  --device cpu
```

### Visualize samples

```bash
probabilistic-unet visualize \
  --data-dir data \
  --checkpoint-dir trained_model \
  --output-dir outputs/run-01 \
  --samples-per-example 4
```

## Legacy entrypoints

The original script-style commands still work with default settings:

```bash
python train_model.py
python visualize.py
```

## Outputs

Training writes the following artifacts into the chosen output directory:

- `model_dict.pth`
- `loss_train.png`
- `loss_val.png`
- `logging.txt`
- `training_config.json`

Visualization writes sampled predictions into:

- `OUTPUT_DIR/visual_results/`

## Notes

- The validation split follows the existing repository behavior: the first portion of the dataset is used for validation and the remainder is used for training.
- The provided `trained_model/model_dict.pth` checkpoint remains usable through the new visualization command.
- This repository does not currently ship an automated dataset-backed test suite, so validation is primarily done through import, CLI, and model smoke checks.

## Example results

*Example Result 1*
![result1](https://github.com/Usman-Rafique/Probabilistic_UNet/blob/master/results/result_0_0.png)

*Example Result 2*
![result2](https://github.com/Usman-Rafique/Probabilistic_UNet/blob/master/results/result_0_2.png)
