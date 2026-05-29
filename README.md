# Probabilistic U-Net

A PyTorch implementation of the Probabilistic U-Net for ambiguous image segmentation.

**Paper:** Kohl et al., *A Probabilistic U-Net for Segmentation of Ambiguous Images* (NeurIPS 2018) — https://arxiv.org/abs/1806.05034

---

## What it does

Standard segmentation models produce a single deterministic mask for each input. When the correct segmentation is genuinely ambiguous — because of annotation disagreement, partial occlusion, or inherent semantic ambiguity — a single prediction is not enough.

The Probabilistic U-Net combines a U-Net with a conditional variational autoencoder (CVAE) latent space. At training time the model learns a posterior distribution `q(z | x, y)` that encodes *what makes a particular segmentation plausible* given the image. At test time it samples `z ~ p(z | x)` from the prior and decodes each sample through the U-Net features into a distinct, valid segmentation hypothesis.

---

## Architecture

```
                   ┌──────────────┐
                   │   image  x   │
                   └──────┬───────┘
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
      ┌─────────┐   ┌──────────┐  ┌──────────┐
      │  U-Net  │   │  Prior   │  │Posterior │  (posterior only during training)
      │features │   │ p(z | x) │  │q(z|x, y) │
      └────┬────┘   └────┬─────┘  └────┬─────┘
           │             │ sample z     │
           └──────┬──────┘             │
                  ▼                    │
            ┌──────────┐               │ KL divergence
            │  Fcomb   │◄──────────────┘
            │ (1×1 CNN)│
            └────┬─────┘
                 ▼
           segmentation ŷ
```

The **prior** and **posterior** are lightweight convolutional encoders that each output the mean and log-variance of a diagonal Gaussian.  The **Fcomb** head tiles the sampled latent vector `z` across the spatial dimensions of the U-Net feature map and combines them with a small 1×1 convolutional network.

---

## Loss function

Training maximises the ELBO per image:

```
ELBO(x, y) = E_{z ~ q(z|x,y)} [ log p(y | x, z) ]  −  β · KL[ q(z|x,y) || p(z|x) ]
```

- The **reconstruction term** is a summed binary cross-entropy between the predicted logits and the ground-truth mask.
- The **KL term** is computed analytically between two diagonal Gaussians.
- `β` (default 10) weights the KL penalty; higher values push the prior and posterior closer together and produce more diverse prior samples at test time.

---

## Installation

```bash
pip install -e .
```

For a GPU-specific PyTorch build, install that first from https://pytorch.org/get-started/locally/, then run the command above.

---

## Quick start

```python
from probabilistic_unet.training import TrainingConfig, ModelConfig, train_model

config = TrainingConfig(
    data_dir="data",          # directory containing preprocessed LIDC .pickle files
    output_dir="outputs/run-01",
    epochs=35,
    batch_size_train=20,
    model=ModelConfig(latent_dim=6),
)
train_model(config)
```

```python
from probabilistic_unet.visualization import VisualizationConfig, visualize_predictions

config = VisualizationConfig(
    data_dir="data",
    checkpoint_dir="outputs/run-01",
    output_dir="outputs/run-01",
    samples_per_example=4,
)
visualize_predictions(config)
```

Each call to `visualize_predictions` draws several independent samples from `p(z | x)` and saves them side-by-side so you can inspect the diversity of the model's hypotheses.

---

## ADE20k example: latent space as semantic class selector

The LIDC benchmark models uncertainty that arises from *annotator disagreement* — different radiologists draw slightly different boundaries around the same nodule.

A structurally identical form of ambiguity arises when a single scene contains many plausible objects to segment.  Given a photograph of a living room, "segment the chair", "segment the table", and "segment the window" are all valid tasks.  If we train the Probabilistic U-Net on (image, single-class binary mask) pairs where the class is chosen randomly at each training step, the model must learn a latent space whose samples correspond to different semantic classes.

`examples/ade20k_multiclass.py` demonstrates this.  It uses a SegFormer model pre-trained on ADE20k (via 🤗 Transformers) to automatically produce per-class binary masks from any collection of RGB images, then trains the Probabilistic U-Net on those pairs.

```bash
pip install transformers Pillow
python examples/ade20k_multiclass.py --image-dir /path/to/images --output-dir outputs/ade20k
```

At inference time, sampling different `z` values from the prior for the same image yields masks that highlight different objects in the scene — the latent space has learned to encode *which* object is being asked about rather than where its boundary is.

---

## Example results (LIDC)

The four small panels on the right are independent samples from `p(z | x)` for a single lung CT patch.  Each sample is a plausible nodule segmentation; together they characterise the model's uncertainty.

*Example 1*
![result1](https://github.com/Usman-Rafique/Probabilistic_UNet/blob/master/results/result_0_0.png)

*Example 2*
![result2](https://github.com/Usman-Rafique/Probabilistic_UNet/blob/master/results/result_0_2.png)
