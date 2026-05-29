from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LIDC_IDRI(Dataset):
    def __init__(self, dataset_location: str | Path, transform=None):
        self.transform = transform
        self.images: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.series_uid: list[str] = []

        dataset_path = Path(dataset_location)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        pickle_files = sorted(path for path in dataset_path.iterdir() if path.suffix == ".pickle")
        if not pickle_files:
            raise FileNotFoundError(f"No .pickle files found in {dataset_path}")

        max_bytes = 2**31 - 1
        data: dict[str, dict[str, np.ndarray | str]] = {}
        for file_path in pickle_files:
            print(f"Loading file {file_path.name}")
            bytes_in = bytearray(0)
            input_size = os.path.getsize(file_path)
            with file_path.open("rb") as file_handle:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += file_handle.read(max_bytes)
            data.update(pickle.loads(bytes_in))

        for value in data.values():
            self.images.append(value["image"].astype(np.float32))
            self.labels.append(value["masks"].astype(np.float32))
            self.series_uid.append(value["series_uid"])

        if not (len(self.images) == len(self.labels) == len(self.series_uid)):
            raise ValueError("Loaded image, label, and series_uid counts do not match")

        for image in self.images:
            if np.max(image) > 1 or np.min(image) < 0:
                raise ValueError("Expected images to be normalized to [0, 1]")
        for label in self.labels:
            if np.max(label) > 1 or np.min(label) < 0:
                raise ValueError("Expected labels to be normalized to [0, 1]")
            if len(label) == 0:
                raise ValueError("Expected each example to contain at least one segmentation mask")

    def __getitem__(self, index: int):
        image = np.expand_dims(self.images[index], axis=0)
        # __init__ guarantees at least one mask per example.
        label_index = torch.randint(len(self.labels[index]), size=(1,)).item()
        label = self.labels[index][label_index].astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)

        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).float()
        return image_tensor, label_tensor, self.series_uid[index]

    def __len__(self) -> int:
        return len(self.images)
