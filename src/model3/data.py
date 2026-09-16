"""DataLoaders over the preprocessed dataset produced by src/preprocessing.py.

The images in data/VN_trash_classification_preprocessing are already 224x224
RGB JPEGs, and the train split already had one round of augmentation baked in
when it was written to disk. Training for 30 epochs on a single frozen
augmentation would overfit, so a light on-the-fly flip + colour jitter is added
on top to give each epoch some variety. Validation and test get no randomness
at all, only the ImageNet normalisation the pretrained backbones expect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import DATA_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS


@dataclass
class DataBundle:
    """The three loaders plus the label vocabulary they share."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def split_sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train_loader.dataset),
            "val": len(self.val_loader.dataset),
            "test": len(self.test_loader.dataset),
        }


def build_train_transform() -> transforms.Compose:
    """Light on-the-fly augmentation - the heavy lifting already happened on disk."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Deterministic pipeline for val/test so scores are comparable run to run."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    data_dir: Path = DATA_DIR,
    seed: int = 42,
) -> DataBundle:
    """Create train/val/test loaders from the preprocessed image folders."""
    for split in ("train", "val", "test"):
        if not (data_dir / split).is_dir():
            raise FileNotFoundError(
                f"Missing {data_dir / split}. Run `python src/preprocessing.py --overwrite` first."
            )

    train_dataset = ImageFolder(data_dir / "train", transform=build_train_transform())
    val_dataset = ImageFolder(data_dir / "val", transform=build_eval_transform())
    test_dataset = ImageFolder(data_dir / "test", transform=build_eval_transform())

    _check_consistent_classes(train_dataset, val_dataset, test_dataset, data_dir)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    return DataBundle(
        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # keeps BatchNorm1d in the head away from 1-sample batches
            generator=generator,
            **loader_kwargs,
        ),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs),
        class_names=list(train_dataset.classes),
    )


def _check_consistent_classes(
    train_dataset: ImageFolder,
    val_dataset: ImageFolder,
    test_dataset: ImageFolder,
    data_dir: Path,
) -> None:
    """Fail loudly if the splits disagree, or drift from preprocessing's mapping."""
    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise RuntimeError(
            "train/val/test contain different class folders: "
            f"{train_dataset.classes} vs {val_dataset.classes} vs {test_dataset.classes}"
        )

    mapping_path = data_dir / "class_to_idx.json"
    if not mapping_path.exists():
        return

    with mapping_path.open(encoding="utf-8") as file:
        expected = json.load(file)
    if train_dataset.class_to_idx != expected:
        raise RuntimeError(
            f"Class order differs from {mapping_path}: "
            f"{train_dataset.class_to_idx} vs {expected}"
        )
