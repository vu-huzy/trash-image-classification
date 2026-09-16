"""Create a physically preprocessed copy of the trash image dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import stat
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torchvision import transforms


RANDOM_SEED = 42
IMAGE_SIZE = 224
VALIDATION_SIZE = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _force_remove_readonly(function, path, excinfo):
    """shutil.rmtree onexc hook: clear the read-only bit and retry once."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def remove_directory_with_retry(path: Path, attempts: int = 5, delay_seconds: float = 1.0) -> None:
    """Delete a directory tree, retrying on Windows file-lock races.

    A fresh shutil.rmtree() right after writing thousands of small JPEGs can
    intermittently fail with WinError 145 ("directory not empty") because
    Windows Defender or the search indexer still holds a handle on a file
    that was just closed. Retrying after a short delay resolves it.
    """
    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(path, onexc=_force_remove_readonly)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(delay_seconds)


def collect_images(split_dir: Path, split_name: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Collect readable images and report files that cannot be opened."""
    records: list[dict[str, str]] = []
    bad_files: list[dict[str, str]] = []

    for image_path in sorted(split_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            bad_files.append({"path": str(image_path), "error": str(error)})
            continue

        records.append(
            {
                "path": str(image_path),
                "label": image_path.parent.name,
                "original_split": split_name,
            }
        )

    return pd.DataFrame(records, columns=["path", "label", "original_split"]), bad_files


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return the notebook's train augmentation and deterministic eval transform."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    )
    return train_transform, eval_transform


def save_split(
    dataframe: pd.DataFrame,
    split_name: str,
    output_dir: Path,
    transform: transforms.Compose,
) -> pd.DataFrame:
    """Transform and save one split while preserving its class folders."""
    split_output_dir = output_dir / split_name
    records: list[dict[str, str]] = []

    for image_index, row in dataframe.iterrows():
        source_path = Path(str(row["path"]))
        label = str(row["label"])
        class_dir = split_output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        output_path = class_dir / f"{image_index:06d}.jpg"

        with Image.open(source_path) as image:
            transformed_image = transform(image.convert("RGB"))
            transformed_image.save(output_path, format="JPEG", quality=95)

        records.append(
            {
                "path": str(output_path),
                "label": label,
                "split": split_name,
                "source_path": str(source_path),
            }
        )

    return pd.DataFrame(records, columns=["path", "label", "split", "source_path"])


def build_dataset(project_dir: Path, overwrite: bool = False) -> Path:
    """Build the preprocessed dataset and return its output path."""
    source_dir = project_dir / "data" / "VN_trash_classification"
    output_dir = project_dir / "data" / "VN_trash_classification_preprocessing"
    train_dir = source_dir / "Train"
    test_dir = source_dir / "Test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"Expected Train and Test directories under {source_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains files. Re-run with --overwrite to replace it."
        )
    if overwrite and output_dir.exists():
        remove_directory_with_retry(output_dir)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_full_df, bad_train_files = collect_images(train_dir, "train")
    test_df, bad_test_files = collect_images(test_dir, "test")
    if train_full_df.empty or test_df.empty:
        raise ValueError("Both source Train and Test must contain readable images.")
    if train_full_df["label"].value_counts().min() < 2:
        raise ValueError("Each Train class needs at least two readable images.")

    train_df, validation_df = train_test_split(
        train_full_df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_full_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_transform, eval_transform = make_transforms()
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_frames = [
        save_split(train_df, "train", output_dir, train_transform),
        save_split(validation_df, "val", output_dir, eval_transform),
        save_split(test_df, "test", output_dir, eval_transform),
    ]
    metadata = pd.concat(processed_frames, ignore_index=True)
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    classes = sorted(train_full_df["label"].unique())
    with (output_dir / "class_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump({class_name: index for index, class_name in enumerate(classes)}, file, indent=2)
    bad_files = bad_train_files + bad_test_files
    if bad_files:
        pd.DataFrame(bad_files).to_csv(output_dir / "bad_images.csv", index=False)

    summary = metadata.groupby(["split", "label"]).size().reset_index(name="count")
    summary.to_csv(output_dir / "class_counts.csv", index=False)
    print(f"Created: {output_dir}")
    print(summary.groupby("split")["count"].sum().to_string())
    print(f"Bad images skipped: {len(bad_files)}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root; defaults to the parent of src.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing preprocessed output directory before processing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_dataset(arguments.project_dir.resolve(), overwrite=arguments.overwrite)