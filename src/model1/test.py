"""Train a small CNN with dense layers for Vietnamese trash classification.

Run from the project root with:

    python src/model1/test.py

Use ``--max-batches 2`` for a quick smoke test.
"""

from __future__ import annotations

import argparse
import json
import random
from queue import Full, Queue
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_DIR / "data" / "VN_trash_classification"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "model1" / "basic_cnn.pt"
DEFAULT_CACHE_DIR = PROJECT_DIR / "data" / "image_cache_224"
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


class BackgroundBatchIterator:
    """Load CPU batches on a thread while CUDA processes the current batch.

    Windows DataLoader worker *processes* are unreliable on this computer,
    so this uses one lightweight thread instead. PIL image decoding and most
    tensor operations release the GIL, allowing useful overlap with CUDA.
    """

    _DONE = object()

    def __init__(self, loader: DataLoader, prefetch_batches: int = 2) -> None:
        self.loader = loader
        self.queue: Queue[object] = Queue(maxsize=prefetch_batches)
        self.stop_event = Event()
        self.closed = False
        self.thread = Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _put(self, item: object) -> bool:
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            for batch in self.loader:
                if self.stop_event.is_set() or not self._put(batch):
                    return
        except BaseException as error:
            self._put(error)
        finally:
            self._put(self._DONE)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is self._DONE:
            self.close()
            raise StopIteration
        if isinstance(item, BaseException):
            self.close()
            raise item
        return item

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.stop_event.set()
            self.thread.join(timeout=1)


def batch_iterator(
    loader: DataLoader, use_thread_prefetch: bool
):
    """Return a normal iterator or an iterator with CPU/GPU overlap."""
    if use_thread_prefetch:
        return BackgroundBatchIterator(loader)
    return iter(loader)


def cache_file_paths(cache_dir: Path, split: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"{split.lower()}_images.npy",
        cache_dir / f"{split.lower()}_labels.npy",
    )


def cache_metadata_path(cache_dir: Path) -> Path:
    return cache_dir / "metadata.json"


def read_cache_metadata(cache_dir: Path, image_size: int) -> dict | None:
    metadata_file = cache_metadata_path(cache_dir)
    if not metadata_file.is_file():
        return None
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("image_size") != image_size:
        return None
    required_files = [
        *cache_file_paths(cache_dir, "Train"),
        *cache_file_paths(cache_dir, "Test"),
    ]
    if not all(path.is_file() for path in required_files):
        return None
    return metadata


class CachedImageDataset(Dataset):
    """Memory-mapped, resized uint8 images with no CPU-side transforms."""

    def __init__(self, cache_dir: Path, split: str, metadata: dict) -> None:
        image_file, label_file = cache_file_paths(cache_dir, split)
        self.images = np.load(image_file, mmap_mode="r")
        self.labels = np.load(label_file, mmap_mode="r")
        self.classes = list(metadata["classes"])
        self.targets = self.labels.tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # Copying one image makes the Tensor writable and lets DataLoader build
        # a contiguous batch without modifying the memory-mapped cache.
        image = torch.from_numpy(np.array(self.images[index], copy=True))
        return image, int(self.labels[index])


def build_image_cache(cache_dir: Path, image_size: int) -> None:
    """Create resized CHW uint8 arrays once, then reuse them every epoch."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_source = datasets.ImageFolder(DATASET_DIR / "Train")
    test_source = datasets.ImageFolder(DATASET_DIR / "Test")
    if train_source.classes != test_source.classes:
        raise ValueError("Class của Train và Test không giống nhau.")

    total_images = len(train_source) + len(test_source)
    estimated_gib = total_images * 3 * image_size**2 / 1024**3
    print(
        f"Building image cache at {cache_dir} "
        f"(~{estimated_gib:.2f} GiB of image data)..."
    )
    for split, source in (("Train", train_source), ("Test", test_source)):
        image_file, label_file = cache_file_paths(cache_dir, split)
        image_array = np.lib.format.open_memmap(
            image_file,
            mode="w+",
            dtype=np.uint8,
            shape=(len(source), 3, image_size, image_size),
        )
        label_array = np.lib.format.open_memmap(
            label_file,
            mode="w+",
            dtype=np.int64,
            shape=(len(source),),
        )
        for index, (image_path, label) in enumerate(
            tqdm(source.samples, desc=f"Caching {split}", unit="image")
        ):
            with Image.open(image_path) as image:
                resized = image.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.BILINEAR
                )
                image_array[index] = np.asarray(resized, dtype=np.uint8).transpose(
                    2, 0, 1
                )
            label_array[index] = label
        image_array.flush()
        label_array.flush()

    cache_metadata_path(cache_dir).write_text(
        json.dumps(
            {
                "image_size": image_size,
                "classes": train_source.classes,
                "train_images": len(train_source),
                "test_images": len(test_source),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Image cache is ready.")


class BasicCNN(nn.Module):
    """A simple CNN followed by dense (fully connected) layers."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloaders(args: argparse.Namespace, device: torch.device):
    train_dir = DATASET_DIR / "Train"
    test_dir = DATASET_DIR / "Test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy dataset. Cần có: {train_dir} và {test_dir}"
        )

    cache_metadata = (
        read_cache_metadata(args.cache_dir, args.image_size)
        if args.image_cache
        else None
    )
    images_need_gpu_preprocess = cache_metadata is not None
    if cache_metadata is not None:
        train_with_augmentation = CachedImageDataset(
            args.cache_dir, "Train", cache_metadata
        )
        # The cached data is raw uint8. Train augmentation happens later on
        # the GPU, so validation can reuse the same memory-mapped images.
        train_for_validation = train_with_augmentation
        test_dataset = CachedImageDataset(args.cache_dir, "Test", cache_metadata)
        print(f"Image cache: enabled ({args.cache_dir})")
    else:
        if args.image_cache:
            print(
                "Image cache: not found for this image size; "
                "using raw JPEG files. Run with --build-image-cache first."
            )
        normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        train_transform = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_with_augmentation = datasets.ImageFolder(
            train_dir, transform=train_transform
        )
        train_for_validation = datasets.ImageFolder(
            train_dir, transform=eval_transform
        )
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    if train_with_augmentation.classes != test_dataset.classes:
        raise ValueError(
            "Class của Train và Test không giống nhau: "
            f"{train_with_augmentation.classes} != {test_dataset.classes}"
        )

    all_indices = np.arange(len(train_with_augmentation))
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=train_with_augmentation.targets,
    )

    train_dataset = Subset(train_with_augmentation, train_indices.tolist())
    validation_dataset = Subset(
        train_for_validation, validation_indices.tolist()
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options.update(
            # On Windows, keeping Train workers alive while Validation starts
            # can exhaust memory. One prefetched batch is enough to overlap
            # loading with CUDA work without retaining worker processes.
            prefetch_factor=1,
        )
    # A constant, large training batch gives cuDNN/Tensor Cores regular work.
    # The final incomplete batch is skipped only for training; validation and
    # test still use every image.
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=device.type == "cuda",
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return (
        train_loader,
        validation_loader,
        test_loader,
        train_with_augmentation.classes,
        images_need_gpu_preprocess,
    )


def prepare_images(
    images: torch.Tensor,
    device: torch.device,
    images_need_gpu_preprocess: bool,
    training: bool,
) -> torch.Tensor:
    """Move a batch to CUDA and preprocess cached uint8 images on the GPU."""
    images = images.to(device, non_blocking=device.type == "cuda")
    if images_need_gpu_preprocess:
        images = images.to(dtype=torch.float32).div_(255.0)
        if training:
            flip_mask = torch.rand(images.size(0), device=device) < 0.5
            images[flip_mask] = images[flip_mask].flip(dims=(3,))
        mean = images.new_tensor(NORMALIZE_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(NORMALIZE_STD).view(1, 3, 1, 1)
        images.sub_(mean).div_(std)
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
    scaler,
    use_amp: bool,
    use_thread_prefetch: bool,
    images_need_gpu_preprocess: bool,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    batches = batch_iterator(loader, use_thread_prefetch)
    try:
        for batch_index, (images, labels) in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = prepare_images(
                images, device, images_need_gpu_preprocess, training=True
            )
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += batch_size
    finally:
        close = getattr(batches, "close", None)
        if close is not None:
            close()

    if total_items == 0:
        raise RuntimeError("DataLoader không có batch nào để train.")
    return total_loss / total_items, total_correct / total_items


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    use_thread_prefetch: bool,
    images_need_gpu_preprocess: bool,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    batches = batch_iterator(loader, use_thread_prefetch)
    try:
        for batch_index, (images, labels) in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = prepare_images(
                images, device, images_need_gpu_preprocess, training=False
            )
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += batch_size
    finally:
        close = getattr(batches, "close", None)
        if close is not None:
            close()

    if total_items == 0:
        raise RuntimeError("DataLoader không có batch nào để evaluate.")
    return total_loss / total_items, total_correct / total_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Stable default for the RTX 4060 Laptop GPU (8 GB VRAM).",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--image-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the resized uint8 cache when it is available.",
    )
    parser.add_argument(
        "--build-image-cache",
        action="store_true",
        help="Create or rebuild the resized image cache before training.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory that contains the resized image cache.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop after this many epochs without validation-loss improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-3,
        help="Minimum validation-loss reduction required to reset early stopping.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of CPU workers used to load images.",
    )
    parser.add_argument(
        "--thread-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepare the next batch on a background thread when num-workers is 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Use CUDA when available; otherwise use CPU.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches in each phase for a quick smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        raise ValueError("epochs, batch-size và image-size phải lớn hơn 0.")
    if not 0 < args.validation_size < 1:
        raise ValueError("validation-size phải nằm trong khoảng (0, 1).")
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience phải lớn hơn hoặc bằng 0.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early-stopping-min-delta phải lớn hơn hoặc bằng 0.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Bạn yêu cầu CUDA nhưng PyTorch không nhận GPU.")
    if args.build_image_cache:
        build_image_cache(args.cache_dir, args.image_size)

    seed_everything(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = device.type == "cuda"
    use_thread_prefetch = (
        args.thread_prefetch and args.num_workers == 0 and device.type == "cuda"
    )
    if device.type == "cuda":
        # `cuda:0` is the RTX 4060 detected by PyTorch on this computer.
        # CUDA never routes this workload through the AMD integrated GPU.
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    (
        train_loader,
        validation_loader,
        test_loader,
        classes,
        images_need_gpu_preprocess,
    ) = make_dataloaders(args, device)
    model = BasicCNN(num_classes=len(classes)).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    try:
        optimizer = Adam(
            model.parameters(), lr=args.learning_rate, fused=device.type == "cuda"
        )
    except (TypeError, RuntimeError):
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA GPU: {torch.cuda.get_device_name(device)}")
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")
    print(
        "Background prefetch: "
        f"{'enabled' if use_thread_prefetch else 'disabled'}"
    )
    print(f"Classes ({len(classes)}): {classes}")
    print(
        f"Samples: train={len(train_loader.dataset)}, "
        f"validation={len(validation_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )
    if args.early_stopping_patience == 0:
        print("Early stopping: disabled")
    else:
        print(
            "Early stopping: "
            f"patience={args.early_stopping_patience}, "
            f"min_delta={args.early_stopping_min_delta}"
        )

    training_started_at = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            use_thread_prefetch,
            images_need_gpu_preprocess,
            args.max_batches,
        )
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            use_amp,
            use_thread_prefetch,
            images_need_gpu_preprocess,
            args.max_batches,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.2%} | "
            f"val loss={validation_loss:.4f}, acc={validation_accuracy:.2%}"
        )

        improved = (
            validation_loss
            < best_validation_loss - args.early_stopping_min_delta
        )
        if args.max_batches is None and improved:
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": args.image_size,
                    "model_name": "BasicCNN",
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                },
                CHECKPOINT_PATH,
            )
            print(f"Saved checkpoint: {CHECKPOINT_PATH}")
        elif args.max_batches is None and args.early_stopping_patience > 0:
            epochs_without_improvement += 1
            print(
                "Early stopping: validation loss did not improve "
                f"({epochs_without_improvement}/{args.early_stopping_patience})"
            )
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
        use_amp,
        use_thread_prefetch,
        images_need_gpu_preprocess,
        max_batches=None,
    )
    print(f"Test loss={test_loss:.4f}, accuracy={test_accuracy:.2%}")
    if best_epoch:
        print(
            f"Best checkpoint: epoch={best_epoch}, "
            f"val loss={best_validation_loss:.4f}, "
            f"val acc={best_validation_accuracy:.2%}"
        )
    print(f"Total elapsed={perf_counter() - training_started_at:.2f} seconds")
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"Peak CUDA allocated={peak_allocated:.2f} GB")
        print(f"Peak CUDA reserved={peak_reserved:.2f} GB")


if __name__ == "__main__":
    main()
