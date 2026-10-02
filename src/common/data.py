from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event, Thread

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_NORMALIZATION_TENSORS = {}


class CachedImageDataset(Dataset):
    """Dataset backed by the uint8 .npy cache described in docs/model.md."""

    def __init__(self, images_path: Path, labels_path: Path, indices=None):
        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.indices = np.arange(len(self.labels)) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image_index = self.indices[index]
        # The copy makes the tensor writable. Augmentation is applied later on GPU.
        return torch.from_numpy(self.images[image_index].copy()), int(self.labels[image_index])


def make_loaders(cache_dir: Path, batch_size: int, workers: int, seed: int):
    """Make the fixed 85% train / 15% stratified validation split and test loader."""
    if sys.platform == "win32" and workers > 0:
        # Windows uses spawn, which must pickle every Dataset.  With the large
        # memory-mapped image cache, several simultaneous loader processes are
        # unreliable and can raise "pickle data was truncated".  A thread
        # prefetcher below keeps loading asynchronous without this risk.
        print("Windows detected: using workers=0 and thread prefetch (safe for the image cache).")
        workers = 0

    labels = np.load(cache_dir / "train_labels.npy")
    all_indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=0.15,
        random_state=seed,
        stratify=labels,
    )

    train_data = CachedImageDataset(cache_dir / "train_images.npy", cache_dir / "train_labels.npy", train_indices)
    val_data = CachedImageDataset(cache_dir / "train_images.npy", cache_dir / "train_labels.npy", val_indices)
    test_data = CachedImageDataset(cache_dir / "test_images.npy", cache_dir / "test_labels.npy")
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        # Keep workers alive and prepare the next batches while GPU is training.
        common["persistent_workers"] = True
        common["prefetch_factor"] = 2
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_data, shuffle=True, generator=generator, **common),
        DataLoader(val_data, shuffle=False, **common),
        DataLoader(test_data, shuffle=False, **common),
    )


class BackgroundBatchIterator:
    """Prefetch DataLoader batches with one thread, without Windows multiprocessing.

    The thread loads the next CPU/pinned-memory batch while CUDA trains on the
    current one.  It is deliberately used only with ``num_workers=0``.
    """

    def __init__(self, loader, prefetch_batches: int = 2):
        self._queue = Queue(maxsize=prefetch_batches)
        self._end = object()
        self._stop = Event()
        self._error = None
        self._thread = Thread(target=self._load, args=(loader,), daemon=True)
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def _load(self, loader):
        try:
            for batch in loader:
                if self._stop.is_set() or not self._put(batch):
                    return
        except BaseException as error:  # Forward Dataset/DataLoader errors to the main thread.
            self._error = error
        finally:
            self._put(self._end)

    def __iter__(self):
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except Empty:
                if not self._thread.is_alive():
                    if self._error is not None:
                        raise RuntimeError("Background batch loader failed.") from self._error
                    return
                continue
            if item is self._end:
                if self._error is not None:
                    raise RuntimeError("Background batch loader failed.") from self._error
                return
            yield item

    def close(self):
        """Stop promptly when an epoch exits early or uses --max-batches."""
        self._stop.set()
        self._thread.join(timeout=1)


def prefetch_batches(loader):
    """Return a safe asynchronous iterator for a single-process DataLoader."""
    if loader.num_workers == 0:
        return BackgroundBatchIterator(loader)
    return iter(loader)


def prepare_images(
    images: torch.Tensor,
    device: torch.device,
    image_size: int,
    training: bool,
    channels_last: bool,
):
    """Move [B,3,224,224] uint8 images to GPU, augment, resize and normalize."""
    images = images.to(device, dtype=torch.float32, non_blocking=True).div_(255.0)

    if image_size != 224:
        images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)

    if training:
        flip_mask = torch.rand(images.size(0), device=device) < 0.5
        images[flip_mask] = torch.flip(images[flip_mask], dims=(3,))

    device_key = (device.type, device.index)
    if device_key not in _NORMALIZATION_TENSORS:
        _NORMALIZATION_TENSORS[device_key] = (
            torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1),
            torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1),
        )
    mean, std = _NORMALIZATION_TENSORS[device_key]
    images = images.sub_(mean).div_(std)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images
