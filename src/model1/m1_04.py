"""M1-04: CNN 3 blocks với Global Average Pooling (GAP).

Chạy từ thư mục gốc của project:
    python src/model1/m1_04.py
"""

import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from metrics import calculate_metrics


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "VN_trash_classification_preprocessing"

# 224x224 sau ba MaxPool sẽ thành feature map 28x28 trước GAP.
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_loaders(device):
    """Đọc các thư mục train/val/test giống pipeline của Model 2 và Model 3."""
    for split in ("train", "val", "test"):
        if not (DATA_DIR / split).is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy {DATA_DIR / split}. "
                "Hãy chạy python src/preprocessing.py --overwrite trước."
            )

    # Train có augmentation nhẹ; validation/test giữ cố định để so sánh đáng tin cậy.
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dataset = ImageFolder(DATA_DIR / "train", transform=train_transform)
    validation_dataset = ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_dataset = ImageFolder(DATA_DIR / "test", transform=eval_transform)

    if train_dataset.classes != validation_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("Các class trong train, val và test không giống nhau.")

    # num_workers=0 tránh lỗi worker trên Windows; pin_memory hỗ trợ chuyển dữ liệu sang CUDA.
    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_options),
        DataLoader(validation_dataset, shuffle=False, **loader_options),
        DataLoader(test_dataset, shuffle=False, **loader_options),
        train_dataset.classes,
    )


class CNNWithGAP(nn.Module):
    """CNN 3 block + GAP: [B, 3, 224, 224] -> [B, 128] -> logits [B, 9]."""

    def __init__(self, number_of_classes):
        super().__init__()

        # Ba block tạo feature map [B, 128, 28, 28] từ ảnh RGB 224x224.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # GAP lấy trung bình mỗi feature map: [B, 128, 28, 28] -> [B, 128].
        # Không thêm Dropout/Dense hidden để chỉ so sánh Dense head của M1-03 với GAP.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def train_one_epoch(model, loader, loss_function, optimizer, device):
    """Chạy forward, backward và cập nhật weight cho một epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # Gradient đi ngược từ loss, sau đó Adam cập nhật weight cho batch này.
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_images += labels.size(0)

    return total_loss / total_images, total_correct / total_images


def evaluate(model, loader, loss_function, device):
    """Đánh giá validation hoặc test mà không tính gradient."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_images = 0
    true_labels = []
    predicted_labels = []

    # Không tính gradient trong validation/test vì chỉ cần dự đoán và loss.
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            outputs = model(images)
            loss = loss_function(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_images += labels.size(0)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    return total_loss / total_images, total_correct / total_images, true_labels, predicted_labels


def main():
    started = time.perf_counter()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data folder: {DATA_DIR}")

    train_loader, validation_loader, test_loader, class_names = make_loaders(device)
    model = CNNWithGAP(len(class_names)).to(device)
    print(f"M1-04 | Dropout=không dùng (GAP head) | parameters={sum(p.numel() for p in model.parameters()):,}")

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_validation_loss = float("inf")
    best_model = None
    best_epoch = 0
    history = []
    result_dir = PROJECT_DIR / "reports" / "model1_preprocessed"
    result_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        epoch_started = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        validation_loss, validation_accuracy, _, _ = evaluate(
            model, validation_loader, loss_function, device
        )

        # Chỉ validation loss được dùng để giữ checkpoint tốt nhất.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        epoch_seconds = time.perf_counter() - epoch_started
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "seconds": epoch_seconds,
        })
        with (result_dir / "M1-04_history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f}, train acc: {train_accuracy:.2%} | "
            f"val loss: {validation_loss:.4f}, val acc: {validation_accuracy:.2%} | "
            f"time: {epoch_seconds:.2f}s"
        )

    # Dùng checkpoint validation tốt nhất để test một lần ở cuối.
    model.load_state_dict(best_model)
    train_validation_seconds = sum(row["seconds"] for row in history)
    test_started = time.perf_counter()
    test_loss, test_accuracy, true_labels, predicted_labels = evaluate(
        model, test_loader, loss_function, device
    )
    metrics = calculate_metrics(true_labels, predicted_labels)
    test_seconds = time.perf_counter() - test_started
    torch.save(best_model, result_dir / "M1-04_best.pt")
    elapsed_seconds = time.perf_counter() - started
    result = {
        "model": "M1-04",
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "data_dir": str(DATA_DIR),
        "classes": class_names,
        "split_sizes": {
            "train": len(train_loader.dataset),
            "val": len(validation_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": SEED,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "train_validation_seconds": train_validation_seconds,
        "test_seconds": test_seconds,
        "elapsed_seconds": elapsed_seconds,
        "test_loss": test_loss,
        "metrics": metrics,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "history": history,
    }
    with (result_dir / "M1-04_result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"Best epoch: {best_epoch} | Total time: {elapsed_seconds:.2f}s")

    print("\nM1-04 test result")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
