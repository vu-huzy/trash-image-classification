import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from metrics import calculate_metrics


# 1. Cấu hình
DATA_DIR = "data/VN_trash_classification_preprocessing"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# 2. Xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3. Load dữ liệu
train_dataset = ImageFolder(DATA_DIR + "/train", transform=transform)
val_dataset = ImageFolder(DATA_DIR + "/val", transform=transform)
test_dataset = ImageFolder(DATA_DIR + "/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# 4. Tạo CNN 3 blocks với Global Average Pooling
class CNN(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        self.layers = nn.Sequential(
            # [B, 3, 224, 224] -> [B, 32, 112, 112]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [B, 32, 112, 112] -> [B, 64, 56, 56]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [B, 64, 56, 56] -> [B, 128, 28, 28]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # GAP: [B, 128, 28, 28] -> [B, 128] -> [B, number_of_classes]
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, number_of_classes)
        )

    def forward(self, images):
        return self.layers(images)


model = CNN(len(train_dataset.classes)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 5. Train và Validation
best_val_loss = float("inf")
best_weights = None
best_epoch = 0
patience = 8
epochs_without_improvement = 0

for epoch in range(50):
    model.train()
    train_loss = 0
    train_true_labels = []
    train_predicted_labels = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)

        train_true_labels.extend(labels.cpu().tolist())
        train_predicted_labels.extend(predictions.cpu().tolist())

    train_loss /= len(train_dataset)
    train_metrics = calculate_metrics(train_true_labels, train_predicted_labels)
    train_accuracy = train_metrics["accuracy"]

    model.eval()
    val_loss = 0
    val_true_labels = []
    val_predicted_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            val_true_labels.extend(labels.cpu().tolist())
            val_predicted_labels.extend(predictions.cpu().tolist())

    val_loss /= len(val_dataset)
    val_metrics = calculate_metrics(val_true_labels, val_predicted_labels)
    val_accuracy = val_metrics["accuracy"]

    # Sao chép trọng số sang RAM để các epoch sau không ghi đè lên bản tốt nhất.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        best_epoch = epoch + 1
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch + 1}/50 | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2%} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2%}"
    )

    # Dừng khi validation loss không giảm trong 8 epoch liên tiếp.
    if epochs_without_improvement >= patience:
        print(f"Early stopping tại epoch {epoch + 1}")
        break


# 6. Test
model.load_state_dict(best_weights)
print(f"Best epoch: {best_epoch} | Best Val Loss: {best_val_loss:.6f}")
model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())

metrics = calculate_metrics(true_labels, predicted_labels)

print("\nTest Result")
print(f"Test Accuracy: {metrics['accuracy']:.2%}")
print(f"Macro Precision: {metrics['macro_precision']:.4f}")
print(f"Macro Recall: {metrics['macro_recall']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
