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
    transforms.Resize((128, 128)),
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


# 4. Tạo MLP
class MLP(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 128 * 128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, number_of_classes)
        )

    def forward(self, images):
        return self.layers(images)


model = MLP(len(train_dataset.classes)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 5. Train và Validation
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

    print(
        f"Epoch {epoch + 1}/50 | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2%} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2%}"
    )


# 6. Test
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
