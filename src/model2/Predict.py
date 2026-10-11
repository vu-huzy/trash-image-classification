from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, precision_recall_fscore_support

#Chọn model: cnn_sequential | cnn_parallel | vgg_sequential | vgg_parallel | resnet
MODEL_NAME = 'cnn_sequential'

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / 'data' / 'VN_trash_classification_preprocessing'

#Đọc dữ liệu từ thư mục test
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_dataset = ImageFolder(DATA_DIR / 'test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes


#Kiến trúc các model (giống hệt notebook trong thư mục notebook/)
class CNNSequential(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, number_of_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x))


class CNNParallel(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            ResidualConvBlock(96, 64),
            ResidualConvBlock(64, 128),
            ResidualConvBlock(128, 192),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, number_of_classes),
        )

    def forward(self, images):
        x = torch.cat([self.branch1(images), self.branch2(images), self.branch3(images)], dim=1)
        return self.classifier(self.features(x))


class VGGSequential(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        layers = []
        in_channels = 3
        for width, count in ((64, 2), (128, 2), (256, 3), (512, 3), (512, 3)):
            for _ in range(count):
                layers += [
                    nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ]
                in_channels = width
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class VGGParallel(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 192 kênh sau khi nối, rồi 5 stage kiểu VGG: 1/2/3/3/3 conv, mỗi stage có MaxPool
        layers = [nn.Conv2d(192, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        in_channels = 64
        for width, count in ((128, 2), (256, 3), (256, 3), (256, 3)):
            for _ in range(count):
                layers += [
                    nn.Conv2d(in_channels, width, 3, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ]
                in_channels = width
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images):
        x = torch.cat([self.branch1(images), self.branch2(images), self.branch3(images)], dim=1)
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        output = self.conv2(self.conv1(x))
        return self.relu(output + shortcut)


class ResNet(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(BasicBlock(64, 64, 1), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images):
        x = self.stem(images)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.classifier(self.pool(x))


MODELS = {
    'cnn_sequential': CNNSequential,
    'cnn_parallel': CNNParallel,
    'vgg_sequential': VGGSequential,
    'vgg_parallel': VGGParallel,
    'resnet': ResNet,
}
model = MODELS[MODEL_NAME](len(class_names))

#Load model: checkpoints/<MODEL_NAME>.pt (do notebook lưu ở dòng cuối)
#Các file cũ trong checkpoints/archive/ là dict {'model': state_dict, ...} nên lấy theo khóa 'model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load(HERE / 'checkpoints' / f'{MODEL_NAME}.pt', map_location='cpu')
model.load_state_dict(state.get('model', state))
model = model.to(device).eval()

#Dự báo nhãn của tập test
true_labels = []
pred_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        true_labels += labels.tolist()
        pred_labels += outputs.argmax(dim=1).cpu().tolist()

#In 10 ảnh đầu tiên: nhãn thật và nhãn dự đoán
print('Model:', MODEL_NAME)
for i in range(10):
    image_name = Path(test_dataset.samples[i][0]).name
    print(image_name, '| thật:', class_names[true_labels[i]], '| dự đoán:', class_names[pred_labels[i]])

#Kiểm tra kết quả trên toàn bộ tập test
accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average='macro', zero_division=0
)
_, _, weighted_f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average='weighted', zero_division=0
)
print()
print('Test Accuracy:', f'{accuracy:.2%}')
print('Macro Precision:', f'{macro_precision:.4f}')
print('Macro Recall:', f'{macro_recall:.4f}')
print('Macro F1:', f'{macro_f1:.4f}')
print('Weighted F1:', f'{weighted_f1:.4f}')
print()
print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4, zero_division=0))
