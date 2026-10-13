import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, classification_report

#Chọn model: m1_01 (MLP) | m1_02 (CNN 2 block) | m1_03 (CNN 3 block) | m1_04 (CNN 3 block + GAP)
MODEL_NAME = 'm1_03'

DATA_DIR = '../../data/VN_trash_classification_preprocessing'

#Đọc dữ liệu từ thư mục test (m1_01 dùng ảnh 128x128, các model còn lại dùng 224x224)
image_size = 128 if MODEL_NAME == 'm1_01' else 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_dataset = ImageFolder(DATA_DIR + '/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
class_names = test_dataset.classes
number_of_classes = len(class_names)


#Tạo kiến trúc model (giống hệt notebook m1_01 ... m1_04)
class Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


def conv_block(in_channels, out_channels):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    ]


if MODEL_NAME == 'm1_01':
    layers = [
        nn.Flatten(),
        nn.Linear(3 * 128 * 128, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, number_of_classes),
    ]
elif MODEL_NAME == 'm1_02':
    layers = conv_block(3, 32) + conv_block(32, 64) + [
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, number_of_classes),
    ]
elif MODEL_NAME == 'm1_03':
    layers = conv_block(3, 32) + conv_block(32, 64) + conv_block(64, 128) + [
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, number_of_classes),
    ]
else:
    layers = conv_block(3, 32) + conv_block(32, 64) + conv_block(64, 128) + [
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, number_of_classes),
    ]

#Load checkpoint: checkpoints/<MODEL_NAME>.pt (do notebook lưu ở dòng cuối)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(layers)
model.load_state_dict(torch.load('checkpoints/' + MODEL_NAME + '.pt', map_location=device))
model = model.to(device)
model.eval()

#Chạy test
true_labels = []
pred_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        true_labels += labels.tolist()
        pred_labels += outputs.argmax(dim=1).cpu().tolist()

#Kết quả trên tập test (classification_report đã có precision/recall/F1 từng lớp, macro avg và weighted avg)
accuracy = accuracy_score(true_labels, pred_labels)
print('Model:', MODEL_NAME)
print('Test Accuracy:', f'{accuracy:.2%}')
print()
print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4, zero_division=0))
