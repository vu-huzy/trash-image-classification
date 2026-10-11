from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

#Chọn model: 3a_frozen_vgg16 | 3b_frozen_resnet50 | 3b_frozen_efficientnet_b0
#          | 3c_lora_resnet50 | 3d_full_finetune_resnet50
MODEL_NAME = '3d_full_finetune_resnet50'

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


#Tạo kiến trúc model (giống hệt notebook), trọng số sẽ được nạp từ file .pt
class LoRAConv2d(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.lora_down = nn.Conv2d(
            base.in_channels,
            rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)

    def forward(self, images):
        return self.base(images) + self.scaling * self.lora_up(self.lora_down(images))


class TransferModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images):
        return self.head(self.backbone(images))


if 'vgg16' in MODEL_NAME:
    backbone = models.vgg16(weights=None)
    backbone.classifier[6] = nn.Identity()
    feature_dim = 4096
elif 'resnet50' in MODEL_NAME:
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    feature_dim = 2048
else:
    backbone = models.efficientnet_b0(weights=None)
    backbone.classifier = nn.Identity()
    feature_dim = 1280

#3C: thay Conv2d của layer3 và layer4 bằng LoRAConv2d (rank 8, alpha 16)
if 'lora' in MODEL_NAME:
    for name, module in list(backbone.named_modules()):
        if name.startswith(('layer3', 'layer4')):
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.Conv2d):
                    setattr(module, child_name, LoRAConv2d(child, 8, 16.0))

head = nn.Sequential(
    nn.Linear(feature_dim, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, len(class_names)),
)
model = TransferModel(backbone, head)

#Load model: checkpoints/<MODEL_NAME>.pt
#Notebook lưu state_dict, code cũ (train.py trong thư mục này) lưu dict {'model_state_dict': state_dict, ...}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load(HERE / 'checkpoints' / f'{MODEL_NAME}.pt', map_location='cpu')
model.load_state_dict(state.get('model_state_dict', state))
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
print('Confusion Matrix:')
print(confusion_matrix(true_labels, pred_labels))
