import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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

#Tạo kiến trúc model từ py/models.py (giống hệt kiến trúc trong notebook)
sys.path.insert(0, str(HERE / 'py'))
from models import build_models

REGISTRY_NAME = {
    'cnn_sequential': 'cnn_sequential',
    'cnn_parallel': 'cnn_parallel',
    'vgg_sequential': 'vgg_sequential',
    'vgg_parallel': 'vgg_parallel',
    'resnet': 'resnet18',
}
model = build_models(len(class_names))[REGISTRY_NAME[MODEL_NAME]]

#Load model: checkpoints/<MODEL_NAME>.pt
#Notebook lưu state_dict, py/train.py lưu dict {'model': state_dict, ...}
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
print('Confusion Matrix:')
print(confusion_matrix(true_labels, pred_labels))
