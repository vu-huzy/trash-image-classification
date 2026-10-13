# Phân loại rác thải Việt Nam bằng ảnh

Bài tập phân loại ảnh rác thải thành 9 lớp bằng học sâu (PyTorch / torchvision). Dự án so sánh ba hướng tiếp cận
trên cùng một tập dữ liệu và cùng một quy trình đánh giá:

1. **Model 1**: mạng nơ-ron cơ bản (MLP và CNN nhỏ) huấn luyện từ đầu.
2. **Model 2**: các kiến trúc CNN / VGG / ResNet tự viết (dạng tuần tự và dạng nhánh song song), huấn luyện từ đầu.
3. **Model 3**: transfer learning từ trọng số ImageNet: đóng băng backbone, LoRA tự viết, fine-tune toàn bộ.

Mô hình tốt nhất là **ResNet50 + LoRA (3C) với 96.53% accuracy và macro F1 0.9643** trên tập test (xem mục 7).

## 1. Dữ liệu

- **9 lớp**: `Alu`, `Carton`, `Foam_box`, `Milk_box`, `Other`, `PET`, `Paper`, `Paper_cup`, `Plastic_cup`.
- **Nguồn**: `data/VN_trash_classification/{Train,Test}/<Lớp>/*.jpg`, nhãn lấy từ tên thư mục cha của ảnh.
  Thư mục `data/` không nằm trong repo (đã `.gitignore`) và repo không có script tải dữ liệu; cần đặt dữ liệu vào
  đúng vị trí trên trước khi chạy.
- **Chia tập**: `Train` gốc (11.227 ảnh) được chia 85% / 15% thành `train` / `val` theo `train_test_split(stratify=nhãn,
  random_state=42)`; `Test` gốc (864 ảnh) được giữ nguyên để đánh giá cuối.

Số ảnh sau khi tiền xử lý (`data/VN_trash_classification_preprocessing/`):

| Lớp | train | val | test |
| --- | ---: | ---: | ---: |
| Alu | 1.059 | 187 | 99 |
| Carton | 1.093 | 193 | 108 |
| Foam_box | 1.061 | 187 | 87 |
| Milk_box | 965 | 170 | 101 |
| Other | 1.124 | 198 | 50 |
| PET | 1.069 | 189 | 102 |
| Paper | 992 | 175 | 106 |
| Paper_cup | 1.095 | 194 | 105 |
| Plastic_cup | 1.084 | 192 | 106 |
| **Tổng** | **9.542** | **1.685** | **864** |

Tập train và val khá cân bằng (khoảng 1.000 và 180 ảnh mỗi lớp); tập test lệch hơn, lớp `Other` chỉ có 50 ảnh.

## 2. Cấu trúc dự án

```
trash-image-classification/
├── README.md                     # tài liệu này (file markdown duy nhất)
├── requirements.txt              # thư viện và phiên bản
├── .gitignore
├── EDA/
│   ├── EDA.ipynb                 # khám phá dữ liệu: phân bố lớp, ảnh lỗi, kích thước ảnh, ảnh mẫu
│   └── preprocessing.ipynb       # bản notebook của bước tiền xử lý (chạy trong bộ nhớ, trực quan hóa)
├── data/                         # dữ liệu (không commit)
│   ├── VN_trash_classification/                 # dữ liệu gốc: Train/, Test/
│   └── VN_trash_classification_preprocessing/   # dữ liệu sau tiền xử lý: train/, val/, test/ + metadata
└── src/
    ├── preprocessing.py          # tiền xử lý: kiểm tra ảnh, chia train/val, ghi ảnh ra đĩa
    ├── model1/
    │   ├── m1_01.ipynb ... m1_04.ipynb   # 4 notebook MLP / CNN, mỗi notebook là một pipeline đầy đủ
    │   ├── metrics.py                    # hàm calculate_metrics (accuracy, macro P/R/F1, weighted F1)
    │   ├── Predict.py                    # nạp checkpoint và chạy test
    │   └── checkpoints/                  # trọng số .pt do notebook lưu (không commit)
    ├── model2/
    │   ├── notebook/
    │   │   ├── cnn_sequential.ipynb, cnn_parallel.ipynb
    │   │   ├── vgg_sequential.ipynb, vgg_parallel.ipynb
    │   │   ├── resnet.ipynb
    │   │   └── metrics.py                # hàm calculate_metrics, import trực tiếp từ notebook
    │   ├── Predict.py
    │   └── checkpoints/                  # <model>.pt; archive/ giữ các bản cũ (không dùng)
    └── model3/
        ├── notebook/
        │   ├── 3a_frozen_vgg16.ipynb
        │   ├── 3b_frozen_resnet50.ipynb, 3b_frozen_efficientnet_b0.ipynb
        │   ├── 3c_lora_resnet50.ipynb
        │   └── 3d_full_finetune_resnet50.ipynb
        ├── Predict.py
        ├── pretrained/               # trọng số ImageNet tải về (torch.hub), không commit
        └── checkpoints/              # <notebook>.pt (không commit)
```

### Từng file có gì

| File | Nội dung |
| --- | --- |
| `src/preprocessing.py` | Quét ảnh bằng `PIL.Image.verify()` để loại ảnh hỏng, chia train/val, áp transform và **ghi ảnh JPEG (q95) ra `data/VN_trash_classification_preprocessing/`** cùng `metadata.csv`, `class_to_idx.json`, `class_counts.csv`, `bad_images.csv`. Tham số: `--project-dir`, `--overwrite`. Có xử lý lỗi khóa file khi xóa thư mục trên Windows. |
| `EDA/EDA.ipynb` | Phân tích khám phá: phân bố số ảnh theo lớp, ảnh lỗi, kích thước ảnh, ảnh minh họa. |
| `EDA/preprocessing.ipynb` | Cùng logic tiền xử lý nhưng chạy trong bộ nhớ: dựng `Dataset`/`DataLoader`, tính class weights, lưu metadata. |
| `src/model1/m1_0x.ipynb` | Mỗi notebook gồm: cấu hình, transform, nạp dữ liệu (`ImageFolder`), định nghĩa model, huấn luyện + validation với early stopping, test, lưu `state_dict`. |
| `src/model2/notebook/*.ipynb` | Cùng bố cục như Model 1 cho 5 kiến trúc tự viết; thêm class weight trong loss và cosine scheduler. |
| `src/model3/notebook/*.ipynb` | Cùng bố cục, backbone ImageNet + head MLP; riêng 3C có lớp `LoRAConv2d` tự viết. |
| `src/model*/metrics.py` | `calculate_metrics(true_labels, predicted_labels)`: accuracy, macro precision / recall / F1, weighted F1. Model 3 tính trực tiếp bằng scikit-learn trong notebook. |
| `src/model*/Predict.py` | Chọn `MODEL_NAME` ở đầu file; dựng lại kiến trúc, nạp `checkpoints/<MODEL_NAME>.pt`, chạy tập test, in accuracy và `classification_report`. |

## 3. Cài đặt

Yêu cầu: Python 3.12, GPU NVIDIA có CUDA (kết quả trong báo cáo chạy trên RTX 5070 12 GB; chạy CPU vẫn được nhưng rất
chậm). `requirements.txt` ghim PyTorch bản CUDA 13 (`torch==2.14.0+cu130`, `torchvision==0.29.0+cu130`).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# đăng ký kernel Jupyter để notebook dùng đúng môi trường này
python -m ipykernel install --user --name trash-classification --display-name "trash-classification (.venv)"
```

- Đặt dữ liệu gốc vào `data/VN_trash_classification/{Train,Test}/<Lớp>/`.
- Model 3 cần trọng số ImageNet; lần chạy đầu notebook tự tải về `src/model3/pretrained/hub/` (cần Internet,
  VGG16 khoảng 530 MB).
- Các notebook ghi `state_dict` vào thư mục `checkpoints/` của từng model (tự tạo nếu chưa có).

## 4. Cách chạy

**Bước 1: tiền xử lý dữ liệu** (từ thư mục gốc dự án):

```powershell
python src/preprocessing.py --overwrite
```

**Bước 2: huấn luyện**: mở từng notebook trong Jupyter (thư mục làm việc là thư mục chứa notebook, vì đường dẫn trong
notebook là đường dẫn tương đối) và chạy từ trên xuống dưới, hoặc chạy bằng dòng lệnh:

```powershell
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 `
    --ExecutePreprocessor.kernel_name=trash-classification src/model1/m1_03.ipynb
```

Thời gian tham khảo trên RTX 5070: Model 1 từ 2 đến 8 phút, Model 2 từ 16 đến 59 phút, Model 3 từ 3 đến 11 phút
mỗi notebook.

**Bước 3: kiểm tra checkpoint trên tập test**: mở `Predict.py` của model tương ứng, đặt `MODEL_NAME`, rồi chạy:

```powershell
python src/model1/Predict.py   # m1_01 | m1_02 | m1_03 | m1_04
python src/model2/Predict.py   # cnn_sequential | cnn_parallel | vgg_sequential | vgg_parallel | resnet
python src/model3/Predict.py   # 3a_frozen_vgg16 | 3b_frozen_resnet50 | 3b_frozen_efficientnet_b0 | 3c_lora_resnet50 | 3d_full_finetune_resnet50
```

`Predict.py` chỉ nạp checkpoint và chạy test, in `Test Accuracy` cùng bảng `classification_report` (precision, recall,
F1, support từng lớp, kèm macro avg và weighted avg).

## 5. Tiền xử lý dữ liệu

- Ảnh hỏng bị loại (`Image.verify()`); ảnh được chuyển sang RGB và lưu JPEG chất lượng 95.
- Ảnh `val` và `test` chỉ được resize về 224×224.
- Ảnh `train` được áp dụng biến đổi ngẫu nhiên **một lần** khi tiền xử lý và lưu ra đĩa: `RandomResizedCrop(224,
  scale 0.8–1.0)`, lật ngang, xoay ±10°, `ColorJitter(0.2)`.
- Khi huấn luyện, notebook nạp ảnh bằng `ImageFolder` và áp thêm: train = Resize + lật ngang + `ColorJitter(0.1)` +
  chuẩn hóa ImageNet; val / test = Resize + chuẩn hóa ImageNet. `m1_01` dùng ảnh 128×128, các notebook còn lại 224×224.

## 6. Mô hình và quy trình huấn luyện

### 6.1 Quy trình chung (cả 14 notebook)

1. Nạp dữ liệu bằng `ImageFolder` trên `train/`, `val/`, `test/`.
2. Huấn luyện, sau mỗi epoch đo accuracy trên tập validation.
3. Giữ trọng số có **validation accuracy cao nhất**; dừng sớm (early stopping) khi không cải thiện sau `patience` epoch.
4. Nạp lại trọng số tốt nhất, đánh giá **một lần** trên tập test.
5. Lưu `state_dict` vào `checkpoints/<tên>.pt`.

Chỉ số test: accuracy, macro precision / recall / F1, weighted F1 và bảng precision / recall / F1 / support từng lớp.

### 6.2 Kiến trúc

| Notebook | Kiến trúc |
| --- | --- |
| `m1_01` | MLP: Flatten → Linear 512 → Linear 128 → 9 (ReLU, Dropout 0.2), ảnh 128×128 |
| `m1_02` | CNN 2 block (Conv3×3 + BatchNorm + ReLU + MaxPool, 32 và 64 kênh) → AdaptiveAvgPool 4×4 → Linear 128 → 9 |
| `m1_03` | CNN 3 block (32, 64, 128 kênh) → AdaptiveAvgPool 4×4 → Linear 128 → 9 |
| `m1_04` | CNN 3 block (32, 64, 128 kênh) → Global Average Pooling → Linear 9 |
| `cnn_sequential` | 3 stage × 2 Conv3×3 + BatchNorm (48, 96, 192), MaxPool mỗi stage → GAP → Linear 256 → 9 (Dropout 0.3), 697.849 tham số |
| `cnn_parallel` | 3 nhánh song song (conv, 2 conv, conv giãn dilation 2; 32 kênh) nối lại → 3 khối residual (64, 128, 192) → GAP → Linear 256 → 9 |
| `vgg_sequential` | VGG16-style 13 lớp Conv3×3 + BatchNorm (64–512 kênh) → GAP → Linear 128 → 9 (Dropout 0.6) |
| `vgg_parallel` | 3 nhánh đầu vào song song (3×3, 3×3 dilation 2, 5×5; 64 kênh) nối lại → các stage kiểu VGG → GAP → Linear 128 → 9 |
| `resnet` | ResNet tự viết: stem Conv3×3 stride 1, 4 stage × 2 BasicBlock (64–512) → GAP → Linear 128 → 9, label smoothing 0.05 |
| `3a_frozen_vgg16` | VGG16 ImageNet đóng băng + head MLP (Linear 512 → BatchNorm → ReLU → Dropout 0.3 → 9) |
| `3b_frozen_resnet50` | ResNet50 ImageNet đóng băng + head MLP như trên |
| `3b_frozen_efficientnet_b0` | EfficientNet-B0 ImageNet đóng băng + head MLP như trên |
| `3c_lora_resnet50` | ResNet50 đóng băng + LoRA (rank 8, alpha 16) trên mọi `Conv2d` của layer3 và layer4 + head MLP |
| `3d_full_finetune_resnet50` | ResNet50 mở khóa toàn bộ trọng số + head MLP |

### 6.3 Cấu hình huấn luyện

| Hạng mục | Model 1 | Model 2 | Model 3 |
| --- | --- | --- | --- |
| Optimizer | Adam, lr 1e-3 | AdamW, lr 1e-4, weight decay 5e-4 | AdamW, head lr 1e-3, weight decay 1e-4 (3D: backbone lr 1e-4, weight decay 0.01) |
| LR scheduler | không | cosine annealing | cosine annealing |
| Batch size | 64 | 32 (`resnet`: 24 do giới hạn VRAM) | 64 (3A, 3B), 32 (3C, 3D) |
| Số epoch tối đa / patience | 50 / 15 | 30 / 7 | 30 / 6 (3D: 5) |
| Loss | CrossEntropy | CrossEntropy có class weight (`resnet` thêm label smoothing 0.05) | CrossEntropy |
| Khác | | | `drop_last=True` cho train; 3C: lr LoRA 1e-3; 3D: gradient clipping 1.0 |
| Seed | không đặt | chỉ `cnn_sequential` | `torch.manual_seed(42)` |

## 7. Kết quả cuối cùng

Đánh giá trên tập test (864 ảnh, 9 lớp), mỗi notebook một lần chạy.

### 7.1 Bảng tổng hợp

| Model | Test acc | Macro P | Macro R | Macro F1 | Weighted F1 | Epoch chạy | Best epoch | Thời gian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model 1 · M1-01 — MLP | 19.79% | 0.1961 | 0.1805 | 0.1263 | 0.1376 | 22 | 7 | 2.2 phút |
| Model 1 · M1-02 — CNN 2 block | 52.31% | 0.5262 | 0.5237 | 0.5080 | 0.5144 | 50 | 49 | 6.9 phút |
| Model 1 · M1-03 — CNN 3 block | 63.08% | 0.6350 | 0.6272 | 0.6253 | 0.6346 | 50 | 45 | 7.9 phút |
| Model 1 · M1-04 — CNN 3 block + GAP | 48.84% | 0.4926 | 0.4983 | 0.4771 | 0.4827 | 50 | 44 | 8.1 phút |
| Model 2 · CNN sequential | 65.97% | 0.6607 | 0.6610 | 0.6526 | 0.6602 | 30 | 30 | 16.1 phút |
| Model 2 · CNN parallel | 64.58% | 0.6383 | 0.6428 | 0.6361 | 0.6459 | 30 | 26 | 18.6 phút |
| Model 2 · VGG sequential | 72.34% | 0.7202 | 0.7175 | 0.7109 | 0.7242 | 30 | 25 | 29.1 phút |
| Model 2 · VGG parallel | 76.04% | 0.7570 | 0.7569 | 0.7477 | 0.7589 | 30 | 28 | 37.4 phút |
| Model 2 · ResNet (tự viết, không phải ResNet-18 chuẩn) | 69.91% | 0.6972 | 0.6973 | 0.6936 | 0.6982 | 30 | 29 | 58.9 phút |
| Model 3 · 3A — VGG16 đóng băng | 82.18% | 0.8192 | 0.8225 | 0.8162 | 0.8229 | 19 | 13 | 7.6 phút |
| Model 3 · 3B — ResNet50 đóng băng | 93.98% | 0.9376 | 0.9366 | 0.9363 | 0.9397 | 12 | 6 | 3.7 phút |
| Model 3 · 3B — EfficientNet-B0 đóng băng | 93.06% | 0.9249 | 0.9272 | 0.9245 | 0.9311 | 18 | 12 | 3.2 phút |
| Model 3 · 3C — ResNet50 + LoRA | 96.53% | 0.9635 | 0.9658 | 0.9643 | 0.9653 | 30 | 27 | 10.7 phút |
| Model 3 · 3D — ResNet50 fine-tune toàn bộ | 92.82% | 0.9205 | 0.9311 | 0.9234 | 0.9286 | 10 | 5 | 6.1 phút |

### 7.2 F1 theo từng lớp

| Lớp | m1_01 | m1_02 | m1_03 | m1_04 | m2_cnn_sequential | m2_cnn_parallel | m2_vgg_sequential | m2_vgg_parallel | m2_resnet | m3_3a_frozen_vgg16 | m3_3b_frozen_resnet50 | m3_3b_frozen_efficientnet_b0 | m3_3c_lora_resnet50 | m3_3d_full_finetune_resnet50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alu | 0.0385 | 0.5419 | 0.7191 | 0.6024 | 0.7264 | 0.7059 | 0.7960 | 0.8254 | 0.7474 | 0.8835 | 0.9648 | 0.9447 | 0.9744 | 0.9548 |
| Carton | 0.4537 | 0.7204 | 0.7739 | 0.7685 | 0.7767 | 0.8058 | 0.8390 | 0.8879 | 0.7885 | 0.8265 | 0.9469 | 0.9533 | 0.9668 | 0.9327 |
| Foam_box | 0.0152 | 0.3590 | 0.4780 | 0.3288 | 0.5027 | 0.4551 | 0.5571 | 0.5839 | 0.5752 | 0.8686 | 0.9146 | 0.9012 | 0.9425 | 0.8914 |
| Milk_box | 0.0708 | 0.7027 | 0.7600 | 0.6699 | 0.8200 | 0.8350 | 0.8687 | 0.8627 | 0.8584 | 0.9128 | 0.9806 | 0.9608 | 0.9901 | 0.9756 |
| Other | 0.0220 | 0.4462 | 0.5167 | 0.4260 | 0.5760 | 0.5323 | 0.5426 | 0.6202 | 0.6545 | 0.6838 | 0.8824 | 0.8257 | 0.9515 | 0.8448 |
| PET | 0.0357 | 0.4925 | 0.5772 | 0.3310 | 0.5093 | 0.5385 | 0.6083 | 0.6341 | 0.5333 | 0.7163 | 0.9412 | 0.9246 | 0.9648 | 0.9091 |
| Paper | 0.0000 | 0.4964 | 0.6857 | 0.5909 | 0.6798 | 0.6496 | 0.7282 | 0.8019 | 0.7182 | 0.7873 | 0.9390 | 0.9519 | 0.9585 | 0.9159 |
| Paper_cup | 0.1808 | 0.5646 | 0.6602 | 0.5761 | 0.7014 | 0.6981 | 0.8326 | 0.8739 | 0.8473 | 0.9116 | 0.9488 | 0.9573 | 0.9859 | 0.9859 |
| Plastic_cup | 0.3200 | 0.2483 | 0.4571 | 0.0000 | 0.5812 | 0.5048 | 0.6256 | 0.6389 | 0.5200 | 0.7553 | 0.9083 | 0.9009 | 0.9439 | 0.9000 |
| *Lớp tốt nhất* | Carton (0.45) | Carton (0.72) | Carton (0.77) | Carton (0.77) | Milk_box (0.82) | Milk_box (0.83) | Milk_box (0.87) | Carton (0.89) | Milk_box (0.86) | Milk_box (0.91) | Milk_box (0.98) | Milk_box (0.96) | Milk_box (0.99) | Paper_cup (0.99) |
| *Lớp kém nhất* | Paper (0.00) | Plastic_cup (0.25) | Plastic_cup (0.46) | Plastic_cup (0.00) | Foam_box (0.50) | Foam_box (0.46) | Other (0.54) | Foam_box (0.58) | Plastic_cup (0.52) | Other (0.68) | Other (0.88) | Other (0.83) | Foam_box (0.94) | Other (0.84) |

### 7.3 Mô hình tốt nhất: ResNet50 + LoRA (3C)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.9896 | 0.9596 | 0.9744 | 99 |
| Carton | 0.9903 | 0.9444 | 0.9668 | 108 |
| Foam_box | 0.9425 | 0.9425 | 0.9425 | 87 |
| Milk_box | 0.9901 | 0.9901 | 0.9901 | 101 |
| Other | 0.9245 | 0.9800 | 0.9515 | 50 |
| PET | 0.9897 | 0.9412 | 0.9648 | 102 |
| Paper | 0.9369 | 0.9811 | 0.9585 | 106 |
| Paper_cup | 0.9722 | 1.0000 | 0.9859 | 105 |
| Plastic_cup | 0.9352 | 0.9528 | 0.9439 | 106 |
| *macro avg* | 0.9635 | 0.9658 | 0.9643 | 864 |
| *weighted avg* | 0.9660 | 0.9653 | 0.9653 | 864 |

### 7.4 Nhận xét

- **Transfer learning vượt xa huấn luyện từ đầu**: Model 3 đạt 82.18–96.53% accuracy, Model 2 đạt 64.58–76.04% và
  Model 1 đạt 19.79–63.08%.
- **Trong Model 3**, LoRA (96.53%) cao hơn fine-tune toàn bộ (92.82%) 3.7 điểm; ResNet50 và EfficientNet-B0 đóng
  băng (93.98% và 93.06%) gần bằng nhau; VGG16 đóng băng thấp nhất (82.18%). Các biến thể chỉ train head dừng sớm
  (epoch tốt nhất ở 6–13) và có khoảng cách lớn giữa train và validation accuracy.
- **Trong Model 2**, `vgg_parallel` tốt nhất (76.04%) và hơn `vgg_sequential` 3.7 điểm; với CNN nhỏ, hai kiến trúc
  tuần tự và song song chênh nhau ít hơn độ dao động giữa các lần chạy (xem lưu ý bên dưới).
- **Trong Model 1**, thêm khối tích chập cải thiện rõ (52.31% → 63.08% khi từ 2 lên 3 block); MLP gần như không học
  được (19.79%, ngẫu nhiên là 11.1%); thay head Linear bằng Global Average Pooling thuần (`m1_04`) kém hơn `m1_03`.
- **Lớp khó và lớp dễ**: lớp có F1 thấp nhất thường là `Plastic_cup` ở Model 1 (3/4 notebook), `Foam_box` ở Model 2
  (3/5) và `Other` ở Model 3 (4/5; riêng 3C là `Foam_box`). Lớp có F1 cao nhất thường là `Milk_box` hoặc `Carton`
  (xem hai hàng cuối của bảng F1).

### 7.5 Lưu ý khi đọc kết quả

- Mỗi notebook chỉ chạy một lần và hầu hết không đặt seed. `cnn_parallel` được chạy hai lần, cho 68.17% và 64.58%
  (lệch 3.6 điểm), nên chênh lệch vài điểm giữa các model không chắc là thật.
- Với Model 1 và 2, epoch tốt nhất nằm sát giới hạn epoch (44–49 / 50 và 25–30 / 30) nên early stopping gần như
  không kích hoạt: các model này vẫn đang cải thiện khi hết epoch, chưa được huấn luyện đủ.
- Model 3 chạy ngày 2026-10-11 (PyTorch 2.14 + CUDA 13); Model 1 và 2 chạy ngày 2026-10-13 bằng kernel `python3` trỏ
  vào môi trường PyTorch 2.11 + CUDA 12.8. Các điều kiện khác nhau về cấu hình huấn luyện (xem 6.3), nên so sánh giữa
  các nhóm chỉ mang tính tham khảo.

## 8. Hạn chế và hướng phát triển

- Chưa có script tải dữ liệu gốc (`kagglehub` có trong `requirements.txt` nhưng chưa được dùng).
- Ảnh train đã được augment cố định một lần khi tiền xử lý, nên mỗi epoch thấy lại cùng các biến thể đó (cộng thêm
  augmentation nhẹ khi huấn luyện).
- Cần thêm epoch (hoặc cách huấn luyện khác) cho Model 1 và 2, và đặt seed cho mọi notebook, chạy lặp nhiều lần để
  có khoảng tin cậy.
- Chưa có kiểm thử tự động cho `src/preprocessing.py`.
