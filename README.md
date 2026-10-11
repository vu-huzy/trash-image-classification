# trash-image-classification

Dự án phân loại rác thải Việt Nam bằng ảnh (Vietnamese trash image classification), dùng PyTorch/torchvision.

Trạng thái hiện tại:

- **Dữ liệu**: xong — EDA + pipeline tiền xử lý sinh ra dataset ảnh đã xử lý trên đĩa.
- **`src/model1`**: 4 notebook `m1_01..04` (MLP → CNN), mỗi notebook lưu `state_dict` vào `src/model1/checkpoints/`; `Predict.py` để dự đoán và kiểm tra kết quả.
- **`src/model2`**: 5 kiến trúc CNN/VGG/ResNet tự viết; `notebook/` (pipeline đầy đủ, chỉ import `py/metrics.py`); `Predict.py` để kiểm tra. Xem [src/model2/README.md](src/model2/README.md).
- **`src/model3`**: transfer learning với pretrained backbone (frozen → LoRA → full finetune); 5 notebook đơn giản trong `notebook/` (mỗi notebook lưu `state_dict` vào `checkpoints/`), `Predict.py` để kiểm tra; framework `.py` cũ đã bị xóa. Xem [src/model3/README.md](src/model3/README.md) (mô tả framework cũ).
- **Kết quả**: ghi chép chi tiết (cấu hình, epoch tốt nhất, chỉ số test, bảng precision/recall/F1/support từng lớp) của cả 14 notebook ở [RESULTS.md](RESULTS.md).

## Cấu trúc thư mục

```
trash-image-classification/
├── data/
│   ├── VN_trash_classification/            # Dataset gốc (Train/, Test/), theo lớp = tên thư mục con
│   └── VN_trash_classification_preprocessing/  # Dataset sau preprocessing: train/, val/, test/ + metadata
├── notebooks/
│   ├── EDA.ipynb            # Khám phá dữ liệu: phân bố lớp, ảnh lỗi, kích thước ảnh, ảnh mẫu
│   └── preprocessing.ipynb  # Bản notebook: quét ảnh, chia train/val, tạo Dataset/DataLoader PyTorch (trong bộ nhớ)
├── src/
│   ├── preprocessing.py     # Bản script: thực hiện lại pipeline như notebook, NHƯNG ghi ảnh đã xử lý ra đĩa
│   ├── model1/              # m1_01..04 (.ipynb), metrics.py, checkpoints/, Predict.py
│   ├── model2/              # notebook/ (5 notebook), py/metrics.py, checkpoints/, results/, Predict.py
│   └── model3/              # Transfer learning 3A–3D (xem src/model3/README.md)
│       ├── notebook/        # 5 notebook đơn giản: 3a, 3b (ResNet50, EfficientNet-B0), 3c, 3d
│       ├── Predict.py       # Nạp checkpoint và kiểm tra trên tập test
│       ├── pretrained/      # Weight ImageNet tải về (TORCH_HOME)
│       ├── checkpoints/     # Trọng số tốt nhất theo val accuracy (notebook ghi vào đây)
│       └── results/         # Kết quả cũ của framework: metrics.json/history/confusion matrix + REPORT.md
├── requirements.txt
└── .venv/                   # Virtualenv cục bộ (không commit)
```

## Dữ liệu

- **Nguồn/gốc**: `data/VN_trash_classification/{Train,Test}/<Lớp>/*.jpg`. Nhãn được suy ra từ **tên thư mục cha** của mỗi ảnh.
- **9 lớp**: `Alu`, `Carton`, `Foam_box`, `Milk_box`, `Other`, `PET`, `Paper`, `Paper_cup`, `Plastic_cup`.
- **Số lượng ảnh gốc**:
  - Train: 11.227 ảnh (dao động 1.135–1.322 ảnh/lớp — tương đối cân bằng).
  - Test: 864 ảnh (dao động 50–108 ảnh/lớp).
- `requirements.txt` có `kagglehub`, gợi ý dataset gốc được tải từ Kaggle, nhưng repo **chưa có script tải dữ liệu** (không tìm thấy `download_data.py`/`load_data.py` dù notebook có nhắc tới) — hiện `data/` được giả định là đã có sẵn.

## Hai đường xử lý dữ liệu (notebook vs script)

Cả hai đều dùng chung logic: quét ảnh bằng `PIL.Image.verify()` để loại ảnh hỏng, gán nhãn theo thư mục cha, và chia `Train` gốc thành `train`/`validation` theo tỉ lệ 85/15 bằng `train_test_split(..., stratify=label, random_state=42)` (giữ nguyên `Test` gốc để đánh giá cuối). Điểm khác biệt:

| | `notebooks/preprocessing.ipynb` | `src/preprocessing.py` |
|---|---|---|
| Kết quả | Không ghi ảnh mới; chỉ lưu **metadata** (`data/processed_metadata/{train,validation,test}.csv`, `class_to_idx.json`, `class_weights.json`) và dựng `Dataset`/`DataLoader` PyTorch trong bộ nhớ để trực quan hoá | Ghi **ảnh đã biến đổi thật ra đĩa** vào `data/VN_trash_classification_preprocessing/{train,val,test}/<Lớp>/*.jpg` |
| Transform | Resize/crop/flip/rotation/color-jitter + `ToTensor()` + `Normalize` (chuẩn ImageNet) → tensor | Resize/crop/flip/rotation/color-jitter, **không** `ToTensor`/`Normalize` (giữ ảnh JPEG để có thể mở lại bằng bất kỳ tool nào) |
| Class weights | Có tính (nghịch đảo tần suất lớp) | Không (không cần vì không train trực tiếp ở bước này) |
| Cách chạy | Mở bằng Jupyter, chạy từng cell | `python src/preprocessing.py [--project-dir DIR] [--overwrite]` |

Tập train dùng augmentation ngẫu nhiên (`RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`); tập `val`/`test` chỉ resize về 224×224 để việc đánh giá ổn định, không ngẫu nhiên.

## `src/preprocessing.py` — chi tiết

1. `collect_images(split_dir, split_name)`: quét đệ quy `Train`/`Test`, mở từng ảnh bằng `Image.verify()`, bỏ qua và ghi nhận file hỏng.
2. `make_transforms()`: trả về transform augmentation cho train và transform resize-only cho val/test.
3. `save_split(df, split_name, output_dir, transform)`: với mỗi ảnh — mở, `convert("RGB")`, áp transform, lưu JPEG chất lượng 95 vào `output_dir/split_name/<label>/000000.jpg` (đánh số theo chỉ số dòng).
4. `build_dataset(project_dir, overwrite)`: điều phối toàn bộ — kiểm tra dữ liệu đầu vào, chia train/val theo stratify, chạy `save_split` cho cả 3 split, rồi ghi `metadata.csv`, `class_to_idx.json`, `class_counts.csv`, và `bad_images.csv` (nếu có ảnh lỗi).

**Lỗi đã gặp và đã sửa**: khi chạy với `--overwrite` trên Windows, `shutil.rmtree()` xoá thư mục output cũ đôi khi báo `WinError 145: The directory is not empty` — nguyên nhân là Windows Defender/search indexer còn giữ handle trên các file JPEG vừa ghi xong trong lần chạy trước, khiến việc xoá bị race condition. Đã thêm `remove_directory_with_retry()` (retry + xoá cờ read-only) để xử lý ổn định vấn đề này.

## Cách chạy

```bash
# Cài dependencies vào virtualenv (đã có sẵn tại .venv/ trong repo này)
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Tạo lại dataset đã tiền xử lý (ghi đè nếu đã tồn tại)
.venv/Scripts/python.exe src/preprocessing.py --overwrite
```

Kết quả nằm tại `data/VN_trash_classification_preprocessing/`:
- `train/<Lớp>/*.jpg`, `val/<Lớp>/*.jpg`, `test/<Lớp>/*.jpg` — ảnh đã tiền xử lý (đã resize/augment, RGB, JPEG q95).
- `metadata.csv` — path/label/split/source_path cho từng ảnh đã lưu.
- `class_to_idx.json` — ánh xạ tên lớp → id (thứ tự alphabet, cố định giữa các lần chạy).
- `class_counts.csv` — số ảnh theo split × lớp.
- `bad_images.csv` — danh sách ảnh gốc không mở được (nếu có).

## Huấn luyện mô hình

### Model 1 — Simple NN / CNN trên JPEG tiền xử lý

Chạy từng notebook `src/model1/m1_01.ipynb` đến `m1_04.ipynb` (thư mục làm việc là `src/model1`).
Notebook dùng `ImageFolder` trên `data/VN_trash_classification_preprocessing/{train,val,test}`
(9.542 / 1.685 / 864 ảnh), Adam lr=0.001, batch size 64, tối đa 50 epoch, early stopping theo validation loss
(patience 15). M1-01 nhận ảnh 128×128, M1-02/03/04 nhận 224×224; ảnh chỉ được Resize + ToTensor
(không chuẩn hóa, không augmentation). `metrics.py` là phần metric dùng chung (`calculate_metrics`).
Các script `.py` cũ của Model 1, `configs/`, `src/common/`, `docs/` và `reports/` đã bị xóa.

Kết quả lần chạy lại bằng notebook (GPU RTX 5070, DataLoader 8 worker); bảng precision/recall/F1/support từng lớp
nằm ở [RESULTS.md](RESULTS.md):

| Model | Test accuracy | Macro F1 | Epoch tốt nhất | Thời gian |
| --- | ---: | ---: | ---: | ---: |
| M1-01 — MLP | 15.62% | 0.1016 | 45 / 50 | 3.5 phút |
| M1-02 — CNN 2 block | 45.95% | 0.4531 | 34 (dừng ở 49) | 7.3 phút |
| M1-03 — CNN 3 block | **48.73%** | **0.4840** | 16 (dừng ở 31) | 5.8 phút |
| M1-04 — CNN 3 block + GAP | 45.95% | 0.4167 | 50 / 50 | 8.5 phút |

Lưu ý khi đọc bảng: các số cũ (18.06% / 56.37% / 60.30% / 52.66%) đến từ script `.py` cũ có augmentation
(HFlip + ColorJitter) và chuẩn hóa ImageNet, nên không so sánh trực tiếp được với bảng trên; chưa có thí nghiệm
riêng để xác định nguyên nhân chênh lệch. M1-04 đạt validation loss thấp nhất đúng ở epoch cuối (50/50), tức là
vẫn đang cải thiện khi chạm giới hạn 50 epoch. M1-01 gần như không học được (accuracy train 18.76% ở epoch tốt nhất).

Notebook `src/model1/m1_0x.ipynb`: early stopping giữ `best_weights` (val loss thấp nhất) trong RAM;
ngay sau vòng lặp train, notebook nạp lại `best_weights` vào `model` (vì khi dừng, `model` đang giữ
trọng số của epoch cuối). Cell cuối cùng lưu `state_dict` của model tốt nhất ra
`src/model1/checkpoints/m1_0x.pt` (thư mục đã nằm trong `.gitignore`). Sau khi chạy notebook, đặt
`MODEL_NAME` trong `src/model1/Predict.py` rồi chạy file đó để xem dự đoán và accuracy/F1 trên tập test.

### Model 2 — CNN / VGG / ResNet tự viết

5 kiến trúc: `cnn_sequential`, `cnn_parallel`, `vgg_sequential`, `vgg_parallel`, `resnet`. Mỗi notebook trong
`src/model2/notebook/` là pipeline đầy đủ (dữ liệu → model → train → test → lưu); file `.py` duy nhất còn lại là
`src/model2/py/metrics.py` (`calculate_metrics`, notebook import qua `sys.path.append('../py')`). Đường dẫn trong
notebook là đường dẫn tương đối từ thư mục notebook. Phần test in các chỉ số tổng và bảng
precision/recall/F1/support từng nhãn (`classification_report`), không có confusion matrix. Mỗi model có đúng 1
file chuẩn `src/model2/checkpoints/<model>.pt`, các lần chạy cũ nằm trong `checkpoints/archive/`. Kiểm tra bằng
`src/model2/Predict.py`. Chi tiết (bảng checkpoint, lưu ý về `resnet`): [src/model2/README.md](src/model2/README.md);
bảng từng lớp: [RESULTS.md](RESULTS.md).

Kết quả lần chạy lại bằng notebook (30 epoch, GPU RTX 5070):

| Model | Test accuracy | Macro F1 | Epoch tốt nhất | Thời gian |
| --- | ---: | ---: | ---: | ---: |
| cnn_sequential | 64.12% | 0.6320 | 28 / 30 | 16.0 phút |
| cnn_parallel | 64.12% | 0.6339 | 27 / 30 | 18.2 phút |
| vgg_sequential | 73.38% | 0.7217 | 29 / 30 | 29.3 phút |
| vgg_parallel | **76.04%** | **0.7487** | 28 / 30 | 45.3 phút |
| resnet | 69.33% | 0.6813 | 30 / 30 | 90.7 phút |

Epoch tốt nhất của cả 5 model nằm ở 27–30 trên 30, tức là chưa model nào ngừng cải thiện trước giới hạn epoch.
So với checkpoint cũ chọn từ các lần chạy `train.py`, bản notebook thấp hơn 0.8–6.4 điểm accuracy; hai pipeline
khác nhau nên chưa quy được cho một nguyên nhân. Mỗi model chỉ chạy một lần.

### Model 3 — transfer learning với pretrained backbone

5 notebook trong `src/model3/notebook/` so sánh các chiến lược chuyển giao học tập trên dataset đã tiền xử lý
(cùng thiết kế: backbone ImageNet + head MLP 512, early stopping theo val accuracy, LoRA tự viết cho 3C).
Mỗi notebook là pipeline đầy đủ và lưu `state_dict` vào `src/model3/checkpoints/<tên>.pt`; `src/model3/Predict.py`
(đặt `MODEL_NAME`) nạp checkpoint và in accuracy/F1 + bảng từng lớp trên tập test.

| Notebook | Backbone | Chiến lược | Test accuracy | Macro F1 | Epoch tốt nhất | Thời gian |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `3a_frozen_vgg16` | VGG16 | Đóng băng, chỉ train head MLP | 82.18% | 0.8162 | 13 (dừng ở 19) | 7.6 phút |
| `3b_frozen_resnet50` | ResNet50 | Đóng băng, chỉ train head MLP | 93.98% | 0.9363 | 6 (dừng ở 12) | 3.7 phút |
| `3b_frozen_efficientnet_b0` | EfficientNet-B0 | Đóng băng, chỉ train head MLP | 93.06% | 0.9245 | 12 (dừng ở 18) | 3.2 phút |
| `3c_lora_resnet50` | ResNet50 | Đóng băng + LoRA (rank 8) ở layer3/layer4 | **96.53%** | **0.9643** | 27 / 30 | 10.7 phút |
| `3d_full_finetune_resnet50` | ResNet50 | Fine-tune toàn bộ | 92.82% | 0.9234 | 5 (dừng ở 10) | 6.1 phút |

Kết quả lần chạy lại bằng notebook (GPU RTX 5070, DataLoader 8 worker, mỗi notebook một lần, seed 42):
LoRA (3C) tốt nhất, cao hơn fine-tune toàn bộ (3D) 3.7 điểm accuracy; 3D đạt val accuracy cao nhất ở epoch 5 rồi
không cải thiện trong 5 epoch liên tiếp nên dừng ở epoch 10. Thứ hạng này **ngược** với framework cũ
(3D 0.9606 > 3C 0.9549) vốn dùng cấu hình huấn luyện khác bản notebook; chưa có thí nghiệm riêng để xác định
nguyên nhân. Bảng precision/recall/F1/support từng lớp:
[RESULTS.md](RESULTS.md). Kết quả của framework cũ vẫn nằm ở
[src/model3/results/REPORT.md](src/model3/results/REPORT.md) (backbone khác: có MobileNetV2 và ViT-B/16).

## Việc còn thiếu / gợi ý tiếp theo

- Kết quả chi tiết của 14 notebook (Model 1/2/3) nằm trong [RESULTS.md](RESULTS.md); file này được dựng từ
  output của các notebook, nên cập nhật lại sau mỗi lần chạy lại.
- Chưa có script tải dữ liệu gốc dù `kagglehub` đã có trong `requirements.txt`.
- Chưa có test tự động (`pytest`) cho `src/preprocessing.py` và `src/model3`.
