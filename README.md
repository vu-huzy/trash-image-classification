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
(9.542 / 1.685 / 864 ảnh), Adam lr=0.001, batch size 64, tối đa 50 epoch, early stopping theo validation accuracy
(patience 15). M1-01 nhận ảnh 128×128, M1-02/03/04 nhận 224×224; train dùng Resize + RandomHorizontalFlip + ColorJitter + ToTensor + Normalize ImageNet;
val/test dùng Resize + ToTensor + Normalize ImageNet. `metrics.py` là phần metric dùng chung (`calculate_metrics`).
Các script `.py` cũ của Model 1, `configs/`, `src/common/`, `docs/` và `reports/` đã bị xóa.

Kết quả lần chạy lại ngày 2026-10-13 với code `cdee048` (GPU RTX 5070, DataLoader 8 worker); bảng
precision/recall/F1/support từng lớp nằm ở [RESULTS.md](RESULTS.md):

| Model | Test accuracy | Macro F1 | Epoch tốt nhất | Thời gian |
| --- | ---: | ---: | ---: | ---: |
| M1-01 — MLP | 19.79% | 0.1263 | 7 (dừng ở 22) | 2.2 phút |
| M1-02 — CNN 2 block | 52.31% | 0.5080 | 49 / 50 | 6.9 phút |
| M1-03 — CNN 3 block | **63.08%** | **0.6253** | 45 / 50 | 7.9 phút |
| M1-04 — CNN 3 block + GAP | 48.84% | 0.4771 | 44 / 50 | 8.1 phút |

Lưu ý khi đọc bảng: các số cũ (18.06% / 56.37% / 60.30% / 52.66%) đến từ script `.py` cũ cùng có augmentation
(HFlip + ColorJitter) và chuẩn hóa ImageNet. M1-02/03/04 chạy đủ 50 epoch với epoch tốt nhất ở 49, 45 và 44, tức là
vẫn đang cải thiện khi chạm giới hạn epoch nên early stopping không kích hoạt. M1-01 vẫn học rất kém (accuracy train
20.86%, validation 23.50% ở epoch tốt nhất). Bảng trước đó (chỉ Resize + ToTensor, chọn epoch theo val loss) cho
15.62% / 45.95% / 48.73% / 45.95%; ba thay đổi (augmentation, chuẩn hóa, tiêu chí chọn epoch) diễn ra cùng lúc nên
chưa tách được từng yếu tố.

Notebook `src/model1/m1_0x.ipynb`: early stopping giữ `best_weights` (val accuracy cao nhất) trong RAM;
ngay sau vòng lặp train, notebook nạp lại `best_weights` vào `model` (vì khi dừng, `model` đang giữ
trọng số của epoch cuối). Cell cuối cùng lưu `state_dict` của model tốt nhất ra
`src/model1/checkpoints/m1_0x.pt` (thư mục đã nằm trong `.gitignore`). Sau khi chạy notebook, đặt
`MODEL_NAME` trong `src/model1/Predict.py` rồi chạy file đó: file chỉ nạp checkpoint, chạy tập test và in accuracy cùng bảng `classification_report` từng lớp.

### Model 2 — CNN / VGG / ResNet tự viết

5 kiến trúc: `cnn_sequential`, `cnn_parallel`, `vgg_sequential`, `vgg_parallel`, `resnet`. Mỗi notebook trong
`src/model2/notebook/` là pipeline đầy đủ (dữ liệu → model → train → test → lưu); file `.py` duy nhất còn lại là
`src/model2/py/metrics.py` (`calculate_metrics`, notebook import qua `sys.path.append('../py')`). Đường dẫn trong
notebook là đường dẫn tương đối từ thư mục notebook. Phần test in các chỉ số tổng và bảng
precision/recall/F1/support từng nhãn (`classification_report`), không có confusion matrix. Mỗi model có đúng 1
file chuẩn `src/model2/checkpoints/<model>.pt`, các lần chạy cũ nằm trong `checkpoints/archive/`. Kiểm tra bằng
`src/model2/Predict.py`. Chi tiết (bảng checkpoint, lưu ý về `resnet`): [src/model2/README.md](src/model2/README.md);
bảng từng lớp: [RESULTS.md](RESULTS.md).

Kết quả (4 notebook chạy lại ngày 2026-10-13 với code `cdee048`, patience 7; `resnet` là kết quả cũ ngày
2026-10-11 vì lần chạy lại bị dừng; GPU RTX 5070):

| Model | Test accuracy | Macro F1 | Epoch tốt nhất | Thời gian |
| --- | ---: | ---: | ---: | ---: |
| cnn_sequential | 65.97% | 0.6526 | 30 / 30 | 16.1 phút |
| cnn_parallel | 68.17% | 0.6754 | 30 / 30 | 17.8 phút |
| vgg_sequential | 72.34% | 0.7109 | 25 / 30 | 29.1 phút |
| vgg_parallel | **76.04%** | **0.7477** | 28 / 30 | 37.4 phút |
| resnet (kết quả cũ) | 69.33% | 0.6813 | 30 / 30 | 90.7 phút |

Epoch tốt nhất của 4 model chạy lại là 30, 30, 25 và 28 trên 30, early stopping không kích hoạt ở model nào, tức là
chưa model nào ngừng cải thiện trước giới hạn epoch. So với checkpoint cũ chọn từ các lần chạy `train.py`, 4 bản
notebook mới thấp hơn 0.9–4.5 điểm accuracy; hai pipeline khác nhau nên chưa quy được cho một nguyên nhân.
Mỗi model chỉ chạy một lần.

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

### Mức đồng nhất workflow giữa 14 notebook (rà soát ở commit `cdee048`)

Đã giống nhau ở cả 14 notebook: dữ liệu `ImageFolder` trên `train/val/test`; train = Resize + RandomHorizontalFlip +
ColorJitter(0.1) + Normalize ImageNet, val/test = Resize + Normalize; chọn best checkpoint theo **val accuracy cao nhất**;
early stopping theo val accuracy; nạp lại best weights sau vòng lặp; test in accuracy / macro P-R-F1 / weighted F1 /
`classification_report`; cell cuối lưu `state_dict` một dòng.

Vẫn khác nhau (chưa gom, cần quyết định trước khi so sánh model với nhau):

| Hạng mục | Model 1 | Model 2 | Model 3 |
| --- | --- | --- | --- |
| Patience / số epoch tối đa | 15 / 50 | 7 / 30 | 6 / 30 (3D: 5 / 30) |
| Optimizer + scheduler | Adam 1e-3, không scheduler | AdamW 1e-4, wd 5e-4, cosine | AdamW, head 1e-3, wd 1e-4 (3D: wd 0.01), cosine |
| Batch size | 64 | 32 (`resnet`: 24) | 64 (3A/3B), 32 (3C/3D) |
| Class weight / label smoothing | không / không | có / chỉ `resnet` (0.05) | không / không |
| Seed | không | chỉ `cnn_sequential` | `torch.manual_seed(42)` |
| Khác | `m1_01` dùng ảnh 128, còn lại 224 | tên biến viết tắt (`train_ds`, `loss_fn`, `LR`) ở 4/5 notebook | 3D có grad clip 1.0; `drop_last=True`; metric viết inline, không dùng `metrics.py` |

`src/model1/metrics.py` và `src/model2/py/metrics.py` là hai bản gần giống nhau (bản Model 2 có thêm kiểm tra rỗng và độ dài).

Môi trường chạy: kernelspec `python3` của user (`%APPDATA%\jupyter\kernels\python3`) trỏ vào `anaconda3\envs\gpu`
(torch 2.11.0+cu128, scikit-learn 1.9.0), **không phải** `.venv` (torch 2.14.0+cu130 theo `requirements.txt`).
Notebook chạy bằng kernel `python3` sẽ dùng env anaconda, cần ghi rõ env khi so sánh kết quả giữa các lần chạy.

### Kết quả chạy lại ngày 2026-10-13 (code `cdee048`, patience Model 2 đặt về 7)

Chạy nối tiếp 14 notebook bằng `nbconvert --execute --inplace` (GPU RTX 5070). Checkpoint cũ được cất ở
`C:\dl-project\backup_checkpoints_20261013\{model1,model2,model3}` trước khi chạy.

| Notebook | Epoch đã chạy | Epoch tốt nhất | Early stop | Val acc | Test acc | Macro F1 | Thời gian |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `m1_01` | 22 / 50 | 7 | có | 23.50% | 19.79% | 0.1263 | 2.2 phút |
| `m1_02` | 50 / 50 | 49 | không | 62.31% | 52.31% | 0.5080 | 6.9 phút |
| `m1_03` | 50 / 50 | 45 | không | 69.14% | 63.08% | 0.6253 | 7.9 phút |
| `m1_04` | 50 / 50 | 44 | không | 60.36% | 48.84% | 0.4771 | 8.1 phút |
| `cnn_sequential` | 30 / 30 | 30 | không | 69.38% | 65.97% | 0.6526 | 16.1 phút |
| `cnn_parallel` | 30 / 30 | 30 | không | 70.74% | 68.17% | 0.6754 | 17.8 phút |
| `vgg_sequential` | 30 / 30 | 25 | không | 79.70% | 72.34% | 0.7109 | 29.1 phút |
| `vgg_parallel` | 30 / 30 | 28 | không | 82.55% | 76.04% | 0.7477 | 37.4 phút |

- Early stopping gần như không kích hoạt ở Model 1/2: epoch tốt nhất nằm sát giới hạn epoch (44–49 / 50, 25–30 / 30),
  tức model còn đang cải thiện khi hết epoch. Đổi patience (7 hay 15) không đổi kết quả ở hai nhóm này; muốn tốt hơn
  phải tăng số epoch hoặc đổi cách train.
- `resnet` (Model 2) **không có kết quả mới**: chạy 140 phút vẫn chưa xong rồi bị dừng. Epoch đầu mất ~16 phút
  (~17 phút/epoch, lần trước ~3 phút/epoch); VRAM 11.9/12.2 GB, GPU báo 100% nhưng chỉ ~52 W, nghi thiếu VRAM.
  Notebook `resnet.ipynb` không bị ghi đè kết quả; đã hạ `BATCH_SIZE` từ 32 xuống 24 để tránh tràn VRAM, **chưa chạy lại**.
  `src/model2/checkpoints/resnet.pt` hiện là file cũ (chạy 2026-10-11, batch 32, chưa có early stopping), được chép lại từ backup.
- Model 3 **không chạy lại**: output trong 5 notebook, `src/model3/results/` và checkpoint là của lần chạy ngày
  2026-10-11 với patience 6 (3D: 5); patience trong 5 notebook đã được đặt lại đúng 6 (3D: 5) để khớp với kết quả đó.
- Kernel `python3` trong lần chạy này là `anaconda3\envs\gpu` (torch 2.11.0+cu128), xem mục môi trường ở trên.
- `RESULTS.md` và các bảng kết quả Model 1/2 ở phía trên (kèm `src/model2/README.md`) đã được cập nhật theo bảng này;
  riêng `resnet` và Model 3 vẫn giữ số của lần chạy ngày 2026-10-11.

## Việc còn thiếu / gợi ý tiếp theo

- Kết quả chi tiết của 14 notebook (Model 1/2/3) nằm trong [RESULTS.md](RESULTS.md); file này được dựng từ
  output của các notebook, nên cập nhật lại sau mỗi lần chạy lại.
- Chưa có script tải dữ liệu gốc dù `kagglehub` đã có trong `requirements.txt`.
- Chưa có test tự động (`pytest`) cho `src/preprocessing.py` và `src/model3`.
