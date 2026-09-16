# trash-image-classification

Dự án phân loại rác thải Việt Nam bằng ảnh (Vietnamese trash image classification), dùng PyTorch/torchvision.

Trạng thái hiện tại:

- **Dữ liệu**: xong — EDA + pipeline tiền xử lý sinh ra dataset ảnh đã xử lý trên đĩa.
- **`src/model3`**: xong — bộ 4 thí nghiệm transfer learning với pretrained backbone (frozen → LoRA → full finetune), có checkpoint và báo cáo chỉ số đầy đủ. Xem [src/model3/README.md](src/model3/README.md).
- **`src/model1`, `src/model2`**: chưa triển khai (file `test.py` còn trống).

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
│   ├── model1/test.py       # Placeholder — chưa có code
│   ├── model2/test.py       # Placeholder — chưa có code
│   └── model3/              # Transfer learning 3A–3D (xem src/model3/README.md)
│       ├── config.py, data.py, backbones.py, heads.py, lora.py, model_builder.py
│       ├── engine.py, metrics.py, report.py, train.py, run_all.py
│       ├── pretrained/      # Weight ImageNet tải về (TORCH_HOME)
│       ├── checkpoints/     # Trọng số tốt nhất theo val accuracy
│       └── results/         # metrics.json/history/confusion matrix mỗi run + REPORT.md
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

### Model 3 — transfer learning với pretrained backbone

Bộ 4 thí nghiệm so sánh các chiến lược chuyển giao học tập trên dataset đã tiền xử lý:

| Biến thể | Backbone | Chiến lược |
| --- | --- | --- |
| 3A | MobileNetV2 | Đóng băng toàn bộ, chỉ train MLP head |
| 3B | ResNet50 / EfficientNet-B0 / ViT-B/16 | Đóng băng toàn bộ, chỉ train MLP head |
| 3C | backbone thắng ở 3B | Đóng băng + LoRA adapter (tự viết) |
| 3D | backbone thắng ở 3B | Fine-tune toàn bộ trọng số |

```bash
.venv/Scripts/python.exe src/model3/run_all.py
```

**Kết quả tốt nhất**: 3D fine-tune toàn bộ ResNet50 — test accuracy **0.9606**, F1 macro **0.9597**
(train 284s / 10 epoch trên RTX 5070). LoRA bám rất sát (0.9549) với chỉ 6,4% số tham số trainable
và 1/3 VRAM. Baseline 3A MobileNetV2 đóng băng đạt 0.8750.

Chi tiết cấu trúc code, cách chạy từng biến thể và ghi chú kỹ thuật: [src/model3/README.md](src/model3/README.md).
Kết quả đầy đủ (accuracy, precision/recall/F1 macro + weighted, per-class, thời gian chạy, peak GPU): [src/model3/results/REPORT.md](src/model3/results/REPORT.md).

## Việc còn thiếu / gợi ý tiếp theo

- `src/model1`, `src/model2`: các file `test.py` hiện trống — chưa có model, training loop, hay evaluation.
- Chưa có script tải dữ liệu gốc dù `kagglehub` đã có trong `requirements.txt`.
- Chưa có test tự động (`pytest`) cho `src/preprocessing.py` và `src/model3`.
