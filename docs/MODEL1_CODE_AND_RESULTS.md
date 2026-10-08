# Model 1 — Code, workflow và kết quả thực nghiệm (Dropout 0.2)

Tài liệu này là bản ghi kết quả chính thức của lần chạy lại M1-01 đến M1-04 với cấu hình hiện tại. Mọi lớp `Dropout` đang dùng đều có xác suất `p=0.2`; M1-04 không có lớp Dropout vì đó là biến thể Global Average Pooling (GAP) để so sánh riêng kiểu classifier head.

## 1. Điều kiện thí nghiệm và nguyên tắc đánh giá

| Hạng mục | Thiết lập chung |
|---|---|
| Bài toán | Phân loại ảnh rác 9 lớp: Alu, Carton, Foam_box, Milk_box, Other, PET, Paper, Paper_cup, Plastic_cup |
| Dữ liệu | Cache `.npy` ảnh `uint8 [3,224,224]`; train/validation split stratified 85%/15%, `seed=42`; test set tách riêng |
| Batch / epoch | Batch `256`, tối đa `50` epoch |
| Loss | `CrossEntropyLoss` |
| Optimizer | M1-01: AdamW (`lr=0.001`, `weight_decay=0.0001`); M1-02 đến M1-04: Adam (`lr=0.001`) |
| Early stopping | Theo **validation loss**, `patience=8`, `min_delta=0.001` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU; CUDA AMP FP16, TF32, channels-last cho CNN, cuDNN benchmark, fused optimizer nếu PyTorch hỗ trợ |

Quy tắc chống data leakage:

```text
train batch  -> tính loss -> backward -> optimizer.step()
validation   -> chỉ tính metric/loss -> chọn checkpoint tốt nhất + early stopping
test         -> nạp checkpoint tốt nhất theo validation loss -> chạy đúng một lần -> báo cáo
```

`Test loss` vẫn được ghi trong báo cáo vì nó là Cross-Entropy trung bình trên test set: loss càng thấp thì logits đúng lớp thường càng tự tin hơn. Nó **không** dùng để cập nhật trọng số, chọn checkpoint hay chỉnh hyperparameter; các quyết định đó chỉ dùng validation loss.

## 2. Cấu trúc file và luồng gọi code

```text
src/
├── common/
│   ├── data.py       # cache, split, DataLoader, augmentation/normalize GPU, prefetch
│   ├── engine.py     # run_epoch: train, validation hoặc test
│   ├── metrics.py    # macro/weighted F1, precision, recall, confusion matrix
│   └── utils.py      # seed, đếm parameter, ghi JSON/CSV
└── model1/
    ├── models/
    │   ├── m1_01.py # MLP baseline
    │   ├── m1_02.py # CNN 2 blocks + Dense head
    │   ├── m1_03.py # CNN 3 blocks + Dense head
    │   └── m1_04.py # CNN 3 blocks + GAP head
    ├── blocks.py     # Conv -> BatchNorm -> ReLU -> MaxPool
    ├── workflow.py   # workflow train/evaluate/checkpoint/report dùng chung
    ├── registry.py   # architecture trong JSON -> module model tương ứng
    ├── train.py      # CLI, CUDA setup, lần lượt gọi model_module.run(...)
```

Luồng chạy thực tế:

```text
run.py model1 -> train.main()
        -> đọc configs/m1_0x.json
        -> registry.get_model_module(architecture)
        -> m1_0x.run(...)
        -> workflow.run_model_workflow(...)
        -> data.make_loaders() + model build_model()
        -> engine.run_epoch() cho train/validation
        -> chọn checkpoint theo validation loss
        -> engine.run_epoch(..., collect_predictions=True) trên test
        -> metrics.classification_report() -> reports/model1/m1_0x.json
```

### 2.1 Các file chung được mọi M1 dùng

| File | Phần chính | Vai trò chi tiết |
|---|---|---|
| `configs/m1_01.json` … `m1_04.json` | `input_size`, `dropout`, optimizer, seed, early stopping | Nguồn hyperparameter cho từng thí nghiệm. M1-01 có thêm `dropout_first`; cả hai dropout của M1-01 đều là `0.2`. |
| `src/common/data.py` | `CachedImageDataset`, `make_loaders`, `prepare_images` | Đọc `train_images.npy`/`test_images.npy` bằng memory map. Mỗi batch được copy writable, chuyển GPU, chia `255`, resize M1-01 về `128×128`, flip ngang ngẫu nhiên khi train, chuẩn hóa ImageNet. Windows dùng `workers=0` và `BackgroundBatchIterator` để tránh lỗi multiprocessing với memory map. |
| `src/common/engine.py` | `run_epoch` | Dùng một `CrossEntropyLoss`. Khi có `optimizer`: `model.train()`, zero grad, autocast, backward/scaler/step. Khi validation/test: `model.eval()`, không gradient; Dropout tự tắt. Loss và accuracy được cộng theo batch size để ra trung bình toàn tập. |
| `src/common/metrics.py` | `classification_report` | Nhận nhãn thật/dự đoán test, tính macro precision/recall/F1, weighted F1, metric từng lớp và confusion matrix. Macro F1 cho mỗi lớp trọng số bằng nhau, nên phù hợp khi số ảnh từng lớp không hoàn toàn bằng nhau. |
| `src/common/utils.py` | `set_seed`, `count_parameters`, `save_json`, `append_experiment` | Cố định random seed, đếm parameter trainable, ghi chi tiết lịch sử vào JSON và một hàng tóm tắt vào CSV. |
| `src/model1/blocks.py` | `conv_bn_relu_pool` | Dùng lại block `Conv2d(k=3,padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)`. Conv/BN/ReLU giữ H×W, MaxPool giảm mỗi chiều không gian còn một nửa. |
| `src/model1/workflow.py` | `run_model_workflow` | Tạo loader/model/optimizer/scaler; thực hiện train + validation; lưu checkpoint tốt nhất; nạp checkpoint đó để test một lần; lưu JSON. CNN được chuyển sang `channels_last`; MLP không dùng layout này. |
| `src/model1/registry.py` | `get_model_module` | Ánh xạ `m1_01` … `m1_04` trong JSON sang chính file model tương ứng, tránh `if/else` kiến trúc dồn vào một file lớn. |
| `src/model1/train.py` | `main` | Bật tối ưu CUDA chung, đọc class names từ cache metadata rồi lần lượt gọi `run()` cho model được yêu cầu. Chạy từ repo root qua `python run.py model1 --models M1-01`. |

## 3. Data pipeline và shape đầu vào chung

```text
cache: uint8 [B, 3, 224, 224]
-> GPU float32, giá trị [0,255] / 255                 [B, 3, 224, 224]
-> M1-01: bilinear resize 224x224 -> 128x128          [B, 3, 128, 128]
   CNN M1-02/M1-03/M1-04: giữ 224x224                 [B, 3, 224, 224]
-> random horizontal flip, chỉ ở train                [B, 3, H, W]
-> ImageNet normalization                              [B, 3, H, W]
-> model trả logits                                    [B, 9]
-> CrossEntropyLoss(logits, labels)
```

`B` là batch size (`256` khi train thật). Logits là 9 điểm số chưa qua softmax; `CrossEntropyLoss` nội bộ đã xử lý log-softmax ổn định số học. Dự đoán class là `logits.argmax(dim=1)`.

## 4. Kết quả chạy lại với Dropout 0.2

| Model | Dropout thực tế | Epoch đã chạy | Best epoch | Parameters | Best val loss | Test loss | Test accuracy | Macro P / R / F1 | Weighted F1 | Thời gian | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| M1-01 | `0.2 / 0.2` | 41 | 33 | 25,233,161 | 1.8442 | 2.0770 | 23.73% | 0.2399 / 0.2405 / 0.2194 | 0.2246 | 0m 40s | 0.68 GB |
| M1-02 | `0.2` | 50 | 49 | 151,945 | 0.9681 | 1.2182 | 60.53% | 0.5992 / 0.6028 / 0.5949 | 0.6064 | 28m 08s | 6.46 GB |
| M1-03 | `0.2` | 40 | 32 | 357,129 | **0.9053** | 1.2215 | **63.77%** | **0.6384 / 0.6351 / 0.6250** | **0.6349** | 29m 58s | 5.84 GB |
| M1-04 | Không có layer Dropout (GAP) | 28 | 20 | 94,857 | 1.4063 | 1.6011 | 44.33% | 0.5013 / 0.4559 / 0.4300 | 0.4362 | 51m 46s* | 4.83 GB |

Tổng thời gian wall-clock: **1 giờ 50 phút 32 giây**.

> `*` Lần M1-04 này có một quãng chờ hệ thống/data-loader bất thường khoảng vài phút sau epoch 15. Vì vậy `51m 46s` là thời gian hoàn thành thực tế nhưng không nên dùng để kết luận M1-04 chậm hơn kiến trúc M1-03. Về compute, M1-04 có ít parameter và VRAM thấp hơn.

### F1 từng lớp trên test set

| Lớp (support) | M1-01 | M1-02 | M1-03 | M1-04 |
|---|---:|---:|---:|---:|
| Alu (99) | 0.0702 | 0.6702 | **0.7188** | 0.4091 |
| Carton (108) | 0.4216 | 0.7925 | **0.8279** | 0.6214 |
| Foam_box (87) | 0.1325 | **0.5276** | 0.5195 | 0.4365 |
| Milk_box (101) | 0.2938 | 0.6919 | **0.7980** | 0.5400 |
| Other (50) | 0.1935 | 0.4348 | **0.4923** | 0.3434 |
| PET (102) | 0.1538 | 0.5300 | **0.5556** | 0.2953 |
| Paper (106) | 0.3613 | 0.6842 | 0.6320 | 0.4762 |
| Paper_cup (105) | 0.2370 | 0.5532 | **0.6911** | 0.2857 |
| Plastic_cup (106) | 0.1111 | **0.4700** | 0.3902 | 0.4623 |

## 5. M1-01 — MLP control baseline

### File code và kiến trúc

- Config: `configs/m1_01.json`: input `128×128`, AdamW, `dropout_first=0.2`, `dropout=0.2`.
- Architecture: `src/model1/models/m1_01.py`, class `M1_01_MLP`.
- Workflow call: `m1_01.run()` gọi `run_model_workflow(..., is_cnn=False)`, vì vậy pipeline resize về `128×128` và không dùng channels-last.

```text
input                                        [B, 3, 128, 128]
Flatten                                     [B, 49,152]
Linear(49,152 -> 512) + ReLU                [B, 512]
Dropout(p=0.2), chỉ khi train                [B, 512]
Linear(512 -> 128) + ReLU                   [B, 128]
Dropout(p=0.2), chỉ khi train                [B, 128]
Linear(128 -> 9)                            [B, 9] logits
```

`Flatten` biến toàn bộ pixel thành một vector dài; Dense layer đầu tiên phải học quan hệ không gian giữa pixel từ đầu. Vì vậy model có tới 25.2 triệu parameter, chủ yếu ở `Linear(49,152 -> 512)`, nhưng không có inductive bias về cạnh, texture hoặc vị trí lân cận.

### Kết quả và nhận xét

- Early stopping ở epoch 41; checkpoint tốt nhất epoch 33.
- Test accuracy **23.73%**, macro F1 **0.2194**, test loss **2.0770**.
- Đây là baseline control: kết quả thấp xác nhận MLP thuần không phù hợp để tự học cấu trúc không gian từ ảnh rác 128×128, dù số parameter rất lớn.

## 6. M1-02 — CNN 2 blocks + BatchNorm + Dense head

### File code và kiến trúc

- Config: `configs/m1_02.json`: input `224×224`, Adam, Dropout `0.2`.
- Architecture: `src/model1/models/m1_02.py`, class `M1_02_CNN`.
- Block chung: `src/model1/blocks.py`, `conv_bn_relu_pool(3,32)` và `conv_bn_relu_pool(32,64)`.
- Workflow call: `m1_02.run()` gọi `run_model_workflow(..., is_cnn=True)`, kích hoạt memory format channels-last trên GPU.

```text
input                                                    [B, 3, 224, 224]
Conv(3 -> 32, 3x3, pad=1) -> BN -> ReLU                 [B, 32, 224, 224]
MaxPool2d(2)                                            [B, 32, 112, 112]
Conv(32 -> 64, 3x3, pad=1) -> BN -> ReLU                [B, 64, 112, 112]
MaxPool2d(2)                                            [B, 64, 56, 56]
AdaptiveAvgPool2d(4,4)                                  [B, 64, 4, 4]
Flatten                                                  [B, 1,024]
Linear(1,024 -> 128) -> ReLU -> Dropout(p=0.2)          [B, 128]
Linear(128 -> 9)                                        [B, 9] logits
```

`BatchNorm2d` chuẩn hóa activation theo từng channel, không đổi shape. `AdaptiveAvgPool2d(4,4)` cố định feature map về lưới `4×4`, giữ một phần thông tin vị trí đồng thời giúp Dense head không phụ thuộc cứng vào kích thước đầu vào.

### Kết quả và nhận xét

- Không dừng sớm: hoàn tất 50 epoch; checkpoint tốt nhất epoch 49.
- Test accuracy **60.53%**, macro F1 **0.5949**, test loss **1.2182**.
- So với M1-01, accuracy tăng **36.81 điểm phần trăm**, macro F1 tăng **0.3755**. Đây là mức cải thiện lớn nhất: convolution chia sẻ weight theo không gian nên học edge/texture hiệu quả hơn Dense/Flatten.

## 7. M1-03 — CNN 3 blocks + BatchNorm + Dense head

### File code và kiến trúc

- Config: `configs/m1_03.json`: giữ input, optimizer và Dropout `0.2` như M1-02.
- Architecture: `src/model1/models/m1_03.py`, class `M1_03_CNN`.
- So với M1-02, file này thêm duy nhất `conv_bn_relu_pool(64,128)`. Dense head vẫn là `AdaptiveAvgPool(4,4) -> Dense(128) -> Dropout(0.2) -> 9`, vì vậy phép so sánh cô lập tác động của block thứ ba.

```text
input                                                    [B, 3, 224, 224]
Block 1: Conv(3 -> 32), BN, ReLU, Pool                  [B, 32, 112, 112]
Block 2: Conv(32 -> 64), BN, ReLU, Pool                 [B, 64, 56, 56]
Block 3: Conv(64 -> 128), BN, ReLU, Pool                [B, 128, 28, 28]
AdaptiveAvgPool2d(4,4)                                  [B, 128, 4, 4]
Flatten                                                  [B, 2,048]
Linear(2,048 -> 128) -> ReLU -> Dropout(p=0.2)          [B, 128]
Linear(128 -> 9)                                        [B, 9] logits
```

Block thứ ba giảm kích thước không gian xuống `28×28` và tăng số channel lên 128, cho phép model tạo feature trừu tượng hơn trước Dense head. Parameter tăng từ 151,945 lên 357,129, phần lớn do `Linear(2,048 -> 128)`.

### Kết quả và nhận xét

- Early stopping ở epoch 40; checkpoint tốt nhất epoch 32.
- Test accuracy **63.77%**, macro F1 **0.6250**, weighted F1 **0.6349**, test loss **1.2215**.
- So với M1-02, accuracy tăng **3.24 điểm phần trăm**, macro F1 tăng **0.0301**. M1-03 là **Simple CNN tốt nhất** trong bốn model và là baseline phù hợp để so với Model 2/transfer learning.
- Test loss cao hơn M1-02 một chút dù accuracy/F1 tốt hơn: Cross-Entropy còn phạt mạnh các dự đoán sai nhưng rất tự tin. Vì vậy cần đọc loss cùng accuracy và macro F1, không suy luận từ một metric duy nhất.

## 8. M1-04 — CNN 3 blocks + Global Average Pooling

### File code và kiến trúc

- Config: `configs/m1_04.json`: schema có `dropout=0.2` để thống nhất config, nhưng `m1_04.py` cố ý không tạo `nn.Dropout`.
- Architecture: `src/model1/models/m1_04.py`, class `M1_04_CNN_GAP`.
- Backbone ba block giống hệt M1-03. Khác biệt duy nhất là classifier head thay `AdaptiveAvgPool(4,4) -> Dense(128) -> Dropout -> 9` bằng GAP trực tiếp.

```text
input                                                    [B, 3, 224, 224]
3 x [Conv -> BatchNorm -> ReLU -> MaxPool]              [B, 128, 28, 28]
AdaptiveAvgPool2d(1,1) / Global Average Pooling         [B, 128, 1, 1]
Flatten                                                  [B, 128]
Linear(128 -> 9)                                        [B, 9] logits
```

GAP lấy trung bình toàn bộ `28×28 = 784` vị trí trên từng channel. Head còn 1,161 parameter (`128×9 + 9`) thay vì Dense head lớn của M1-03. Tổng parameter giảm từ 357,129 xuống **94,857** (giảm khoảng **73.4%**) và peak VRAM giảm khoảng **1.01 GB**.

### Kết quả và nhận xét

- Early stopping ở epoch 28; checkpoint tốt nhất epoch 20.
- Test accuracy **44.33%**, macro F1 **0.4300**, test loss **1.6011**.
- So với M1-03, accuracy giảm **19.44 điểm phần trăm**, macro F1 giảm **0.1950**. GAP nén spatial information quá sớm; lưới `4×4` của Dense head M1-03 phù hợp hơn với dataset này.
- M1-04 là lựa chọn model nhẹ hơn khi hạn chế parameter/VRAM; không phải lựa chọn tốt nhất nếu mục tiêu chính là accuracy và macro F1.

## 9. Kết luận dùng cho báo cáo

1. **M1-01** là baseline Dense/Flatten yếu: nhiều parameter không bù được việc mất cấu trúc không gian.
2. **M1-02** chứng minh CNN + BatchNorm + Dense head là bước nâng cấp quan trọng nhất, tăng 36.81 điểm phần trăm accuracy so với MLP.
3. **M1-03** cô lập tác động của block Conv thứ ba và đạt metric tốt nhất: test accuracy **63.77%**, macro F1 **0.6250**.
4. **M1-04** cô lập tác động của GAP head: model nhẹ hơn rất nhiều nhưng metric giảm mạnh, nên giảm parameter không đồng nghĩa tăng chất lượng phân loại.
5. Các JSON nguồn của bảng này là `reports/model1/m1_01.json` đến `m1_04.json`; các lần chạy/tinh chỉnh sau cần được ghi là run mới, không thay số liệu của run hiện tại.
