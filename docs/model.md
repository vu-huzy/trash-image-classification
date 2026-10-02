# Kế hoạch Model 1 (Simple NN) & Model 2 (Complex CNN) — VN Trash Classification

---

# A. TÓM TẮT

## A.1 Phạm vi

- Dataset: 9 lớp — `Alu`, `Carton`, `Foam_box`, `Milk_box`, `Other`, `PET`, `Paper`, `Paper_cup`, `Plastic_cup`.
- **Model 1 (Simple NN/CNN):** một nhánh tuần tự duy nhất, không residual, không parallel, train từ đầu. 4 biến thể M1‑01 → M1‑04.
- **Model 2 (Complex CNN):** có residual/skip connection và (từ M2‑05) parallel block đa kernel, kế thừa backbone M1‑04, train từ đầu. 6–7 biến thể M2‑01 → M2‑07.
- Model 3 (transfer learning/fine‑tune) không thuộc phạm vi tài liệu này.
- Mỗi biến thể chỉ đổi đúng một yếu tố so với biến thể ngay trước.
- **BatchNorm và Dropout dùng tự do** trong mọi model có CNN (không tính là biến ablation riêng). Biến so sánh chính thức chỉ gồm: độ sâu (số block/stage), kiểu pooling/head, và kiểu khối (plain/residual/parallel/attention).

## A.2 Chuẩn chung

| Hạng mục       | Model 1                                                       | Model 2                                                       |
| -------------- | ------------------------------------------------------------- | ------------------------------------------------------------- |
| Split          | Train `9,542`, val `1,685`, test `864`; stratified, `seed=42` | Giống Model 1                                                 |
| Input          | M1‑01: `128×128`; M1‑02→M1‑04: `224×224`                      | `224×224` toàn bộ                                             |
| Cache          | `data/image_cache_224`; resize 1 lần, flip/normalize trên GPU | Giống Model 1                                                 |
| Batch size     | `256`                                                         | Bắt đầu `128`, thử `64` nếu OOM (không giảm resolution trước) |
| Optimizer      | Adam/AdamW, `lr=1e-3`                                         | Giống Model 1                                                 |
| Loss           | CrossEntropyLoss                                              | Giống Model 1                                                 |
| Precision      | AMP FP16, TF32, channels‑last, cudnn benchmark                | Giống Model 1                                                 |
| Epoch tối đa   | `50`                                                          | Giống Model 1                                                 |
| Early stopping | Monitor validation loss, `patience=8`, `min_delta=0.001`      | Giống Model 1                                                 |
| Checkpoint     | Val loss thấp nhất, không phải epoch cuối                     | Giống Model 1                                                 |
| Test           | Chỉ test checkpoint tốt nhất, đúng 1 lần                      | Giống Model 1                                                 |

### Quy ước shape và luồng dữ liệu (áp dụng cả 2 model)

- PyTorch dùng thứ tự tensor `[B, C, H, W]`. Batch 256 ảnh RGB 224×224 là `[256, 3, 224, 224]`.
- `Conv(k=3, stride=1, padding=1)`, BatchNorm, ReLU, Dropout không đổi shape không gian.
- `MaxPool2d(2,2)` giảm mỗi chiều không gian còn một nửa: `224→112→56→28→14`.
- `AdaptiveAvgPool2d(4,4)` ép spatial về `4×4`; Global Average Pooling (GAP) ép về `1×1` rồi bỏ 2 chiều đó.
- `Flatten` biến `[B,C,H,W]` thành `[B, C×H×W]`; `Linear(a,b)` biến `[B,a]` thành `[B,b]`.
- Cache ban đầu `uint8 [B,3,224,224]` trên CPU → GPU → `float32` → chia `255` → random flip khi train → normalize ImageNet. Tất cả giữ nguyên shape `[B,3,224,224]`.
- Label `y: [B]`, mỗi phần tử `0..8`. Mọi model xuất `logits: [B,9]`; CrossEntropyLoss nhận `(logits, y)`, trả về scalar `[]`.
- Kernel chẵn (vd. 2×2) không có "same" padding đối xứng: cần pad thủ công (vd. 1 pixel phải & dưới) trước conv để giữ nguyên spatial size.

### Công thức shape Conv2d

```text
output = floor((input + 2×padding − dilation×(kernel−1) − 1) / stride + 1)
```

Conv 3×3 mặc định `stride=1, padding=1` → output = input. Khi `stride=2` (downsample chính) → `224→112`, `112→56`, `56→28`.

## A.3 Chỉ số phải lưu cho từng model

- Train/validation loss và accuracy mỗi epoch.
- Epoch tốt nhất, thời gian train tổng, peak VRAM, số parameter trainable.
- Test accuracy, macro precision, macro recall, macro F1, weighted F1.
- Confusion matrix 9×9; ghi rõ các cặp lớp dễ nhầm nhất.
- Per‑class recall, đặc biệt cho `PET`, `Plastic_cup`, `Paper_cup`, `Milk_box`, `Carton`.

## A.4 Quy tắc ra quyết định khi chuyển model

- Chấp nhận model mới khi macro F1 hoặc validation loss tốt hơn rõ rệt, ổn định qua ≥2 seed (`42`, `2026`).
- Nếu metric xấu đi, giữ log nhưng quay lại checkpoint/kiến trúc tốt nhất trước đó.
- Nếu `train acc − val acc > 12%`, ưu tiên regularization/augmentation thay vì tăng độ phức tạp tiếp.
- Nếu OOM, giảm batch size trước; không giảm resolution trước khi có benchmark chứng minh cần thiết.

## A.5 Mốc baseline hiện có

Kiến trúc 3 conv block tuần tự + BN + AdaptiveAvgPool(4,4) + Dense head — tương ứng **M1‑03**:

- Best validation accuracy: `69.67%` tại epoch 31.
- Test accuracy: `62.15%`.
- Thời gian: ~`27 phút 44 giây` cho 39 epoch (~`6 phút 34 giây`/10 epoch).

Đây là mốc phải vượt qua khi nâng cấp. `src/model1/test.py` cần implement lại M1‑03 trước khi dùng để tái lập baseline.

## A.6 Bảng tổng hợp Model 1

| ID    | Ý tưởng chính                        | Độ phức tạp | Vai trò                                      |
| ----- | ------------------------------------ | ----------: | -------------------------------------------- |
| M1‑01 | Flatten + MLP                        |    Rất thấp | Control baseline                             |
| M1‑02 | CNN 2 blocks (BN + Dropout mặc định) |        Thấp | Chứng minh CNN (kèm BN/Dropout) vượt MLP     |
| M1‑03 | CNN 3 blocks (BN + Dropout)          |  Trung bình | Đo riêng block Conv thứ ba; baseline hiện có |
| M1‑04 | CNN 3 blocks + GAP                   |  Trung bình | Đo riêng tác động của Global Average Pooling |

## A.7 Bảng tổng hợp Model 2

| ID                 | Ý tưởng chính                                             |     Độ phức tạp | Vai trò                                     |
| ------------------ | --------------------------------------------------------- | --------------: | ------------------------------------------- |
| M2‑01              | Residual tối giản (1 conv + shortcut)                     | Trung bình thấp | Đo riêng tác động của phép cộng residual    |
| M2‑02              | Residual block chuẩn (2 conv + shortcut)                  |      Trung bình | Đo riêng tác động của "BasicBlock" chuẩn    |
| M2‑03              | 2 residual unit/stage                                     |  Trung bình cao | Đo riêng tác động tăng depth trong residual |
| M2‑04              | Thêm stage 4 (ResNet‑18 rút gọn)                          |             Cao | Xem giới hạn tăng depth trên dataset nhỏ    |
| M2‑05              | Parallel block đa kernel (2x2/3x3/5x5/pool) thay stage 4  |             Cao | Câu hỏi cốt lõi: sequential vs parallel     |
| M2‑06              | + SE block (channel attention)                            |             Cao | Đo riêng tác động của attention             |
| M2‑07 _(tuỳ chọn)_ | Cardinality (nhánh nhỏ song song, cùng ngân sách tham số) |             Cao | Kiểm chứng ý tưởng ResNeXt ở quy mô nhỏ     |

## A.8 Thiết kế đã cân nhắc và loại bỏ khỏi Model 2

| Thiết kế                                                       | Lý do loại bỏ                                                                                                                                                                                                        |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Parallel branches cùng kernel size, cùng số filter, rồi concat | Về toán học tương đương một Conv2D duy nhất filter lớn hơn (ReLU tính theo channel độc lập; ghép nhiều nhánh cùng kernel không tương tác = 1 conv với filter xếp chồng). Đo "tăng width", không đo "xử lý song song" |
| ResNet‑18/ResNet‑50 train from scratch                         | ~25 triệu tham số (ResNet‑50) cần dataset lớn hơn nhiều so với ~12–13 nghìn ảnh hiện có; rủi ro overfit cao. Vai trò này thuộc về Model 3 (pretrained)                                                               |

## A.9 Cấu trúc mã

```text
src/
├── common/
│   ├── data.py              # cache/raw dataset, split cố định, transforms
│   ├── engine.py            # train_one_epoch, evaluate, early stopping
│   ├── metrics.py           # F1, confusion matrix, report
│   └── utils.py             # seed, checkpoint, logging, timer
├── model1/
│   ├── models/
│   │   ├── m1_01.py ... m1_04.py  # mỗi kiến trúc Simple NN/CNN một file
│   ├── blocks.py            # Conv-BN-ReLU-MaxPool dùng chung
│   ├── workflow.py          # train/evaluate/checkpoint/report dùng chung
│   ├── registry.py          # config architecture -> module model
│   ├── train.py             # CLI điều phối
│   └── test.py              # entry point ngắn gọn
├── model2/
│   ├── blocks.py            # ResidualBlock, ParallelBlock (đa kernel), SEBlock
│   ├── architectures.py     # Complex CNN M2-01 ... M2-07
│   └── train.py
├── configs/
│   ├── m1_01.json ... m1_04.json
│   └── m2_01.json ... m2_07.json
checkpoints/
├── model1/
└── model2/
reports/
├── experiments.csv
└── figures/
```

### Trình tự triển khai cho mỗi model

1. Sao chép config của model liền trước.
2. Chỉ sửa `architecture` và hyperparameter được mô tả ở model đó.
3. Smoke test: `--epochs 1 --max-batches 2` — kiểm tra shape, cache, CUDA, VRAM.
4. Train thật tối đa 50 epoch với early stopping.
5. Load checkpoint tốt nhất, test đúng 1 lần, lưu report/figure.
6. Cập nhật `reports/experiments.csv`, ghi quyết định: giữ, bỏ, hoặc dùng làm nền cho model tiếp theo.

---

# B. CHI TIẾT

# B.1 Model 1 — Simple NN / Simple CNN

Một nhánh tuần tự duy nhất: không residual, không parallel, không pretrained. BatchNorm và Dropout dùng tự do ngay từ model CNN đầu tiên, không tách thành bước ablation riêng. Từ M1‑02 dùng cùng ảnh `224×224`, split, cache/pipeline GPU, optimizer, batch `256`, early stopping, augmentation, hidden head `128` (trừ M1‑04).

## M1‑01 — MLP control baseline

```text
Input 128×128×3
→ Flatten
→ Linear(49,152, 512) → ReLU → Dropout(0.2)
→ Linear(512, 128) → ReLU → Dropout(0.2)
→ Linear(128, 9)
```

| Bước | Layer/thao tác                 | Shape output         | Ý nghĩa                                          |
| ---: | ------------------------------ | -------------------- | ------------------------------------------------ |
|    0 | Ảnh RGB sau resize + normalize | `[256, 3, 128, 128]` | 256 ảnh, 3 channel màu                           |
|    1 | Flatten                        | `[256, 49,152]`      | `3×128×128=49,152`; mất thông tin vị trí lân cận |
|    2 | Linear `49,152→512`            | `[256, 512]`         | Vector 512 đặc trưng                             |
|    3 | ReLU + Dropout(0.2)            | `[256, 512]`         | Không đổi shape                                  |
|    4 | Linear `512→128`               | `[256, 128]`         | Nén feature                                      |
|    5 | ReLU + Dropout(0.2)            | `[256, 128]`         | Regularization                                   |
|    6 | Linear `128→9`                 | `[256, 9]`           | Logits                                           |
|    7 | CrossEntropyLoss với `[256]`   | `[]`                 | Loss scalar                                      |

- Không có feature map `[B,C,H,W]` sau Flatten → model không biết quan hệ không gian giữa các pixel.
- Dense đầu tiên có `49,152 × 512` weight → dễ overfit dù kiến trúc đơn giản.
- Optimizer AdamW, `lr=1e-3`, `weight_decay=1e-4`.
- **So sánh chính:** control để CNN chứng minh có hơn MLP hay không.

## M1‑02 — CNN 2 blocks (BatchNorm + Dropout mặc định)

**Thay đổi duy nhất so với M1‑01:** chuyển sang feature extractor CNN 2 block, có BatchNorm và Dropout dùng tự do ngay từ đầu (không tách bước "chưa có BN").

```text
Conv(3, 32, 3×3, padding=1) → BN → ReLU → MaxPool(2)
Conv(32, 64, 3×3, padding=1) → BN → ReLU → MaxPool(2)
AdaptiveAvgPool(4×4)
Flatten → Dense(128) → ReLU → Dropout(0.2) → Dense(9)
```

| Bước | Layer/thao tác                          | Shape output          | Giải thích                         |
| ---: | --------------------------------------- | --------------------- | ---------------------------------- |
|    0 | Input                                   | `[256, 3, 224, 224]`  |                                    |
|    1 | Conv `3→32` + BN + ReLU                 | `[256, 32, 224, 224]` | Channel `3→32`, spatial giữ nguyên |
|    2 | MaxPool 2×2                             | `[256, 32, 112, 112]` |                                    |
|    3 | Conv `32→64` + BN + ReLU                | `[256, 64, 112, 112]` | Block conv thứ hai                 |
|    4 | MaxPool 2×2                             | `[256, 64, 56, 56]`   |                                    |
|    5 | AdaptiveAvgPool(4,4)                    | `[256, 64, 4, 4]`     |                                    |
|    6 | Flatten                                 | `[256, 1,024]`        | `64×4×4=1,024`                     |
|    7 | Dense `1,024→128` + ReLU + Dropout(0.2) | `[256, 128]`          |                                    |
|    8 | Dense `128→9`                           | `[256, 9]`            | Logits                             |

- Batch `256`, Adam `lr=1e-3`.
- **So sánh chính:** M1‑02 so với M1‑01 trả lời: CNN 2 block (kèm BN/Dropout mặc định) có vượt MLP rõ rệt không?
- **Rủi ro:** underfitting nếu feature extractor vẫn còn nông so với độ đa dạng của 9 lớp rác.

## M1‑03 — CNN 3 blocks (baseline hiện có)

**Thay đổi duy nhất so với M1‑02:** thêm block Conv thứ ba `64→128`.

```text
Conv(3, 32, 3×3) → BN → ReLU → MaxPool(2)
Conv(32, 64, 3×3) → BN → ReLU → MaxPool(2)
Conv(64, 128, 3×3) → BN → ReLU → MaxPool(2)
AdaptiveAvgPool(4×4)
Flatten → Dense(128) → ReLU → Dropout(0.2) → Dense(9)
```

| Bước | Layer/thao tác                          | Shape output          | Giải thích                                |
| ---: | --------------------------------------- | --------------------- | ----------------------------------------- |
|    0 | Input                                   | `[256, 3, 224, 224]`  |                                           |
|    1 | Conv `3→32` + BN + ReLU                 | `[256, 32, 224, 224]` |                                           |
|    2 | MaxPool                                 | `[256, 32, 112, 112]` |                                           |
|    3 | Conv `32→64` + BN + ReLU                | `[256, 64, 112, 112]` |                                           |
|    4 | MaxPool                                 | `[256, 64, 56, 56]`   |                                           |
|    5 | Conv `64→128` + BN + ReLU               | `[256, 128, 56, 56]`  | **Block Conv thứ ba — thay đổi duy nhất** |
|    6 | MaxPool                                 | `[256, 128, 28, 28]`  |                                           |
|    7 | AdaptiveAvgPool(4,4)                    | `[256, 128, 4, 4]`    |                                           |
|    8 | Flatten                                 | `[256, 2,048]`        | `128×4×4=2,048`                           |
|    9 | Dense `2,048→128` + ReLU + Dropout(0.2) | `[256, 128]`          |                                           |
|   10 | Dense `128→9`                           | `[256, 9]`            | Logits                                    |

- Đây là architecture baseline `62.15%` test accuracy (early stopping epoch 39; trước đây gọi là M1‑05 trong bản roadmap cũ).
- **So sánh chính:** M1‑03 so với M1‑02 trả lời: block Conv thứ ba có làm feature sâu hơn hữu ích không?

## M1‑04 — CNN 3 blocks + Global Average Pooling

**Thay đổi duy nhất so với M1‑03:** thay `AdaptiveAvgPool2d(4,4) → Flatten → Dense(2,048→128)` bằng `AdaptiveAvgPool2d(1,1)` (GAP) → `Dense(128→9)`.

```text
Conv(3, 32, 3×3) → BN → ReLU → MaxPool(2)
Conv(32, 64, 3×3) → BN → ReLU → MaxPool(2)
Conv(64, 128, 3×3) → BN → ReLU → MaxPool(2)
GlobalAveragePool
Dense(128, 9)
```

| Bước | Layer/thao tác                 | Shape output          | Giải thích                                       |
| ---: | ------------------------------ | --------------------- | ------------------------------------------------ |
|    0 | Input                          | `[256, 3, 224, 224]`  |                                                  |
|    1 | Conv `3→32` + BN + ReLU        | `[256, 32, 224, 224]` | Backbone không đổi                               |
|    2 | MaxPool                        | `[256, 32, 112, 112]` |                                                  |
|    3 | Conv `32→64` + BN + ReLU       | `[256, 64, 112, 112]` |                                                  |
|    4 | MaxPool                        | `[256, 64, 56, 56]`   |                                                  |
|    5 | Conv `64→128` + BN + ReLU      | `[256, 128, 56, 56]`  |                                                  |
|    6 | MaxPool                        | `[256, 128, 28, 28]`  |                                                  |
|    7 | `AdaptiveAvgPool2d(1,1)` / GAP | `[256, 128, 1, 1]`    | Mỗi feature map `28×28` còn 1 giá trị trung bình |
|    8 | Flatten                        | `[256, 128]`          |                                                  |
|    9 | Dense `128→9`                  | `[256, 9]`            | Không cần hidden dense lớn                       |

- Khác duy nhất với M1‑03: `AdaptiveAvgPool(4,4)→Dense(2,048→128→9)` thay bằng `GAP→Dense(128→9)`.
- GAP giảm mạnh parameter classifier, thường giảm overfit, nhưng có thể mất thông tin vị trí chi tiết.
- **So sánh chính:** M1‑04 so với M1‑03 trả lời: Global Average Pooling có tốt hơn AdaptiveAvgPool(4,4) + dense head lớn không?
- **Dừng nâng cấp Model 1 sau M1‑04** — các ý tưởng residual/parallel chuyển sang Model 2.

---

# B.2 Model 2 — Complex CNN

Kế thừa backbone M1‑04 (3 block Conv‑BN‑ReLU‑Pool, channel 32→64→128, GAP, Dense(128→9)). Nếu M1‑03 thắng M1‑04 sau khi chạy thật, chỉ đổi phần head ở mọi biến thể dưới đây — backbone 3 block giữ nguyên.

## M2‑01 — Residual tối giản

**Thay đổi duy nhất so với M1‑04:** mỗi block cộng thêm nhánh shortcut (Conv1x1) song song với nhánh Conv chính, cộng lại rồi mới ReLU.

```text
Block k:  y = ReLU( BN(Conv3x3(x)) + Conv1x1_shortcut(x) )
          MaxPool(2)
```

| Bước | Layer/thao tác                                            | Shape output       | Giải thích                              |
| ---: | --------------------------------------------------------- | ------------------ | --------------------------------------- |
|    0 | Input                                                     | `[128,3,224,224]`  |                                         |
|    1 | Conv3x3(3→32)+BN, và Conv1x1(3→32) shortcut, cộng, ReLU   | `[128,32,224,224]` | Hai nhánh cùng shape, cộng element‑wise |
|    2 | MaxPool                                                   | `[128,32,112,112]` |                                         |
|    3 | Conv3x3(32→64)+BN + shortcut Conv1x1(32→64), cộng, ReLU   | `[128,64,112,112]` |                                         |
|    4 | MaxPool                                                   | `[128,64,56,56]`   |                                         |
|    5 | Conv3x3(64→128)+BN + shortcut Conv1x1(64→128), cộng, ReLU | `[128,128,56,56]`  |                                         |
|    6 | MaxPool                                                   | `[128,128,28,28]`  |                                         |
|    7 | GAP                                                       | `[128,128]`        |                                         |
|    8 | Dense 128→9                                               | `[128,9]`          | Logits                                  |

- Shortcut `Conv1x1` không BN/ReLU, chỉ khớp channel.
- **So sánh chính:** phép cộng residual (chưa thêm lớp) có giúp gradient flow/hội tụ tốt hơn không?

## M2‑02 — Residual block chuẩn (2 conv/block)

**Thay đổi duy nhất so với M2‑01:** mỗi block có 2 lớp Conv3x3 trước khi cộng shortcut (BasicBlock).

```text
Block k:  h = ReLU(BN(Conv3x3_1(x)))
          y = BN(Conv3x3_2(h))
          out = ReLU(y + Shortcut(x))     # Conv1x1 nếu đổi channel, identity nếu không
          MaxPool(2)
```

| Bước | Layer/thao tác                                     | Shape output               | Giải thích              |
| ---: | -------------------------------------------------- | -------------------------- | ----------------------- |
|    0 | Input                                              | `[128,3,224,224]`          |                         |
|    1 | Conv3x3(3→32)+BN+ReLU                              | `[128,32,224,224]`         | Conv thứ nhất           |
|    2 | Conv3x3(32→32)+BN                                  | `[128,32,224,224]`         | Conv thứ hai, chưa ReLU |
|    3 | Shortcut Conv1x1(3→32)                             | `[128,32,224,224]`         | Nhánh chiếu             |
|    4 | Cộng (2)+(3), rồi ReLU                             | `[128,32,224,224]`         | Residual add            |
|    5 | MaxPool                                            | `[128,32,112,112]`         |                         |
|    — | Block 2 (32→64), Block 3 (64→128) lặp lại đúng mẫu | `[128,128,28,28]` sau pool |                         |
|    8 | GAP → Dense 128→9                                  | `[128,9]`                  |                         |

- **So sánh chính:** residual block chuẩn (2 conv) có tốt hơn residual tối giản (1 conv) không?

## M2‑03 — 2 residual unit/stage

**Thay đổi duy nhất so với M2‑02:** mỗi stage (32/64/128) có 2 residual unit liên tiếp thay vì 1.

```text
Stage k:  Unit1 (in→out, shortcut chiếu)
          Unit2 (out→out, shortcut identity)
          MaxPool(2)
```

| Bước | Layer/thao tác                   | Shape output       | Giải thích                         |
| ---: | -------------------------------- | ------------------ | ---------------------------------- |
|    0 | Input                            | `[128,3,224,224]`  |                                    |
|    1 | Stage1 Unit1 (3→32)              | `[128,32,224,224]` |                                    |
|    2 | Stage1 Unit2 (32→32, identity)   | `[128,32,224,224]` | **Lớp mới duy nhất trong stage 1** |
|    3 | MaxPool                          | `[128,32,112,112]` |                                    |
|    4 | Stage2 Unit1 (32→64)             | `[128,64,112,112]` |                                    |
|    5 | Stage2 Unit2 (64→64, identity)   | `[128,64,112,112]` | Lớp mới                            |
|    6 | MaxPool                          | `[128,64,56,56]`   |                                    |
|    7 | Stage3 Unit1 (64→128)            | `[128,128,56,56]`  |                                    |
|    8 | Stage3 Unit2 (128→128, identity) | `[128,128,56,56]`  | Lớp mới                            |
|    9 | MaxPool                          | `[128,128,28,28]`  |                                    |
|   10 | GAP → Dense 128→9                | `[128,9]`          |                                    |

- **So sánh chính:** thêm 1 residual unit mỗi stage có tiếp tục cải thiện hay đã bão hòa/overfit?
- Nếu `train acc − val acc > 12%` ở bước này, ưu tiên regularization trước khi sang M2‑04.

## M2‑04 — Thêm stage 4 (256 channel, ResNet‑18 rút gọn)

**Thay đổi duy nhất so với M2‑03:** thêm 1 stage (2 residual unit, 128→256).

```text
... (Stage 1-3 giống M2-03, output [128,128,28,28]) ...
Stage4 Unit1 (128→256, shortcut chiếu)
Stage4 Unit2 (256→256, identity)
MaxPool(2)
GAP → Dense(256→9)
```

|      Bước | Layer/thao tác         | Shape output      | Giải thích                      |
| --------: | ---------------------- | ----------------- | ------------------------------- |
| (kế thừa) | ...hết Stage3          | `[128,128,28,28]` |                                 |
|        11 | Stage4 Unit1 (128→256) | `[128,256,28,28]` | Stage mới                       |
|        12 | Stage4 Unit2 (256→256) | `[128,256,28,28]` | Lớp mới                         |
|        13 | MaxPool                | `[128,256,14,14]` | Downsample thứ 4                |
|        14 | GAP                    | `[128,256]`       | Channel vào Dense đổi thành 256 |
|        15 | Dense 256→9            | `[128,9]`         | Logits                          |

- **So sánh chính:** sâu thêm 1 stage (≈ResNet‑18 rút gọn) có còn lợi, hay overfit rõ trên ~12–13k ảnh?
- **Điểm rẽ nhánh:** nếu train‑val gap tăng mạnh so với M2‑03 → dừng tại M2‑03, dùng M2‑03 (không phải M2‑04) làm nền cho M2‑05.

## M2‑05 — Parallel block đa kernel thay stage 4

**Thay đổi duy nhất so với M2‑04:** giữ Stage 1‑3, thay Stage 4 bằng 1 Parallel block 4 nhánh kernel khác nhau (**2x2/3x3/5x5/pool**), giữ nguyên shape vào/ra `[128,256,14,14]` để so sánh 1‑biến đúng nghĩa.

```text
Input stage4: [128,128,28,28]
  ├─ Nhánh 2x2:  ZeroPad2d(0,1,0,1) → Conv2d(128→64, kernel=2x2, stride=1, padding=0)
  ├─ Nhánh 3x3:  Conv1x1(128→32) → Conv3x3(32→64, pad=1)
  ├─ Nhánh 5x5:  Conv1x1(128→32) → Conv5x5(32→64, pad=2)
  └─ Nhánh pool: MaxPool3x3(stride=1,pad=1) → Conv1x1(128→64)
Concat 4 nhánh (64×4 = 256 channel, vẫn 28×28)
MaxPool(2) → [128,256,14,14]
GAP → Dense(256→9)
```

|      Bước | Layer/thao tác                                      | Shape output      | Giải thích                                                                                             |
| --------: | --------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------ |
| (kế thừa) | ...hết Stage3                                       | `[128,128,28,28]` |                                                                                                        |
|       11a | Nhánh 2x2: ZeroPad2d(0,1,0,1) → Conv2d k=2 (128→64) | `[128,64,28,28]`  | Kernel chẵn không có "same" đối xứng — pad thêm 1 pixel (phải & dưới) trước conv để giữ nguyên `28×28` |
|       11b | Nhánh 3x3: Conv1x1(128→32)→Conv3x3(32→64)           | `[128,64,28,28]`  | Receptive field vừa                                                                                    |
|       11c | Nhánh 5x5: Conv1x1(128→32)→Conv5x5(32→64)           | `[128,64,28,28]`  | Receptive field lớn                                                                                    |
|       11d | Nhánh pool: MaxPool3x3(s=1)→Conv1x1(128→64)         | `[128,64,28,28]`  | Ngữ cảnh rộng                                                                                          |
|        12 | Concat 4 nhánh                                      | `[128,256,28,28]` | Cùng 256 channel như M2‑04, khác cách tạo                                                              |
|        13 | MaxPool                                             | `[128,256,14,14]` | Giống hệt shape M2‑04                                                                                  |
|        14 | GAP → Dense 256→9                                   | `[128,9]`         | Giống hệt đầu ra M2‑04                                                                                 |

- Kết hợp sequential + parallel thật sự (4 nhánh khác kernel size: 2×2, 3×3, 5×5, pool).
- **So sánh chính:** ở cùng vị trí, khối đa tỷ lệ (parallel) có tốt hơn xếp thêm 1 stage residual tuần tự không? — câu hỏi cốt lõi của Model 2.

## M2‑06 — Thêm SE block (channel attention)

**Thay đổi duy nhất so với M2‑05:** chèn Squeeze‑and‑Excitation ngay sau Concat.

```text
... Concat 4 nhánh → [128,256,28,28] (giống M2-05, nhánh 2x2/3x3/5x5/pool)
SE block:
  Squeeze: GlobalAvgPool → [128,256,1,1] → Flatten [128,256]
  Excite:  Dense(256→16)→ReLU→Dense(16→256)→Sigmoid → [128,256]
  Scale:   nhân broadcast với feature map gốc → [128,256,28,28]
MaxPool(2) → GAP → Dense(256→9)
```

|      Bước | Layer/thao tác                        | Shape output      | Giải thích                   |
| --------: | ------------------------------------- | ----------------- | ---------------------------- |
| (kế thừa) | Concat 4 nhánh                        | `[128,256,28,28]` | Giống M2‑05                  |
|       12a | Squeeze: GAP + Flatten                | `[128,256]`       | Tóm tắt mỗi channel          |
|       12b | Excite: Dense 256→16→256 + Sigmoid    | `[128,256]`       | Trọng số 0–1/channel         |
|       12c | Scale: nhân element‑wise theo channel | `[128,256,28,28]` | Cân chỉnh theo độ quan trọng |
|        13 | MaxPool → GAP → Dense                 | `[128,9]`         | Giống hệt M2‑05              |

- **So sánh chính:** channel attention (SE) có cải thiện thêm so với chỉ có đa tỷ lệ (parallel) không?

## M2‑07 (tuỳ chọn) — Cardinality: nhánh nhỏ song song cùng ngân sách tham số

Thay Stage 3 (64→128, 1 unit) bằng 3 nhánh residual song song, mỗi nhánh 64→43 channel (~1/3), concat lại đúng 128 channel — tổng tham số/tính toán gần tương đương 1 nhánh đơn, đúng tinh thần "cardinality" (ResNeXt). Chỉ thực hiện nếu M2‑06 đã ổn định và còn thời gian; không bắt buộc trong chuỗi chính.

- **So sánh chính:** chia nhóm (kiểu ResNeXt) có tốt hơn 1 nhánh lớn cùng ngân sách tham số không?
