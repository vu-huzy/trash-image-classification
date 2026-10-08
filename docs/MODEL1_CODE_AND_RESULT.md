# Model 1 — kết quả huấn luyện

## Mục đích

Model 1 so sánh cách một mạng Dense thuần và các CNN đơn giản phân loại 9 loại rác.
Giải thích ngắn về data loader, shape, kiến trúc, training, validation và test được đặt
trực tiếp ở các comment trong từng file code:

- `src/model1/m1_01.py` — MLP cơ bản.
- `src/model1/m1_02.py` — CNN 2 block.
- `src/model1/m1_03.py` — CNN 3 block.
- `src/model1/m1_04.py` — CNN 3 block với Global Average Pooling.
- `src/model1/metrics.py` — các metric dùng chung.

## Cấu hình chung

- Thiết bị: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA), Python 3.11.9.
- Dữ liệu: `data/VN_trash_classification_preprocessing`.
- Split: train 9.542 ảnh, validation 1.685 ảnh, test 864 ảnh; 9 class.
- Batch size: 64; 50 epoch; Adam (`lr=0.001`); `CrossEntropyLoss`; seed 42.
- Train: Resize, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize.
- Validation/test: Resize, ToTensor, Normalize; không augmentation.
- Chọn model: validation loss nhỏ nhất. Test chỉ chạy một lần sau khi chọn checkpoint.

## Kiến trúc

| Model | Kiến trúc | Input | Parameter |
|---|---|---|---:|
| M1-01 | Flatten → Dense 512 → Dense 128 → 9 logits | `[B,3,128,128]` | 25.233.161 |
| M1-02 | 2 × (Conv + BatchNorm + ReLU + MaxPool) → Dense head | `[B,3,224,224]` | 151.945 |
| M1-03 | 3 × (Conv + BatchNorm + ReLU + MaxPool) → Dense head | `[B,3,224,224]` | 357.129 |
| M1-04 | 3 × (Conv + BatchNorm + ReLU + MaxPool) → GAP → Linear | `[B,3,224,224]` | 94.857 |

M1-01, M1-02 và M1-03 dùng Dropout = 0.2 trong Dense head. M1-04 không có Dense
hidden layer nên không dùng Dropout: GAP biến `[B,128,28,28]` thành `[B,128]`, sau đó
Linear tạo trực tiếp `[B,9]`.

## Kết quả chạy lại ngày 07/10/2026

| Model | Best epoch | Best val loss | Test loss | Test accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Tổng thời gian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1-01 | 19 | 2.0645 | 2.1733 | 18.06% | 0.2121 | 0.1641 | 0.0926 | 0.1011 | 1,350.43s |
| M1-02 | 48 | 1.1314 | 1.4478 | 56.37% | 0.5707 | 0.5643 | 0.5565 | 0.5681 | 2,560.40s |
| M1-03 | 42 | 1.0360 | 1.2068 | **60.30%** | 0.6048 | 0.6071 | **0.5916** | **0.6017** | 2,294.71s |
| M1-04 | 48 | 1.2664 | 1.3845 | 52.66% | 0.5353 | 0.5352 | 0.5112 | 0.5120 | 2,921.26s |

## Nhận xét

- CNN vượt MLP rất rõ: M1-02 tăng 38.31 điểm phần trăm test accuracy so với M1-01.
- Thêm block Conv thứ ba giúp M1-03 tăng 3.93 điểm phần trăm accuracy và 0.0351 Macro F1 so với M1-02.
- M1-04 dùng GAP nên ít hơn M1-03 262.272 parameter (giảm 73.4%), nhưng accuracy thấp hơn 7.64 điểm phần trăm trong lượt chạy này.
- M1-03 và M1-04 thay cả Dense head/Dropout bằng GAP + Linear, vì vậy chưa thể kết luận phần chênh lệch chỉ do GAP. Muốn cô lập GAP, cần một lượt CNN 3 block giữ nguyên head nhưng chỉ đổi phép pooling.

## Artifact

Mỗi model đã lưu các file tại `reports/model1_preprocessed/`:

- `M1-0x_history.json`: loss, accuracy và thời gian từng epoch.
- `M1-0x_best.pt`: checkpoint có validation loss tốt nhất.
- `M1-0x_result.json`: cấu hình và toàn bộ metric test.

`test_loss` vẫn được tính bình thường vì test chỉ forward trong `model.eval()` và
`torch.no_grad()`: không có `backward()` hay `optimizer.step()`. Test loss không được
dùng để chọn epoch hoặc chỉnh hyperparameter.
