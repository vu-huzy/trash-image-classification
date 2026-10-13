# Kết quả thực nghiệm

Cập nhật lần cuối: 2026-10-13. Số liệu lấy từ output của các notebook, gồm hai đợt chạy:

- **2026-10-13** (8 notebook: `m1_01`–`m1_04`, `cnn_sequential`, `cnn_parallel`, `vgg_sequential`, `vgg_parallel`): chạy lại với code ở commit `cdee048` (thêm HFlip + ColorJitter + chuẩn hóa ImageNet cho Model 1, bỏ Rotation ở Model 2, chọn epoch theo validation accuracy, thêm early stopping; patience Model 1 là 15, Model 2 là 7). Kernel `python3` trỏ vào env `anaconda3\envs\gpu` (PyTorch 2.11 + CUDA 12.8), GPU NVIDIA RTX 5070, DataLoader 8 worker.
- **2026-10-11** (6 notebook: `resnet` của Model 2 và 5 notebook Model 3): giữ nguyên kết quả cũ (PyTorch 2.14 + CUDA 13). `resnet` chưa có kết quả mới vì lần chạy lại bị dừng (quá chậm do thiếu VRAM), notebook hiện đã đặt batch 24 và patience 7 nhưng chưa chạy lại.

Mỗi notebook chỉ chạy một lần. Chỉ model 3 và `cnn_sequential` đặt seed; các notebook còn lại không đặt seed, nên chạy lại có thể lệch vài điểm phần trăm. Vì hai đợt chạy dùng cấu hình khác nhau, `resnet` và Model 3 chưa so sánh công bằng được với 8 notebook còn lại.

## Dữ liệu và cách đánh giá

- Dữ liệu: `data/VN_trash_classification_preprocessing`, 9 lớp (Alu, Carton, Foam_box, Milk_box, Other, PET, Paper, Paper_cup, Plastic_cup).
- Chia tập: train 9.542 ảnh, validation 1.685 ảnh, test 864 ảnh.
- Chọn epoch tốt nhất bằng tập validation; tập test chỉ dùng một lần sau khi nạp lại trọng số tốt nhất.
- Chỉ số test: accuracy, macro precision / recall / F1, weighted F1, và bảng precision / recall / F1 / support cho từng lớp (không dùng confusion matrix).

## Bảng tổng hợp

| Model | Test acc | Macro P | Macro R | Macro F1 | Weighted F1 | Epoch chạy | Best epoch | Thời gian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model 1 · M1-01 — MLP | 19.79% | 0.1961 | 0.1805 | 0.1263 | 0.1376 | 22 | 7 | 2.2 phút |
| Model 1 · M1-02 — CNN 2 block | 52.31% | 0.5262 | 0.5237 | 0.5080 | 0.5144 | 50 | 49 | 6.9 phút |
| Model 1 · M1-03 — CNN 3 block | 63.08% | 0.6350 | 0.6272 | 0.6253 | 0.6346 | 50 | 45 | 7.9 phút |
| Model 1 · M1-04 — CNN 3 block + GAP | 48.84% | 0.4926 | 0.4983 | 0.4771 | 0.4827 | 50 | 44 | 8.1 phút |
| Model 2 · CNN sequential | 65.97% | 0.6607 | 0.6610 | 0.6526 | 0.6602 | 30 | 30 | 16.1 phút |
| Model 2 · CNN parallel | 68.17% | 0.6782 | 0.6801 | 0.6754 | 0.6841 | 30 | 30 | 17.8 phút |
| Model 2 · VGG sequential | 72.34% | 0.7202 | 0.7175 | 0.7109 | 0.7242 | 30 | 25 | 29.1 phút |
| Model 2 · VGG parallel | 76.04% | 0.7570 | 0.7569 | 0.7477 | 0.7589 | 30 | 28 | 37.4 phút |
| Model 2 · ResNet (tự viết, không phải ResNet-18 chuẩn) | 69.33% | 0.6861 | 0.6840 | 0.6813 | 0.6927 | 30 | 30 | 90.7 phút |
| Model 3 · 3A — VGG16 đóng băng | 82.18% | 0.8192 | 0.8225 | 0.8162 | 0.8229 | 19 | 13 | 7.6 phút |
| Model 3 · 3B — ResNet50 đóng băng | 93.98% | 0.9376 | 0.9366 | 0.9363 | 0.9397 | 12 | 6 | 3.7 phút |
| Model 3 · 3B — EfficientNet-B0 đóng băng | 93.06% | 0.9249 | 0.9272 | 0.9245 | 0.9311 | 18 | 12 | 3.2 phút |
| Model 3 · 3C — ResNet50 + LoRA | 96.53% | 0.9635 | 0.9658 | 0.9643 | 0.9653 | 30 | 27 | 10.7 phút |
| Model 3 · 3D — ResNet50 fine-tune toàn bộ | 92.82% | 0.9205 | 0.9311 | 0.9234 | 0.9286 | 10 | 5 | 6.1 phút |

## F1 theo từng lớp

| Lớp | m1_01 | m1_02 | m1_03 | m1_04 | m2_cnn_sequential | m2_cnn_parallel | m2_vgg_sequential | m2_vgg_parallel | m2_resnet | m3_3a_frozen_vgg16 | m3_3b_frozen_resnet50 | m3_3b_frozen_efficientnet_b0 | m3_3c_lora_resnet50 | m3_3d_full_finetune_resnet50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alu | 0.0385 | 0.5419 | 0.7191 | 0.6024 | 0.7264 | 0.7097 | 0.7960 | 0.8254 | 0.7716 | 0.8835 | 0.9648 | 0.9447 | 0.9744 | 0.9548 |
| Carton | 0.4537 | 0.7204 | 0.7739 | 0.7685 | 0.7767 | 0.8116 | 0.8390 | 0.8879 | 0.8365 | 0.8265 | 0.9469 | 0.9533 | 0.9668 | 0.9327 |
| Foam_box | 0.0152 | 0.3590 | 0.4780 | 0.3288 | 0.5027 | 0.5000 | 0.5571 | 0.5839 | 0.5342 | 0.8686 | 0.9146 | 0.9012 | 0.9425 | 0.8914 |
| Milk_box | 0.0708 | 0.7027 | 0.7600 | 0.6699 | 0.8200 | 0.8438 | 0.8687 | 0.8627 | 0.8804 | 0.9128 | 0.9806 | 0.9608 | 0.9901 | 0.9756 |
| Other | 0.0220 | 0.4462 | 0.5167 | 0.4260 | 0.5760 | 0.5950 | 0.5426 | 0.6202 | 0.5405 | 0.6838 | 0.8824 | 0.8257 | 0.9515 | 0.8448 |
| PET | 0.0357 | 0.4925 | 0.5772 | 0.3310 | 0.5093 | 0.5532 | 0.6083 | 0.6341 | 0.5182 | 0.7163 | 0.9412 | 0.9246 | 0.9648 | 0.9091 |
| Paper | 0.0000 | 0.4964 | 0.6857 | 0.5909 | 0.6798 | 0.7075 | 0.7282 | 0.8019 | 0.6786 | 0.7873 | 0.9390 | 0.9519 | 0.9585 | 0.9159 |
| Paper_cup | 0.1808 | 0.5646 | 0.6602 | 0.5761 | 0.7014 | 0.7407 | 0.8326 | 0.8739 | 0.8325 | 0.9116 | 0.9488 | 0.9573 | 0.9859 | 0.9859 |
| Plastic_cup | 0.3200 | 0.2483 | 0.4571 | 0.0000 | 0.5812 | 0.6174 | 0.6256 | 0.6389 | 0.5392 | 0.7553 | 0.9083 | 0.9009 | 0.9439 | 0.9000 |

## Ghi chú khi đọc kết quả

- Số liệu Model 1 cũ trong README (M1-01 18.06%, M1-02 56.37%, M1-03 60.30%, M1-04 52.66%) đến từ các script
  `.py` ở commit `44c0af8`, nay đã bị xóa. Từ commit `cdee048` các notebook Model 1 đã có cùng augmentation
  (HFlip + ColorJitter) và chuẩn hóa ImageNet như script cũ, nên bộ số liệu mới (M1-01 19.79%, M1-02 52.31%,
  M1-03 63.08%, M1-04 48.84%) gần với số script cũ hơn bản notebook trước đó (15.62% / 45.95% / 48.73% / 45.95%,
  chỉ Resize + ToTensor). Một lần chạy mỗi cấu hình và ba thay đổi cùng lúc (augmentation, chuẩn hóa, tiêu chí chọn
  epoch) nên chưa tách được từng yếu tố.
- M1-01 (MLP) vẫn học rất kém: ở epoch tốt nhất (7) accuracy train 20.86%, validation 23.50%, test 19.79% (đoán ngẫu nhiên là 11.1%); chỉ lớp Paper có F1 bằng 0 nhưng 6 trong 9 lớp có F1 dưới 0.1. Notebook dừng sớm ở epoch 22 do validation accuracy không tăng thêm 15 epoch liên tiếp. Các lần chạy khác của cùng notebook từng cho 5.67% và khoảng 18–20%, nghĩa là luôn ở mức thấp và dao động mạnh giữa các lần chạy.
- Model 1: M1-02/03/04 chạy đủ 50 epoch với best epoch ở 49, 45 và 44, tức vẫn đang cải thiện khi chạm giới hạn; early stopping (patience 15) không kích hoạt ở ba notebook này.
- Model 2: 4 notebook chạy lại ngày 2026-10-13 có best epoch ở 30, 30, 25 và 28 trên 30 epoch, early stopping (patience 7) không kích hoạt ở notebook nào, nên chưa model nào ngừng cải thiện trước giới hạn epoch (chưa thử tăng epoch). Số so sánh với checkpoint cũ chọn từ các lần chạy `train.py` chưa được tính lại cho bộ số liệu mới.
- Model 3: ba biến thể chỉ train head (VGG16, ResNet50, EfficientNet-B0) đều dừng sớm và có chênh lệch lớn giữa train và validation (train accuracy 98.7–99.6% so với val 86.5–93.2% ở epoch tốt nhất). LoRA (3C) cao hơn fine-tune toàn bộ (3D) 3.7 điểm accuracy, ngược với framework cũ (3D 0.9606 > 3C 0.9549) vốn dùng cấu hình khác bản notebook.
- Các notebook model 3 lưu toàn bộ `state_dict` (cả backbone đóng băng), nên `3a_frozen_vgg16.pt` nặng khoảng 520 MB.
- Mọi so sánh giữa các model chỉ dựa trên một lần chạy, một lần chia tập.

## Model 1

### M1-01 — MLP

- Notebook: `src/model1/m1_01.ipynb`
- Cấu hình: ảnh 128×128; Flatten → Linear 512 → Linear 128 → 9 (Dropout 0.2). train: Resize + HFlip + ColorJitter 0.1 + chuẩn hóa ImageNet, val/test: Resize + chuẩn hóa ImageNet, batch 64, Adam lr=1e-3, `CrossEntropyLoss`, tối đa 50 epoch, early stopping theo validation accuracy (patience 15), nạp lại trọng số có val accuracy cao nhất trước khi test
- Huấn luyện: chạy 22/50 epoch (early stopping ở epoch 22), mất 2.2 phút
- Epoch tốt nhất (7): Train Loss 2.0702, Train Acc 20.86%, Val Loss 2.0781, Val Acc 23.50%
- Epoch cuối (22): Train Loss 1.9449, Train Acc 25.70%, Val Loss 2.0605, Val Acc 19.82%
- Test (864 ảnh): accuracy 19.79%, macro P 0.1961, macro R 0.1805, macro F1 0.1263, weighted F1 0.1376
- Model đã lưu: `src/model1/checkpoints/m1_01.pt` (96.3 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.4000 | 0.0202 | 0.0385 | 99 |
| Carton | 0.4537 | 0.4537 | 0.4537 | 108 |
| Foam_box | 0.0222 | 0.0115 | 0.0152 | 87 |
| Milk_box | 0.3333 | 0.0396 | 0.0708 | 101 |
| Other | 0.0244 | 0.0200 | 0.0220 | 50 |
| PET | 0.2000 | 0.0196 | 0.0357 | 102 |
| Paper | 0.0000 | 0.0000 | 0.0000 | 106 |
| Paper_cup | 0.1285 | 0.3048 | 0.1808 | 105 |
| Plastic_cup | 0.2030 | 0.7547 | 0.3200 | 106 |
| *macro avg* | 0.1961 | 0.1805 | 0.1263 | 864 |
| *weighted avg* | 0.2093 | 0.1979 | 0.1376 | 864 |

Lớp tốt nhất: Carton (F1 0.4537). Lớp kém nhất: Paper (F1 0.0000).

### M1-02 — CNN 2 block

- Notebook: `src/model1/m1_02.ipynb`
- Cấu hình: ảnh 224×224; 2 × (Conv + BatchNorm + ReLU + MaxPool: 32, 64) → AdaptiveAvgPool 4×4 → Linear 128 → 9 (Dropout 0.2). train: Resize + HFlip + ColorJitter 0.1 + chuẩn hóa ImageNet, val/test: Resize + chuẩn hóa ImageNet, batch 64, Adam lr=1e-3, `CrossEntropyLoss`, tối đa 50 epoch, early stopping theo validation accuracy (patience 15), nạp lại trọng số có val accuracy cao nhất trước khi test
- Huấn luyện: chạy 50/50 epoch, mất 6.9 phút
- Epoch tốt nhất (49): Train Loss 0.9841, Train Acc 66.67%, Val Loss 1.1380, Val Acc 62.31%
- Epoch cuối (50): Train Loss 0.9697, Train Acc 66.24%, Val Loss 1.2064, Val Acc 60.59%
- Test (864 ảnh): accuracy 52.31%, macro P 0.5262, macro R 0.5237, macro F1 0.5080, weighted F1 0.5144
- Model đã lưu: `src/model1/checkpoints/m1_02.pt` (0.6 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.5288 | 0.5556 | 0.5419 | 99 |
| Carton | 0.8590 | 0.6204 | 0.7204 | 108 |
| Foam_box | 0.4058 | 0.3218 | 0.3590 | 87 |
| Milk_box | 0.6446 | 0.7723 | 0.7027 | 101 |
| Other | 0.3625 | 0.5800 | 0.4462 | 50 |
| PET | 0.5052 | 0.4804 | 0.4925 | 102 |
| Paper | 0.4012 | 0.6509 | 0.4964 | 106 |
| Paper_cup | 0.5673 | 0.5619 | 0.5646 | 105 |
| Plastic_cup | 0.4615 | 0.1698 | 0.2483 | 106 |
| *macro avg* | 0.5262 | 0.5237 | 0.5080 | 864 |
| *weighted avg* | 0.5396 | 0.5231 | 0.5144 | 864 |

Lớp tốt nhất: Carton (F1 0.7204). Lớp kém nhất: Plastic_cup (F1 0.2483).

### M1-03 — CNN 3 block

- Notebook: `src/model1/m1_03.ipynb`
- Cấu hình: ảnh 224×224; 3 × (Conv + BatchNorm + ReLU + MaxPool: 32, 64, 128) → AdaptiveAvgPool 4×4 → Linear 128 → 9 (Dropout 0.2). train: Resize + HFlip + ColorJitter 0.1 + chuẩn hóa ImageNet, val/test: Resize + chuẩn hóa ImageNet, batch 64, Adam lr=1e-3, `CrossEntropyLoss`, tối đa 50 epoch, early stopping theo validation accuracy (patience 15), nạp lại trọng số có val accuracy cao nhất trước khi test
- Huấn luyện: chạy 50/50 epoch, mất 7.9 phút
- Epoch tốt nhất (45): Train Loss 0.7144, Train Acc 75.11%, Val Loss 0.9700, Val Acc 69.14%
- Epoch cuối (50): Train Loss 0.6846, Train Acc 75.60%, Val Loss 1.0685, Val Acc 65.93%
- Test (864 ảnh): accuracy 63.08%, macro P 0.6350, macro R 0.6272, macro F1 0.6253, weighted F1 0.6346
- Model đã lưu: `src/model1/checkpoints/m1_03.pt` (1.4 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.8101 | 0.6465 | 0.7191 | 99 |
| Carton | 0.8462 | 0.7130 | 0.7739 | 108 |
| Foam_box | 0.5278 | 0.4368 | 0.4780 | 87 |
| Milk_box | 0.7677 | 0.7525 | 0.7600 | 101 |
| Other | 0.4429 | 0.6200 | 0.5167 | 50 |
| PET | 0.4931 | 0.6961 | 0.5772 | 102 |
| Paper | 0.6923 | 0.6792 | 0.6857 | 106 |
| Paper_cup | 0.6733 | 0.6476 | 0.6602 | 105 |
| Plastic_cup | 0.4615 | 0.4528 | 0.4571 | 106 |
| *macro avg* | 0.6350 | 0.6272 | 0.6253 | 864 |
| *weighted avg* | 0.6487 | 0.6308 | 0.6346 | 864 |

Lớp tốt nhất: Carton (F1 0.7739). Lớp kém nhất: Plastic_cup (F1 0.4571).

### M1-04 — CNN 3 block + GAP

- Notebook: `src/model1/m1_04.ipynb`
- Cấu hình: ảnh 224×224; 3 × (Conv + BatchNorm + ReLU + MaxPool: 32, 64, 128) → Global Average Pooling → Linear 9. train: Resize + HFlip + ColorJitter 0.1 + chuẩn hóa ImageNet, val/test: Resize + chuẩn hóa ImageNet, batch 64, Adam lr=1e-3, `CrossEntropyLoss`, tối đa 50 epoch, early stopping theo validation accuracy (patience 15), nạp lại trọng số có val accuracy cao nhất trước khi test
- Huấn luyện: chạy 50/50 epoch, mất 8.1 phút
- Epoch tốt nhất (44): Train Loss 1.1526, Train Acc 61.61%, Val Loss 1.2513, Val Acc 60.36%
- Epoch cuối (50): Train Loss 1.1192, Train Acc 63.15%, Val Loss 1.2766, Val Acc 58.87%
- Test (864 ảnh): accuracy 48.84%, macro P 0.4926, macro R 0.4983, macro F1 0.4771, weighted F1 0.4827
- Model đã lưu: `src/model1/checkpoints/m1_04.pt` (0.4 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.7463 | 0.5051 | 0.6024 | 99 |
| Carton | 0.8211 | 0.7222 | 0.7685 | 108 |
| Foam_box | 0.4068 | 0.2759 | 0.3288 | 87 |
| Milk_box | 0.6571 | 0.6832 | 0.6699 | 101 |
| Other | 0.3025 | 0.7200 | 0.4260 | 50 |
| PET | 0.2582 | 0.4608 | 0.3310 | 102 |
| Paper | 0.5702 | 0.6132 | 0.5909 | 106 |
| Paper_cup | 0.6709 | 0.5048 | 0.5761 | 105 |
| Plastic_cup | 0.0000 | 0.0000 | 0.0000 | 106 |
| *macro avg* | 0.4926 | 0.4983 | 0.4771 | 864 |
| *weighted avg* | 0.5054 | 0.4884 | 0.4827 | 864 |

Lớp tốt nhất: Carton (F1 0.7685). Lớp kém nhất: Plastic_cup (F1 0.0000).

## Model 2

### CNN sequential

- Notebook: `src/model2/notebook/cnn_sequential.ipynb`
- Cấu hình: 3 stage × 2 Conv 3×3 (48, 96, 192) + BatchNorm, MaxPool mỗi stage → GAP → Linear 256 → 9 (Dropout 0.3). Ảnh 224×224, train: HFlip + ColorJitter 0.1, chuẩn hóa ImageNet; batch 32, AdamW lr=1e-4 weight_decay=5e-4, cosine annealing, `CrossEntropyLoss` có trọng số lớp (cân bằng lớp), tối đa 30 epoch, early stopping theo validation accuracy (patience 7), nạp lại trọng số có validation accuracy cao nhất trước khi test
- Tham số: 697.849
- Huấn luyện: chạy 30/30 epoch, mất 16.1 phút
- Epoch tốt nhất (30): Train Loss 0.8944, Train Acc 71.75%, Val Loss 0.9804, Val Acc 69.38%
- Epoch cuối (30): Train Loss 0.8944, Train Acc 71.75%, Val Loss 0.9804, Val Acc 69.38%
- Test (864 ảnh): accuracy 65.97%, macro P 0.6607, macro R 0.6610, macro F1 0.6526, weighted F1 0.6602
- Model đã lưu: `src/model2/checkpoints/cnn_sequential.pt` (2.7 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.7157 | 0.7374 | 0.7264 | 99 |
| Carton | 0.8163 | 0.7407 | 0.7767 | 108 |
| Foam_box | 0.4700 | 0.5402 | 0.5027 | 87 |
| Milk_box | 0.8283 | 0.8119 | 0.8200 | 101 |
| Other | 0.4800 | 0.7200 | 0.5760 | 50 |
| PET | 0.6949 | 0.4020 | 0.5093 | 102 |
| Paper | 0.7113 | 0.6509 | 0.6798 | 106 |
| Paper_cup | 0.6981 | 0.7048 | 0.7014 | 105 |
| Plastic_cup | 0.5312 | 0.6415 | 0.5812 | 106 |
| *macro avg* | 0.6607 | 0.6610 | 0.6526 | 864 |
| *weighted avg* | 0.6753 | 0.6597 | 0.6602 | 864 |

Lớp tốt nhất: Milk_box (F1 0.8200). Lớp kém nhất: Foam_box (F1 0.5027).

### CNN parallel

- Notebook: `src/model2/notebook/cnn_parallel.ipynb`
- Cấu hình: 3 nhánh Conv song song (1 conv, 2 conv, conv dilation 2; 32 kênh mỗi nhánh) nối thành 96 kênh → 3 khối residual stride 2 (64, 128, 192) → GAP → Linear 256 → 9 (Dropout 0.4). Ảnh 224×224, train: HFlip + ColorJitter 0.1, chuẩn hóa ImageNet; batch 32, AdamW lr=1e-4 weight_decay=5e-4, cosine annealing, `CrossEntropyLoss` có trọng số lớp (cân bằng lớp), tối đa 30 epoch, early stopping theo validation accuracy (patience 7), nạp lại trọng số có validation accuracy cao nhất trước khi test
- Huấn luyện: chạy 30/30 epoch, mất 17.8 phút
- Epoch tốt nhất (30): Train Loss 0.8483, Train Acc 72.90%, Val Loss 0.9281, Val Acc 70.74%
- Epoch cuối (30): Train Loss 0.8483, Train Acc 72.90%, Val Loss 0.9281, Val Acc 70.74%
- Test (864 ảnh): accuracy 68.17%, macro P 0.6782, macro R 0.6801, macro F1 0.6754, weighted F1 0.6841
- Model đã lưu: `src/model2/checkpoints/cnn_parallel.pt` (3.7 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.7586 | 0.6667 | 0.7097 | 99 |
| Carton | 0.8485 | 0.7778 | 0.8116 | 108 |
| Foam_box | 0.4944 | 0.5057 | 0.5000 | 87 |
| Milk_box | 0.8901 | 0.8020 | 0.8438 | 101 |
| Other | 0.5070 | 0.7200 | 0.5950 | 50 |
| PET | 0.6047 | 0.5098 | 0.5532 | 102 |
| Paper | 0.7075 | 0.7075 | 0.7075 | 106 |
| Paper_cup | 0.7207 | 0.7619 | 0.7407 | 105 |
| Plastic_cup | 0.5726 | 0.6698 | 0.6174 | 106 |
| *macro avg* | 0.6782 | 0.6801 | 0.6754 | 864 |
| *weighted avg* | 0.6922 | 0.6817 | 0.6841 | 864 |

Lớp tốt nhất: Milk_box (F1 0.8438). Lớp kém nhất: Foam_box (F1 0.5000).

### VGG sequential

- Notebook: `src/model2/notebook/vgg_sequential.ipynb`
- Cấu hình: kiểu VGG-16 có BatchNorm (13 conv: 64, 128, 256, 512, 512) → GAP → Linear 128 → 9 (Dropout 0.6). Ảnh 224×224, train: HFlip + ColorJitter 0.1, chuẩn hóa ImageNet; batch 32, AdamW lr=1e-4 weight_decay=5e-4, cosine annealing, `CrossEntropyLoss` có trọng số lớp (cân bằng lớp), tối đa 30 epoch, early stopping theo validation accuracy (patience 7), nạp lại trọng số có validation accuracy cao nhất trước khi test
- Huấn luyện: chạy 30/30 epoch, mất 29.1 phút
- Epoch tốt nhất (25): Train Loss 0.5318, Train Acc 83.29%, Val Loss 0.6664, Val Acc 79.70%
- Epoch cuối (30): Train Loss 0.4763, Train Acc 85.40%, Val Loss 0.6561, Val Acc 79.47%
- Test (864 ảnh): accuracy 72.34%, macro P 0.7202, macro R 0.7175, macro F1 0.7109, weighted F1 0.7242
- Model đã lưu: `src/model2/checkpoints/vgg_sequential.pt` (56.5 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.7843 | 0.8081 | 0.7960 | 99 |
| Carton | 0.8866 | 0.7963 | 0.8390 | 108 |
| Foam_box | 0.7358 | 0.4483 | 0.5571 | 87 |
| Milk_box | 0.8866 | 0.8515 | 0.8687 | 101 |
| Other | 0.4430 | 0.7000 | 0.5426 | 50 |
| PET | 0.5739 | 0.6471 | 0.6083 | 102 |
| Paper | 0.7500 | 0.7075 | 0.7282 | 106 |
| Paper_cup | 0.7931 | 0.8762 | 0.8326 | 105 |
| Plastic_cup | 0.6286 | 0.6226 | 0.6256 | 106 |
| *macro avg* | 0.7202 | 0.7175 | 0.7109 | 864 |
| *weighted avg* | 0.7373 | 0.7234 | 0.7242 | 864 |

Lớp tốt nhất: Milk_box (F1 0.8687). Lớp kém nhất: Other (F1 0.5426).

### VGG parallel

- Notebook: `src/model2/notebook/vgg_parallel.ipynb`
- Cấu hình: 3 nhánh đầu vào (3×3, 3×3 dilation 2, 5×5; 64 kênh mỗi nhánh) nối thành 192 kênh → 5 stage kiểu VGG → GAP → Linear 128 → 9 (Dropout 0.6). Ảnh 224×224, train: HFlip + ColorJitter 0.1, chuẩn hóa ImageNet; batch 32, AdamW lr=1e-4 weight_decay=5e-4, cosine annealing, `CrossEntropyLoss` có trọng số lớp (cân bằng lớp), tối đa 30 epoch, early stopping theo validation accuracy (patience 7), nạp lại trọng số có validation accuracy cao nhất trước khi test
- Huấn luyện: chạy 30/30 epoch, mất 37.4 phút
- Epoch tốt nhất (28): Train Loss 0.3572, Train Acc 89.24%, Val Loss 0.5527, Val Acc 82.55%
- Epoch cuối (30): Train Loss 0.3375, Train Acc 89.53%, Val Loss 0.5584, Val Acc 82.55%
- Test (864 ảnh): accuracy 76.04%, macro P 0.7570, macro R 0.7569, macro F1 0.7477, weighted F1 0.7589
- Model đã lưu: `src/model2/checkpoints/vgg_parallel.pt` (20.6 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.8667 | 0.7879 | 0.8254 | 99 |
| Carton | 0.8962 | 0.8796 | 0.8879 | 108 |
| Foam_box | 0.8000 | 0.4598 | 0.5839 | 87 |
| Milk_box | 0.8544 | 0.8713 | 0.8627 | 101 |
| Other | 0.5063 | 0.8000 | 0.6202 | 50 |
| PET | 0.6311 | 0.6373 | 0.6341 | 102 |
| Paper | 0.8019 | 0.8019 | 0.8019 | 106 |
| Paper_cup | 0.8291 | 0.9238 | 0.8739 | 105 |
| Plastic_cup | 0.6273 | 0.6509 | 0.6389 | 106 |
| *macro avg* | 0.7570 | 0.7569 | 0.7477 | 864 |
| *weighted avg* | 0.7717 | 0.7604 | 0.7589 | 864 |

Lớp tốt nhất: Carton (F1 0.8879). Lớp kém nhất: Foam_box (F1 0.5839).

### ResNet (tự viết, không phải ResNet-18 chuẩn)

- Lưu ý: đây là kết quả **cũ** (chạy 2026-10-11, batch 32, chưa có early stopping). Notebook hiện đã đổi `BATCH_SIZE` 32 → 24 và `PATIENCE = 7` nhưng chưa chạy lại, và `src/model2/checkpoints/resnet.pt` vẫn là file của lần chạy cũ.
- Notebook: `src/model2/notebook/resnet.ipynb`
- Cấu hình: stem Conv 3×3 stride 1, 4 stage × 2 BasicBlock (64, 128, 256, 512) → GAP → Dropout 0.5 → Linear 128 → Dropout 0.5 → 9; thêm label smoothing 0.05. Ảnh 224×224, train: HFlip + Rotation 10° + ColorJitter 0.1, chuẩn hóa ImageNet; batch 32, AdamW lr=1e-4 weight_decay=5e-4, cosine annealing, `CrossEntropyLoss` có trọng số lớp (cân bằng lớp), chạy đủ 30 epoch (không early stopping), giữ trọng số có validation accuracy cao nhất
- Huấn luyện: chạy 30/30 epoch, mất 90.7 phút
- Epoch tốt nhất (30 — suy ra từ log: epoch có Val Acc cao nhất, notebook không in): Train Acc 72.85%, Val Acc 75.07%
- Epoch cuối (30): Train Acc 72.85%, Val Acc 75.07%
- Test (864 ảnh): accuracy 69.33%, macro P 0.6861, macro R 0.6840, macro F1 0.6813, weighted F1 0.6927
- Model đã lưu: `src/model2/checkpoints/resnet.pt` (42.9 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.7755 | 0.7677 | 0.7716 | 99 |
| Carton | 0.8700 | 0.8056 | 0.8365 | 108 |
| Foam_box | 0.6610 | 0.4483 | 0.5342 | 87 |
| Milk_box | 0.8519 | 0.9109 | 0.8804 | 101 |
| Other | 0.4918 | 0.6000 | 0.5405 | 50 |
| PET | 0.4831 | 0.5588 | 0.5182 | 102 |
| Paper | 0.6441 | 0.7170 | 0.6786 | 106 |
| Paper_cup | 0.8365 | 0.8286 | 0.8325 | 105 |
| Plastic_cup | 0.5612 | 0.5189 | 0.5392 | 106 |
| *macro avg* | 0.6861 | 0.6840 | 0.6813 | 864 |
| *weighted avg* | 0.6988 | 0.6933 | 0.6927 | 864 |

Lớp tốt nhất: Milk_box (F1 0.8804). Lớp kém nhất: PET (F1 0.5182).

## Model 3

### 3A — VGG16 đóng băng

- Notebook: `src/model3/notebook/3a_frozen_vgg16.ipynb`
- Cấu hình: VGG16 (IMAGENET1K_V1, đặc trưng 4096) đóng băng, chỉ train head; batch 64, head lr 1e-3, weight_decay 1e-4, tối đa 30 epoch, patience 6. Ảnh 224×224, train: HFlip + ColorJitter 0.1, chuẩn hóa ImageNet; backbone ImageNet + head MLP (Linear→512, BatchNorm, ReLU, Dropout 0.3, Linear→9); AdamW, cosine annealing, seed 42, early stopping theo validation accuracy; giữ trọng số tốt nhất trước khi test
- Tham số: 2,103,305 huấn luyện / 136,363,849 tổng
- Huấn luyện: chạy 19/30 epoch, dừng sớm (early stopping), mất 7.6 phút
- Epoch tốt nhất (13): Train Loss 0.0455, Train Acc 98.74%, Val Loss 0.6206, Val Acc 86.47%
- Epoch cuối (19): Train Loss 0.0208, Train Acc 99.53%, Val Loss 0.6794, Val Acc 86.05%
- Test (864 ảnh): accuracy 82.18%, macro P 0.8192, macro R 0.8225, macro F1 0.8162, weighted F1 0.8229
- Model đã lưu: `src/model3/checkpoints/3a_frozen_vgg16.pt` (520.2 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.8505 | 0.9192 | 0.8835 | 99 |
| Carton | 0.9205 | 0.7500 | 0.8265 | 108 |
| Foam_box | 0.8636 | 0.8736 | 0.8686 | 87 |
| Milk_box | 0.9468 | 0.8812 | 0.9128 | 101 |
| Other | 0.5970 | 0.8000 | 0.6838 | 50 |
| PET | 0.6814 | 0.7549 | 0.7163 | 102 |
| Paper | 0.7565 | 0.8208 | 0.7873 | 106 |
| Paper_cup | 0.8909 | 0.9333 | 0.9116 | 105 |
| Plastic_cup | 0.8659 | 0.6698 | 0.7553 | 106 |
| *macro avg* | 0.8192 | 0.8225 | 0.8162 | 864 |
| *weighted avg* | 0.8325 | 0.8218 | 0.8229 | 864 |

Lớp tốt nhất: Milk_box (F1 0.9128). Lớp kém nhất: Other (F1 0.6838).

### 3B — ResNet50 đóng băng

- Notebook: `src/model3/notebook/3b_frozen_resnet50.ipynb`
- Cấu hình: ResNet50 (IMAGENET1K_V2, đặc trưng 2048) đóng băng, chỉ train head; batch 64, head lr 1e-3, weight_decay 1e-4, tối đa 30 epoch, patience 6. Ảnh 224×224, train: HFlip + ColorJitter 0.1, chuẩn hóa ImageNet; backbone ImageNet + head MLP (Linear→512, BatchNorm, ReLU, Dropout 0.3, Linear→9); AdamW, cosine annealing, seed 42, early stopping theo validation accuracy; giữ trọng số tốt nhất trước khi test
- Tham số: 1,054,729 huấn luyện / 24,562,761 tổng
- Huấn luyện: chạy 12/30 epoch, dừng sớm (early stopping), mất 3.7 phút
- Epoch tốt nhất (6): Train Loss 0.0382, Train Acc 98.94%, Val Loss 0.2604, Val Acc 93.23%
- Epoch cuối (12): Train Loss 0.0114, Train Acc 99.85%, Val Loss 0.2982, Val Acc 93.00%
- Test (864 ảnh): accuracy 93.98%, macro P 0.9376, macro R 0.9366, macro F1 0.9363, weighted F1 0.9397
- Model đã lưu: `src/model3/checkpoints/3b_frozen_resnet50.pt` (94.0 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.9600 | 0.9697 | 0.9648 | 99 |
| Carton | 0.9899 | 0.9074 | 0.9469 | 108 |
| Foam_box | 0.9740 | 0.8621 | 0.9146 | 87 |
| Milk_box | 0.9619 | 1.0000 | 0.9806 | 101 |
| Other | 0.8654 | 0.9000 | 0.8824 | 50 |
| PET | 0.9412 | 0.9412 | 0.9412 | 102 |
| Paper | 0.9346 | 0.9434 | 0.9390 | 106 |
| Paper_cup | 0.9273 | 0.9714 | 0.9488 | 105 |
| Plastic_cup | 0.8839 | 0.9340 | 0.9083 | 106 |
| *macro avg* | 0.9376 | 0.9366 | 0.9363 | 864 |
| *weighted avg* | 0.9412 | 0.9398 | 0.9397 | 864 |

Lớp tốt nhất: Milk_box (F1 0.9806). Lớp kém nhất: Other (F1 0.8824).

### 3B — EfficientNet-B0 đóng băng

- Notebook: `src/model3/notebook/3b_frozen_efficientnet_b0.ipynb`
- Cấu hình: EfficientNet-B0 (IMAGENET1K_V1, đặc trưng 1280) đóng băng, chỉ train head; batch 64, head lr 1e-3, weight_decay 1e-4, tối đa 30 epoch, patience 6. Ảnh 224×224, train: HFlip + ColorJitter 0.1, chuẩn hóa ImageNet; backbone ImageNet + head MLP (Linear→512, BatchNorm, ReLU, Dropout 0.3, Linear→9); AdamW, cosine annealing, seed 42, early stopping theo validation accuracy; giữ trọng số tốt nhất trước khi test
- Tham số: 661,513 huấn luyện / 4,669,061 tổng
- Huấn luyện: chạy 18/30 epoch, dừng sớm (early stopping), mất 3.2 phút
- Epoch tốt nhất (12): Train Loss 0.0182, Train Acc 99.64%, Val Loss 0.3872, Val Acc 91.51%
- Epoch cuối (18): Train Loss 0.0072, Train Acc 99.83%, Val Loss 0.4189, Val Acc 91.22%
- Test (864 ảnh): accuracy 93.06%, macro P 0.9249, macro R 0.9272, macro F1 0.9245, weighted F1 0.9311
- Model đã lưu: `src/model3/checkpoints/3b_frozen_efficientnet_b0.pt` (18.1 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.9400 | 0.9495 | 0.9447 | 99 |
| Carton | 0.9623 | 0.9444 | 0.9533 | 108 |
| Foam_box | 0.9733 | 0.8391 | 0.9012 | 87 |
| Milk_box | 0.9515 | 0.9703 | 0.9608 | 101 |
| Other | 0.7627 | 0.9000 | 0.8257 | 50 |
| PET | 0.9485 | 0.9020 | 0.9246 | 102 |
| Paper | 0.9706 | 0.9340 | 0.9519 | 106 |
| Paper_cup | 0.9528 | 0.9619 | 0.9573 | 105 |
| Plastic_cup | 0.8621 | 0.9434 | 0.9009 | 106 |
| *macro avg* | 0.9249 | 0.9272 | 0.9245 | 864 |
| *weighted avg* | 0.9340 | 0.9306 | 0.9311 | 864 |

Lớp tốt nhất: Milk_box (F1 0.9608). Lớp kém nhất: Other (F1 0.8257).

### 3C — ResNet50 + LoRA

- Notebook: `src/model3/notebook/3c_lora_resnet50.ipynb`
- Cấu hình: ResNet50 đóng băng, LoRA (rank 8, alpha 16) trên mọi Conv2d của layer3 và layer4 + head; batch 32, lr 1e-3 cho head và LoRA, weight_decay 1e-4, tối đa 30 epoch, patience 6. Ảnh 224×224, train: HFlip + ColorJitter 0.1, chuẩn hóa ImageNet; backbone ImageNet + head MLP (Linear→512, BatchNorm, ReLU, Dropout 0.3, Linear→9); AdamW, cosine annealing, seed 42, early stopping theo validation accuracy; giữ trọng số tốt nhất trước khi test
- Tham số: 1,570,825 huấn luyện / 25,078,857 tổng
- Huấn luyện: chạy 30/30 epoch, mất 10.7 phút
- Epoch tốt nhất (27): Train Loss 0.0008, Train Acc 99.97%, Val Loss 0.2378, Val Acc 95.43%
- Epoch cuối (30): Train Loss 0.0004, Train Acc 100.00%, Val Loss 0.2380, Val Acc 95.25%
- Test (864 ảnh): accuracy 96.53%, macro P 0.9635, macro R 0.9658, macro F1 0.9643, weighted F1 0.9653
- Model đã lưu: `src/model3/checkpoints/3c_lora_resnet50.pt` (96.0 MB)

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

Lớp tốt nhất: Milk_box (F1 0.9901). Lớp kém nhất: Foam_box (F1 0.9425).

### 3D — ResNet50 fine-tune toàn bộ

- Notebook: `src/model3/notebook/3d_full_finetune_resnet50.ipynb`
- Cấu hình: ResNet50 mở khóa toàn bộ trọng số; batch 32, head lr 1e-3, backbone lr 1e-4, weight_decay 0.01, grad clip 1.0, tối đa 30 epoch, patience 5. Ảnh 224×224, train: HFlip + ColorJitter 0.1, chuẩn hóa ImageNet; backbone ImageNet + head MLP (Linear→512, BatchNorm, ReLU, Dropout 0.3, Linear→9); AdamW, cosine annealing, seed 42, early stopping theo validation accuracy; giữ trọng số tốt nhất trước khi test
- Tham số: 24,562,761 huấn luyện / 24,562,761 tổng
- Huấn luyện: chạy 10/30 epoch, dừng sớm (early stopping), mất 6.1 phút
- Epoch tốt nhất (5): Train Loss 0.0527, Train Acc 98.46%, Val Loss 0.2567, Val Acc 94.24%
- Epoch cuối (10): Train Loss 0.0209, Train Acc 99.35%, Val Loss 0.3394, Val Acc 93.65%
- Test (864 ảnh): accuracy 92.82%, macro P 0.9205, macro R 0.9311, macro F1 0.9234, weighted F1 0.9286
- Model đã lưu: `src/model3/checkpoints/3d_full_finetune_resnet50.pt` (94.0 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.9500 | 0.9596 | 0.9548 | 99 |
| Carton | 0.9700 | 0.8981 | 0.9327 | 108 |
| Foam_box | 0.8864 | 0.8966 | 0.8914 | 87 |
| Milk_box | 0.9615 | 0.9901 | 0.9756 | 101 |
| Other | 0.7424 | 0.9800 | 0.8448 | 50 |
| PET | 0.9375 | 0.8824 | 0.9091 | 102 |
| Paper | 0.9074 | 0.9245 | 0.9159 | 106 |
| Paper_cup | 0.9722 | 1.0000 | 0.9859 | 105 |
| Plastic_cup | 0.9574 | 0.8491 | 0.9000 | 106 |
| *macro avg* | 0.9205 | 0.9311 | 0.9234 | 864 |
| *weighted avg* | 0.9323 | 0.9282 | 0.9286 | 864 |

Lớp tốt nhất: Paper_cup (F1 0.9859). Lớp kém nhất: Other (F1 0.8448).
