# Kết quả thực nghiệm

Cập nhật lần cuối: 2026-10-11 14:57. Tất cả số liệu lấy từ output của các notebook sau lần chạy lại toàn bộ ngày 2026-10-11 (GPU NVIDIA RTX 5070, PyTorch 2.14 + CUDA 13, DataLoader 8 worker). Mỗi notebook chỉ chạy một lần. Chỉ model 3 và `cnn_sequential` đặt seed; các notebook còn lại không đặt seed, nên chạy lại có thể lệch vài điểm phần trăm.

## Dữ liệu và cách đánh giá

- Dữ liệu: `data/VN_trash_classification_preprocessing`, 9 lớp (Alu, Carton, Foam_box, Milk_box, Other, PET, Paper, Paper_cup, Plastic_cup).
- Chia tập: train 9.542 ảnh, validation 1.685 ảnh, test 864 ảnh.
- Chọn epoch tốt nhất bằng tập validation; tập test chỉ dùng một lần sau khi nạp lại trọng số tốt nhất.
- Chỉ số test: accuracy, macro precision / recall / F1, weighted F1, và bảng precision / recall / F1 / support cho từng lớp (không dùng confusion matrix).

## Bảng tổng hợp

| Model | Test acc | Macro P | Macro R | Macro F1 | Weighted F1 | Epoch chạy | Best epoch | Thời gian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model 1 · M1-01 — MLP | 15.62% | 0.1451 | 0.1508 | 0.1016 | 0.0939 | 50 | 45 | 3.5 phút |
| Model 1 · M1-02 — CNN 2 block | 45.95% | 0.4757 | 0.4611 | 0.4531 | 0.4656 | 49 | 34 | 7.3 phút |
| Model 1 · M1-03 — CNN 3 block | 48.73% | 0.5196 | 0.4879 | 0.4840 | 0.4920 | 31 | 16 | 5.8 phút |
| Model 1 · M1-04 — CNN 3 block + GAP | 45.95% | 0.4792 | 0.4610 | 0.4167 | 0.4189 | 50 | 50 | 8.5 phút |
| Model 2 · CNN sequential | 64.12% | 0.6358 | 0.6330 | 0.6320 | 0.6411 | 30 | 28 | 16.0 phút |
| Model 2 · CNN parallel | 64.12% | 0.6348 | 0.6397 | 0.6339 | 0.6428 | 30 | 27 | 18.2 phút |
| Model 2 · VGG sequential | 73.38% | 0.7276 | 0.7271 | 0.7217 | 0.7330 | 30 | 29 | 29.3 phút |
| Model 2 · VGG parallel | 76.04% | 0.7561 | 0.7601 | 0.7487 | 0.7530 | 30 | 28 | 45.3 phút |
| Model 2 · ResNet (tự viết, không phải ResNet-18 chuẩn) | 69.33% | 0.6861 | 0.6840 | 0.6813 | 0.6927 | 30 | 30 | 90.7 phút |
| Model 3 · 3A — VGG16 đóng băng | 82.18% | 0.8192 | 0.8225 | 0.8162 | 0.8229 | 19 | 13 | 7.6 phút |
| Model 3 · 3B — ResNet50 đóng băng | 93.98% | 0.9376 | 0.9366 | 0.9363 | 0.9397 | 12 | 6 | 3.7 phút |
| Model 3 · 3B — EfficientNet-B0 đóng băng | 93.06% | 0.9249 | 0.9272 | 0.9245 | 0.9311 | 18 | 12 | 3.2 phút |
| Model 3 · 3C — ResNet50 + LoRA | 96.53% | 0.9635 | 0.9658 | 0.9643 | 0.9653 | 30 | 27 | 10.7 phút |
| Model 3 · 3D — ResNet50 fine-tune toàn bộ | 92.82% | 0.9205 | 0.9311 | 0.9234 | 0.9286 | 10 | 5 | 6.1 phút |

## F1 theo từng lớp

| Lớp | m1_01 | m1_02 | m1_03 | m1_04 | m2_cnn_sequential | m2_cnn_parallel | m2_vgg_sequential | m2_vgg_parallel | m2_resnet | m3_3a_frozen_vgg16 | m3_3b_frozen_resnet50 | m3_3b_frozen_efficientnet_b0 | m3_3c_lora_resnet50 | m3_3d_full_finetune_resnet50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alu | 0.0000 | 0.5427 | 0.5943 | 0.5977 | 0.6965 | 0.6337 | 0.7677 | 0.8077 | 0.7716 | 0.8835 | 0.9648 | 0.9447 | 0.9744 | 0.9548 |
| Carton | 0.3176 | 0.6778 | 0.6989 | 0.7330 | 0.7941 | 0.7814 | 0.8571 | 0.8436 | 0.8365 | 0.8265 | 0.9469 | 0.9533 | 0.9668 | 0.9327 |
| Foam_box | 0.0000 | 0.3578 | 0.4211 | 0.1611 | 0.5190 | 0.4938 | 0.5775 | 0.5000 | 0.5342 | 0.8686 | 0.9146 | 0.9012 | 0.9425 | 0.8914 |
| Milk_box | 0.0000 | 0.5795 | 0.7368 | 0.6829 | 0.7389 | 0.8325 | 0.8812 | 0.8796 | 0.8804 | 0.9128 | 0.9806 | 0.9608 | 0.9901 | 0.9756 |
| Other | 0.2903 | 0.2775 | 0.3662 | 0.4375 | 0.5283 | 0.5440 | 0.5882 | 0.7544 | 0.5405 | 0.6838 | 0.8824 | 0.8257 | 0.9515 | 0.8448 |
| PET | 0.0345 | 0.4400 | 0.3128 | 0.4257 | 0.4817 | 0.4577 | 0.6047 | 0.6549 | 0.5182 | 0.7163 | 0.9412 | 0.9246 | 0.9648 | 0.9091 |
| Paper | 0.0000 | 0.5498 | 0.5700 | 0.5260 | 0.6789 | 0.7129 | 0.7713 | 0.7867 | 0.6786 | 0.7873 | 0.9390 | 0.9519 | 0.9585 | 0.9159 |
| Paper_cup | 0.1053 | 0.5116 | 0.4103 | 0.1864 | 0.6634 | 0.6634 | 0.8585 | 0.8785 | 0.8325 | 0.9116 | 0.9488 | 0.9573 | 0.9859 | 0.9859 |
| Plastic_cup | 0.1670 | 0.1410 | 0.2456 | 0.0000 | 0.5868 | 0.5856 | 0.5894 | 0.6327 | 0.5392 | 0.7553 | 0.9083 | 0.9009 | 0.9439 | 0.9000 |

## Ghi chú khi đọc kết quả

- Số liệu Model 1 cũ trong README (M1-01 18.06%, M1-02 56.37%, M1-03 60.30%, M1-04 52.66%) đến từ các script
  `.py` ở commit `44c0af8`, nay đã bị xóa. Script `m1_03.py` ở commit đó có augmentation (HFlip + ColorJitter)
  và chuẩn hóa ImageNet; notebook hiện tại chỉ Resize + ToTensor. Hai bộ số liệu khác cấu hình, không so
  sánh trực tiếp được, và chưa có thí nghiệm riêng để xác định nguyên nhân chênh lệch.
- M1-01 (MLP) gần như không học được: ở epoch tốt nhất accuracy train 18.76%, validation 19.94%, test 15.62% (đoán ngẫu nhiên là 11.1%); 4 trong 9 lớp có F1 bằng 0. Các lần chạy khác của cùng notebook cho 5.67% và khoảng 18–20%, nghĩa là luôn ở mức thấp và dao động mạnh giữa các lần chạy.
- Model 2: cả 5 model có best epoch ở 27–30 trên 30 epoch, nên chưa model nào ngừng cải thiện trước giới hạn epoch (chưa thử tăng epoch). Cả 5 thấp hơn checkpoint cũ chọn từ các lần chạy `train.py` (0.8–6.4 điểm accuracy); hai pipeline khác nhau nên chưa quy được cho một nguyên nhân.
- Model 3: ba biến thể chỉ train head (VGG16, ResNet50, EfficientNet-B0) đều dừng sớm và có chênh lệch lớn giữa train và validation (train accuracy 98.7–99.6% so với val 86.5–93.2% ở epoch tốt nhất). LoRA (3C) cao hơn fine-tune toàn bộ (3D) 3.7 điểm accuracy, ngược với framework cũ (3D 0.9606 > 3C 0.9549) vốn dùng cấu hình khác bản notebook.
- Các notebook model 3 lưu toàn bộ `state_dict` (cả backbone đóng băng), nên `3a_frozen_vgg16.pt` nặng khoảng 520 MB.
- Mọi so sánh giữa các model chỉ dựa trên một lần chạy, một lần chia tập.

## Model 1

### M1-01 — MLP

- Notebook: `src/model1/m1_01.ipynb`
- Cấu hình: ảnh 128×128; Flatten → Linear 512 → Linear 128 → 9 (Dropout 0.2). Resize rồi `ToTensor` (không chuẩn hóa, không augmentation), batch 64, Adam lr=1e-3, `CrossEntropyLoss`, tối đa 50 epoch, early stopping theo validation loss (patience 15), nạp lại trọng số có val loss thấp nhất trước khi test
- Huấn luyện: chạy 50/50 epoch, mất 3.5 phút
- Epoch tốt nhất (45): Train Loss 2.1073, Train Acc 18.76%, Val Loss 2.1118, Val Acc 19.94%
- Epoch cuối (50): Train Loss 2.1120, Train Acc 18.20%, Val Loss 2.1493, Val Acc 19.05%
- Test (864 ảnh): accuracy 15.62%, macro P 0.1451, macro R 0.1508, macro F1 0.1016, weighted F1 0.0939
- Model đã lưu: `src/model1/checkpoints/m1_01.pt` (96.3 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.0000 | 0.0000 | 0.0000 | 99 |
| Carton | 0.2067 | 0.6852 | 0.3176 | 108 |
| Foam_box | 0.0000 | 0.0000 | 0.0000 | 87 |
| Milk_box | 0.0000 | 0.0000 | 0.0000 | 101 |
| Other | 0.7500 | 0.1800 | 0.2903 | 50 |
| PET | 0.1429 | 0.0196 | 0.0345 | 102 |
| Paper | 0.0000 | 0.0000 | 0.0000 | 106 |
| Paper_cup | 0.0976 | 0.1143 | 0.1053 | 105 |
| Plastic_cup | 0.1089 | 0.3585 | 0.1670 | 106 |
| *macro avg* | 0.1451 | 0.1508 | 0.1016 | 864 |
| *weighted avg* | 0.1113 | 0.1562 | 0.0939 | 864 |

Lớp tốt nhất: Carton (F1 0.3176). Lớp kém nhất: Alu (F1 0.0000).

### M1-02 — CNN 2 block

- Notebook: `src/model1/m1_02.ipynb`
- Cấu hình: ảnh 224×224; 2 × (Conv + BatchNorm + ReLU + MaxPool: 32, 64) → AdaptiveAvgPool 4×4 → Linear 128 → 9 (Dropout 0.2). Resize rồi `ToTensor` (không chuẩn hóa, không augmentation), batch 64, Adam lr=1e-3, `CrossEntropyLoss`, tối đa 50 epoch, early stopping theo validation loss (patience 15), nạp lại trọng số có val loss thấp nhất trước khi test
- Huấn luyện: chạy 49/50 epoch, dừng sớm (early stopping), mất 7.3 phút
- Epoch tốt nhất (34): Train Loss 0.9055, Train Acc 69.04%, Val Loss 1.2707, Val Acc 59.35%
- Epoch cuối (49): Train Loss 0.7656, Train Acc 73.83%, Val Loss 1.5445, Val Acc 55.67%
- Test (864 ảnh): accuracy 45.95%, macro P 0.4757, macro R 0.4611, macro F1 0.4531, weighted F1 0.4656
- Model đã lưu: `src/model1/checkpoints/m1_02.pt` (0.6 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.5400 | 0.5455 | 0.5427 | 99 |
| Carton | 0.8472 | 0.5648 | 0.6778 | 108 |
| Foam_box | 0.2977 | 0.4483 | 0.3578 | 87 |
| Milk_box | 0.6800 | 0.5050 | 0.5795 | 101 |
| Other | 0.1951 | 0.4800 | 0.2775 | 50 |
| PET | 0.4490 | 0.4314 | 0.4400 | 102 |
| Paper | 0.5524 | 0.5472 | 0.5498 | 106 |
| Paper_cup | 0.5000 | 0.5238 | 0.5116 | 105 |
| Plastic_cup | 0.2200 | 0.1038 | 0.1410 | 106 |
| *macro avg* | 0.4757 | 0.4611 | 0.4531 | 864 |
| *weighted avg* | 0.4971 | 0.4595 | 0.4656 | 864 |

Lớp tốt nhất: Carton (F1 0.6778). Lớp kém nhất: Plastic_cup (F1 0.1410).

### M1-03 — CNN 3 block

- Notebook: `src/model1/m1_03.ipynb`
- Cấu hình: ảnh 224×224; 3 × (Conv + BatchNorm + ReLU + MaxPool: 32, 64, 128) → AdaptiveAvgPool 4×4 → Linear 128 → 9 (Dropout 0.2). Resize rồi `ToTensor` (không chuẩn hóa, không augmentation), batch 64, Adam lr=1e-3, `CrossEntropyLoss`, tối đa 50 epoch, early stopping theo validation loss (patience 15), nạp lại trọng số có val loss thấp nhất trước khi test
- Huấn luyện: chạy 31/50 epoch, dừng sớm (early stopping), mất 5.8 phút
- Epoch tốt nhất (16): Train Loss 1.0292, Train Acc 64.66%, Val Loss 1.1974, Val Acc 59.94%
- Epoch cuối (31): Train Loss 0.7460, Train Acc 73.70%, Val Loss 1.2028, Val Acc 62.14%
- Test (864 ảnh): accuracy 48.73%, macro P 0.5196, macro R 0.4879, macro F1 0.4840, weighted F1 0.4920
- Model đã lưu: `src/model1/checkpoints/m1_03.pt` (1.4 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.5575 | 0.6364 | 0.5943 | 99 |
| Carton | 0.8333 | 0.6019 | 0.6989 | 108 |
| Foam_box | 0.6087 | 0.3218 | 0.4211 | 87 |
| Milk_box | 0.6614 | 0.8317 | 0.7368 | 101 |
| Other | 0.2826 | 0.5200 | 0.3662 | 50 |
| PET | 0.2695 | 0.3725 | 0.3128 | 102 |
| Paper | 0.6064 | 0.5377 | 0.5700 | 106 |
| Paper_cup | 0.6275 | 0.3048 | 0.4103 | 105 |
| Plastic_cup | 0.2295 | 0.2642 | 0.2456 | 106 |
| *macro avg* | 0.5196 | 0.4879 | 0.4840 | 864 |
| *weighted avg* | 0.5336 | 0.4873 | 0.4920 | 864 |

Lớp tốt nhất: Milk_box (F1 0.7368). Lớp kém nhất: Plastic_cup (F1 0.2456).

### M1-04 — CNN 3 block + GAP

- Notebook: `src/model1/m1_04.ipynb`
- Cấu hình: ảnh 224×224; 3 × (Conv + BatchNorm + ReLU + MaxPool: 32, 64, 128) → Global Average Pooling → Linear 9. Resize rồi `ToTensor` (không chuẩn hóa, không augmentation), batch 64, Adam lr=1e-3, `CrossEntropyLoss`, tối đa 50 epoch, early stopping theo validation loss (patience 15), nạp lại trọng số có val loss thấp nhất trước khi test
- Huấn luyện: chạy 50/50 epoch, mất 8.5 phút
- Epoch tốt nhất (50): Train Loss 1.0731, Train Acc 64.70%, Val Loss 1.3353, Val Acc 55.43%
- Epoch cuối (50): Train Loss 1.0731, Train Acc 64.70%, Val Loss 1.3353, Val Acc 55.43%
- Test (864 ảnh): accuracy 45.95%, macro P 0.4792, macro R 0.4610, macro F1 0.4167, weighted F1 0.4189
- Model đã lưu: `src/model1/checkpoints/m1_04.pt` (0.4 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.6933 | 0.5253 | 0.5977 | 99 |
| Carton | 0.8434 | 0.6481 | 0.7330 | 108 |
| Foam_box | 0.1935 | 0.1379 | 0.1611 | 87 |
| Milk_box | 0.6731 | 0.6931 | 0.6829 | 101 |
| Other | 0.3590 | 0.5600 | 0.4375 | 50 |
| PET | 0.3029 | 0.7157 | 0.4257 | 102 |
| Paper | 0.4010 | 0.7642 | 0.5260 | 106 |
| Paper_cup | 0.8462 | 0.1048 | 0.1864 | 105 |
| Plastic_cup | 0.0000 | 0.0000 | 0.0000 | 106 |
| *macro avg* | 0.4792 | 0.4610 | 0.4167 | 864 |
| *weighted avg* | 0.4916 | 0.4595 | 0.4189 | 864 |

Lớp tốt nhất: Carton (F1 0.7330). Lớp kém nhất: Plastic_cup (F1 0.0000).

## Model 2

### CNN sequential

- Notebook: `src/model2/notebook/cnn_sequential.ipynb`
- Cấu hình: 3 stage × 2 Conv 3×3 (48, 96, 192) + BatchNorm, MaxPool mỗi stage → GAP → Linear 256 → 9 (Dropout 0.3). Ảnh 224×224, train: HFlip + Rotation 10° + ColorJitter 0.1, chuẩn hóa ImageNet; batch 32, AdamW lr=1e-4 weight_decay=5e-4, cosine annealing, `CrossEntropyLoss` có trọng số lớp (cân bằng lớp), chạy đủ 30 epoch (không early stopping), giữ trọng số có validation accuracy cao nhất
- Tham số: 697.849
- Huấn luyện: chạy 30/30 epoch, mất 16.0 phút
- Epoch tốt nhất (28): Train Loss 0.9434, Train Acc 69.14%, Val Loss 1.0261, Val Acc 68.25%
- Epoch cuối (30): Train Loss 0.9398, Train Acc 69.87%, Val Loss 1.0201, Val Acc 68.13%
- Test (864 ảnh): accuracy 64.12%, macro P 0.6358, macro R 0.6330, macro F1 0.6320, weighted F1 0.6411
- Model đã lưu: `src/model2/checkpoints/cnn_sequential.pt` (2.7 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.6863 | 0.7071 | 0.6965 | 99 |
| Carton | 0.8438 | 0.7500 | 0.7941 | 108 |
| Foam_box | 0.5775 | 0.4713 | 0.5190 | 87 |
| Milk_box | 0.7353 | 0.7426 | 0.7389 | 101 |
| Other | 0.5000 | 0.5600 | 0.5283 | 50 |
| PET | 0.5169 | 0.4510 | 0.4817 | 102 |
| Paper | 0.6607 | 0.6981 | 0.6789 | 106 |
| Paper_cup | 0.6800 | 0.6476 | 0.6634 | 105 |
| Plastic_cup | 0.5221 | 0.6698 | 0.5868 | 106 |
| *macro avg* | 0.6358 | 0.6330 | 0.6320 | 864 |
| *weighted avg* | 0.6459 | 0.6412 | 0.6411 | 864 |

Lớp tốt nhất: Carton (F1 0.7941). Lớp kém nhất: PET (F1 0.4817).

### CNN parallel

- Notebook: `src/model2/notebook/cnn_parallel.ipynb`
- Cấu hình: 3 nhánh Conv song song (1 conv, 2 conv, conv dilation 2; 32 kênh mỗi nhánh) nối thành 96 kênh → 3 khối residual stride 2 (64, 128, 192) → GAP → Linear 256 → 9 (Dropout 0.4). Ảnh 224×224, train: HFlip + Rotation 10° + ColorJitter 0.1, chuẩn hóa ImageNet; batch 32, AdamW lr=1e-4 weight_decay=5e-4, cosine annealing, `CrossEntropyLoss` có trọng số lớp (cân bằng lớp), chạy đủ 30 epoch (không early stopping), giữ trọng số có validation accuracy cao nhất
- Huấn luyện: chạy 30/30 epoch, mất 18.2 phút
- Epoch tốt nhất (27 — suy ra từ log: epoch có Val Acc cao nhất, notebook không in): Train Acc 69.08%, Val Acc 69.73%
- Epoch cuối (30): Train Acc 69.06%, Val Acc 69.38%
- Test (864 ảnh): accuracy 64.12%, macro P 0.6348, macro R 0.6397, macro F1 0.6339, weighted F1 0.6428
- Model đã lưu: `src/model2/checkpoints/cnn_parallel.pt` (3.7 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.6214 | 0.6465 | 0.6337 | 99 |
| Carton | 0.7850 | 0.7778 | 0.7814 | 108 |
| Foam_box | 0.5333 | 0.4598 | 0.4938 | 87 |
| Milk_box | 0.8542 | 0.8119 | 0.8325 | 101 |
| Other | 0.4533 | 0.6800 | 0.5440 | 50 |
| PET | 0.4646 | 0.4510 | 0.4577 | 102 |
| Paper | 0.7500 | 0.6792 | 0.7129 | 106 |
| Paper_cup | 0.6907 | 0.6381 | 0.6634 | 105 |
| Plastic_cup | 0.5603 | 0.6132 | 0.5856 | 106 |
| *macro avg* | 0.6348 | 0.6397 | 0.6339 | 864 |
| *weighted avg* | 0.6487 | 0.6412 | 0.6428 | 864 |

Lớp tốt nhất: Milk_box (F1 0.8325). Lớp kém nhất: PET (F1 0.4577).

### VGG sequential

- Notebook: `src/model2/notebook/vgg_sequential.ipynb`
- Cấu hình: kiểu VGG-16 có BatchNorm (13 conv: 64, 128, 256, 512, 512) → GAP → Linear 128 → 9 (Dropout 0.6). Ảnh 224×224, train: HFlip + Rotation 10° + ColorJitter 0.1, chuẩn hóa ImageNet; batch 32, AdamW lr=1e-4 weight_decay=5e-4, cosine annealing, `CrossEntropyLoss` có trọng số lớp (cân bằng lớp), chạy đủ 30 epoch (không early stopping), giữ trọng số có validation accuracy cao nhất
- Huấn luyện: chạy 30/30 epoch, mất 29.3 phút
- Epoch tốt nhất (29 — suy ra từ log: epoch có Val Acc cao nhất, notebook không in): Train Acc 84.24%, Val Acc 80.71%
- Epoch cuối (30): Train Acc 85.10%, Val Acc 80.00%
- Test (864 ảnh): accuracy 73.38%, macro P 0.7276, macro R 0.7271, macro F1 0.7217, weighted F1 0.7330
- Model đã lưu: `src/model2/checkpoints/vgg_sequential.pt` (56.5 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.7677 | 0.7677 | 0.7677 | 99 |
| Carton | 0.8824 | 0.8333 | 0.8571 | 108 |
| Foam_box | 0.7455 | 0.4713 | 0.5775 | 87 |
| Milk_box | 0.8812 | 0.8812 | 0.8812 | 101 |
| Other | 0.5072 | 0.7000 | 0.5882 | 50 |
| PET | 0.5752 | 0.6373 | 0.6047 | 102 |
| Paper | 0.7350 | 0.8113 | 0.7713 | 106 |
| Paper_cup | 0.8505 | 0.8667 | 0.8585 | 105 |
| Plastic_cup | 0.6040 | 0.5755 | 0.5894 | 106 |
| *macro avg* | 0.7276 | 0.7271 | 0.7217 | 864 |
| *weighted avg* | 0.7412 | 0.7338 | 0.7330 | 864 |

Lớp tốt nhất: Milk_box (F1 0.8812). Lớp kém nhất: Foam_box (F1 0.5775).

### VGG parallel

- Notebook: `src/model2/notebook/vgg_parallel.ipynb`
- Cấu hình: 3 nhánh đầu vào (3×3, 3×3 dilation 2, 5×5; 64 kênh mỗi nhánh) nối thành 192 kênh → 5 stage kiểu VGG → GAP → Linear 128 → 9 (Dropout 0.6). Ảnh 224×224, train: HFlip + Rotation 10° + ColorJitter 0.1, chuẩn hóa ImageNet; batch 32, AdamW lr=1e-4 weight_decay=5e-4, cosine annealing, `CrossEntropyLoss` có trọng số lớp (cân bằng lớp), chạy đủ 30 epoch (không early stopping), giữ trọng số có validation accuracy cao nhất
- Huấn luyện: chạy 30/30 epoch, mất 45.3 phút
- Epoch tốt nhất (28 — suy ra từ log: epoch có Val Acc cao nhất, notebook không in): Train Acc 87.63%, Val Acc 82.08%
- Epoch cuối (30): Train Acc 87.82%, Val Acc 81.72%
- Test (864 ảnh): accuracy 76.04%, macro P 0.7561, macro R 0.7601, macro F1 0.7487, weighted F1 0.7530
- Model đã lưu: `src/model2/checkpoints/vgg_parallel.pt` (20.6 MB)

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Alu | 0.7706 | 0.8485 | 0.8077 | 99 |
| Carton | 0.8641 | 0.8241 | 0.8436 | 108 |
| Foam_box | 0.7333 | 0.3793 | 0.5000 | 87 |
| Milk_box | 0.8261 | 0.9406 | 0.8796 | 101 |
| Other | 0.6719 | 0.8600 | 0.7544 | 50 |
| PET | 0.5968 | 0.7255 | 0.6549 | 102 |
| Paper | 0.7905 | 0.7830 | 0.7867 | 106 |
| Paper_cup | 0.8624 | 0.8952 | 0.8785 | 105 |
| Plastic_cup | 0.6889 | 0.5849 | 0.6327 | 106 |
| *macro avg* | 0.7561 | 0.7601 | 0.7487 | 864 |
| *weighted avg* | 0.7624 | 0.7604 | 0.7530 | 864 |

Lớp tốt nhất: Milk_box (F1 0.8796). Lớp kém nhất: Foam_box (F1 0.5000).

### ResNet (tự viết, không phải ResNet-18 chuẩn)

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
