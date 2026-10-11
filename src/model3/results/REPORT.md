# Model 3 - Transfer learning results

Pretrained ImageNet backbones on the preprocessed VN trash dataset (9 classes, 9,542 train / 1,685 val / 864 test images). Model selection is on validation accuracy; every number below is on the held-out test split unless stated otherwise.

- Runs: **5**
- Total wall time: **32.3 min**
- Best test accuracy: **3d_full_finetune_resnet50** at **0.9606** (F1 macro 0.9597)
- Device: NVIDIA GeForce RTX 5070, torch 2.14.0+cu130, AMP bfloat16

## Summary

| Variant | Run | Backbone | Strategy | Trainable params | Trainable % | Test acc | Test bal-acc | Test P (macro) | Test R (macro) | Test F1 (macro) | Test F1 (weighted) | Val acc | Best epoch | Epochs | Train time (s) | Epoch (s) | ms/batch | Model img/s | Pipeline img/s | Peak GPU (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3A | 3a_frozen_vgg16 | VGG16 | frozen | 2,103,305 | 1.54% | 0.8380 | 0.8400 | 0.8392 | 0.8400 | 0.8330 | 0.8395 | 0.8617 | 3 | 9 | 138.3 | 11.6 | 60.4 | 1060 | 884 | 2910 |
| 3B | 3b_frozen_efficientnet_b0 | EfficientNet-B0 | frozen | 661,513 | 14.17% | 0.9201 | 0.9164 | 0.9148 | 0.9164 | 0.9132 | 0.9202 | 0.9122 | 4 | 10 | 131.7 | 10.9 | 105.9 | 604 | 877 | 472 |
| 3B | 3b_frozen_resnet50 | ResNet50 | frozen | 1,054,729 | 4.29% | 0.9294 | 0.9309 | 0.9227 | 0.9309 | 0.9227 | 0.9303 | 0.9341 | 7 | 13 | 163.3 | 10.7 | 76.5 | 836 | 737 | 851 |
| 3C | 3c_lora_resnet50 | ResNet50 | lora | 1,570,825 | 6.26% | 0.9549 | 0.9554 | 0.9529 | 0.9554 | 0.9536 | 0.9551 | 0.9543 | 23 | 29 | 1036.9 | 29.4 | 152.2 | 210 | 351 | 630 |
| 3D | 3d_full_finetune_resnet50 | ResNet50 | full_finetune | 24,562,761 | 100.00% | 0.9606 | 0.9613 | 0.9589 | 0.9613 | 0.9597 | 0.9607 | 0.9484 | 5 | 10 | 283.7 | 26.0 | 61.1 | 524 | 704 | 1785 |

## Variants

### 3A - 3a_frozen_vgg16

Freeze all of VGG16, train only the MLP head

- Backbone: VGG16 (4096-d features)
- Head: MLP(4096->512->9, dropout=0.3)
- Trainable: 2,103,305 / 136,363,849 (1.54%)
- Epochs: 9 ran (best at 3, early stopped)
- Test: acc 0.8380 | balanced acc 0.8400 | P 0.8392 | R 0.8400 | F1 0.8330 (weighted F1 0.8395)
- Val: acc 0.8617 | F1 macro 0.8617
- Time: 138.3s training (11.6s/epoch), inference 60.4 ms/batch = 1060 img/s model-only (884 img/s end-to-end over the test loader), peak GPU 2910 MB

Per-class test scores:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Alu | 0.9118 | 0.9394 | 0.9254 | 99 |
| Carton | 0.9747 | 0.7130 | 0.8235 | 108 |
| Foam_box | 0.9487 | 0.8506 | 0.8970 | 87 |
| Milk_box | 0.9126 | 0.9307 | 0.9216 | 101 |
| Other | 0.5915 | 0.8400 | 0.6942 | 50 |
| PET | 0.6825 | 0.8431 | 0.7544 | 102 |
| Paper | 0.8384 | 0.7830 | 0.8098 | 106 |
| Paper_cup | 0.8772 | 0.9524 | 0.9132 | 105 |
| Plastic_cup | 0.8152 | 0.7075 | 0.7576 | 106 |

Artefacts: `results/3a_frozen_vgg16/` (curves.png, confusion_matrix_test.png, classification_report_test.txt, history.csv)

### 3B - 3b_frozen_efficientnet_b0

Freeze all of efficientnet_b0, train only the MLP head

- Backbone: EfficientNet-B0 (1280-d features)
- Head: MLP(1280->512->9, dropout=0.3)
- Trainable: 661,513 / 4,669,061 (14.17%)
- Epochs: 10 ran (best at 4, early stopped)
- Test: acc 0.9201 | balanced acc 0.9164 | P 0.9148 | R 0.9164 | F1 0.9132 (weighted F1 0.9202)
- Val: acc 0.9122 | F1 macro 0.9123
- Time: 131.7s training (10.9s/epoch), inference 105.9 ms/batch = 604 img/s model-only (877 img/s end-to-end over the test loader), peak GPU 472 MB

Per-class test scores:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Alu | 0.9406 | 0.9596 | 0.9500 | 99 |
| Carton | 0.9204 | 0.9630 | 0.9412 | 108 |
| Foam_box | 0.9855 | 0.7816 | 0.8718 | 87 |
| Milk_box | 0.9700 | 0.9604 | 0.9652 | 101 |
| Other | 0.7377 | 0.9000 | 0.8108 | 50 |
| PET | 0.9167 | 0.8627 | 0.8889 | 102 |
| Paper | 0.9519 | 0.9340 | 0.9429 | 106 |
| Paper_cup | 0.9537 | 0.9810 | 0.9671 | 105 |
| Plastic_cup | 0.8571 | 0.9057 | 0.8807 | 106 |

Artefacts: `results/3b_frozen_efficientnet_b0/` (curves.png, confusion_matrix_test.png, classification_report_test.txt, history.csv)

### 3B - 3b_frozen_resnet50

Freeze all of resnet50, train only the MLP head

- Backbone: ResNet50 (2048-d features)
- Head: MLP(2048->512->9, dropout=0.3)
- Trainable: 1,054,729 / 24,562,761 (4.29%)
- Epochs: 13 ran (best at 7, early stopped)
- Test: acc 0.9294 | balanced acc 0.9309 | P 0.9227 | R 0.9309 | F1 0.9227 (weighted F1 0.9303)
- Val: acc 0.9341 | F1 macro 0.9341
- Time: 163.3s training (10.7s/epoch), inference 76.5 ms/batch = 836 img/s model-only (737 img/s end-to-end over the test loader), peak GPU 851 MB

Per-class test scores:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Alu | 0.9143 | 0.9697 | 0.9412 | 99 |
| Carton | 0.9700 | 0.8981 | 0.9327 | 108 |
| Foam_box | 0.9512 | 0.8966 | 0.9231 | 87 |
| Milk_box | 0.9712 | 1.0000 | 0.9854 | 101 |
| Other | 0.6857 | 0.9600 | 0.8000 | 50 |
| PET | 1.0000 | 0.8137 | 0.8973 | 102 |
| Paper | 0.9434 | 0.9434 | 0.9434 | 106 |
| Paper_cup | 0.9369 | 0.9905 | 0.9630 | 105 |
| Plastic_cup | 0.9320 | 0.9057 | 0.9187 | 106 |

Artefacts: `results/3b_frozen_resnet50/` (curves.png, confusion_matrix_test.png, classification_report_test.txt, history.csv)

### 3C - 3c_lora_resnet50

Frozen resnet50 + LoRA adapters, train adapters + MLP head

- Backbone: ResNet50 (2048-d features)
- Head: MLP(2048->512->9, dropout=0.3)
- Trainable: 1,570,825 / 25,078,857 (6.26%)
- LoRA: rank 8, alpha 16.0, 29 adapted layers, 516,096 adapter params
- Epochs: 29 ran (best at 23, early stopped)
- Test: acc 0.9549 | balanced acc 0.9554 | P 0.9529 | R 0.9554 | F1 0.9536 (weighted F1 0.9551)
- Val: acc 0.9543 | F1 macro 0.9545
- Time: 1036.9s training (29.4s/epoch), inference 152.2 ms/batch = 210 img/s model-only (351 img/s end-to-end over the test loader), peak GPU 630 MB

Per-class test scores:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Alu | 0.9796 | 0.9697 | 0.9746 | 99 |
| Carton | 0.9806 | 0.9352 | 0.9573 | 108 |
| Foam_box | 0.8557 | 0.9540 | 0.9022 | 87 |
| Milk_box | 0.9901 | 0.9901 | 0.9901 | 101 |
| Other | 0.9231 | 0.9600 | 0.9412 | 50 |
| PET | 0.9895 | 0.9216 | 0.9543 | 102 |
| Paper | 0.9444 | 0.9623 | 0.9533 | 106 |
| Paper_cup | 0.9813 | 1.0000 | 0.9906 | 105 |
| Plastic_cup | 0.9320 | 0.9057 | 0.9187 | 106 |

Artefacts: `results/3c_lora_resnet50/` (curves.png, confusion_matrix_test.png, classification_report_test.txt, history.csv)

### 3D - 3d_full_finetune_resnet50

Fine-tune every weight of resnet50 together with the MLP head

- Backbone: ResNet50 (2048-d features)
- Head: MLP(2048->512->9, dropout=0.3)
- Trainable: 24,562,761 / 24,562,761 (100.00%)
- Epochs: 10 ran (best at 5, early stopped)
- Test: acc 0.9606 | balanced acc 0.9613 | P 0.9589 | R 0.9613 | F1 0.9597 (weighted F1 0.9607)
- Val: acc 0.9484 | F1 macro 0.9489
- Time: 283.7s training (26.0s/epoch), inference 61.1 ms/batch = 524 img/s model-only (704 img/s end-to-end over the test loader), peak GPU 1785 MB

Per-class test scores:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Alu | 0.9600 | 0.9697 | 0.9648 | 99 |
| Carton | 0.9636 | 0.9815 | 0.9725 | 108 |
| Foam_box | 0.9213 | 0.9425 | 0.9318 | 87 |
| Milk_box | 0.9899 | 0.9703 | 0.9800 | 101 |
| Other | 0.9245 | 0.9800 | 0.9515 | 50 |
| PET | 1.0000 | 0.9020 | 0.9485 | 102 |
| Paper | 0.9626 | 0.9717 | 0.9671 | 106 |
| Paper_cup | 0.9905 | 0.9905 | 0.9905 | 105 |
| Plastic_cup | 0.9174 | 0.9434 | 0.9302 | 106 |

Artefacts: `results/3d_full_finetune_resnet50/` (curves.png, confusion_matrix_test.png, classification_report_test.txt, history.csv)
