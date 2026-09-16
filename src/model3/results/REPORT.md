# Model 3 - Transfer learning results

Pretrained ImageNet backbones on the preprocessed VN trash dataset (9 classes, 9,542 train / 1,685 val / 864 test images). Model selection is on validation accuracy; every number below is on the held-out test split unless stated otherwise.

- Runs: **8**
- Total wall time: **75.6 min**
- Best test accuracy: **3d_full_finetune_resnet50** at **0.9606** (F1 macro 0.9597)
- Device: NVIDIA GeForce RTX 5070, torch 2.14.0+cu130, AMP bfloat16

## Summary

| Variant | Run | Backbone | Strategy | Trainable params | Trainable % | Test acc | Test bal-acc | Test P (macro) | Test R (macro) | Test F1 (macro) | Test F1 (weighted) | Val acc | Best epoch | Epochs | Train time (s) | Epoch (s) | ms/batch | Model img/s | Pipeline img/s | Peak GPU (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3A | 3a_frozen_mobilenet_v2 | MobileNetV2 | frozen | 661,513 | 22.93% | 0.8750 | 0.8746 | 0.8726 | 0.8746 | 0.8714 | 0.8761 | 0.8991 | 9 | 15 | 278.8 | 13.6 | 65.5 | 978 | 1231 | 785 |
| 3B | 3b_frozen_efficientnet_b0 | EfficientNet-B0 | frozen | 661,513 | 14.17% | 0.9201 | 0.9164 | 0.9148 | 0.9164 | 0.9132 | 0.9202 | 0.9122 | 4 | 10 | 131.7 | 10.9 | 105.9 | 604 | 877 | 472 |
| 3B | 3b_frozen_resnet50 | ResNet50 | frozen | 1,054,729 | 4.29% | 0.9294 | 0.9309 | 0.9227 | 0.9309 | 0.9227 | 0.9303 | 0.9341 | 7 | 13 | 163.3 | 10.7 | 76.5 | 836 | 737 | 851 |
| 3B | 3b_frozen_vit_b_16 | ViT-B/16 | frozen | 399,369 | 0.46% | 0.9317 | 0.9330 | 0.9261 | 0.9330 | 0.9284 | 0.9317 | 0.9169 | 4 | 10 | 129.6 | 10.3 | 95.9 | 667 | 640 | 770 |
| 3C | 3c_lora_resnet50 | ResNet50 | lora | 1,570,825 | 6.26% | 0.9549 | 0.9554 | 0.9529 | 0.9554 | 0.9536 | 0.9551 | 0.9543 | 23 | 29 | 1036.9 | 29.4 | 152.2 | 210 | 351 | 630 |
| 3C | 3c_lora_vit_b_16 | ViT-B/16 | lora | 1,136,649 | 1.31% | 0.9537 | 0.9534 | 0.9479 | 0.9534 | 0.9501 | 0.9538 | 0.9389 | 25 | 30 | 1450.2 | 42.2 | 142.4 | 225 | 401 | 2530 |
| 3D | 3d_full_finetune_resnet50 | ResNet50 | full_finetune | 24,562,761 | 100.00% | 0.9606 | 0.9613 | 0.9589 | 0.9613 | 0.9597 | 0.9607 | 0.9484 | 5 | 10 | 283.7 | 26.0 | 61.1 | 524 | 704 | 1785 |
| 3D | 3d_full_finetune_vit_b_16 | ViT-B/16 | full_finetune | 86,198,025 | 100.00% | 0.9468 | 0.9492 | 0.9459 | 0.9492 | 0.9470 | 0.9466 | 0.9442 | 12 | 17 | 795.1 | 43.7 | 98.0 | 327 | 408 | 3476 |

## Variants

### 3A - 3a_frozen_mobilenet_v2

Freeze all of MobileNetV2, train only the MLP head

- Backbone: MobileNetV2 (1280-d features)
- Head: MLP(1280->512->9, dropout=0.3)
- Trainable: 661,513 / 2,885,385 (22.93%)
- Epochs: 15 ran (best at 9, early stopped)
- Test: acc 0.8750 | balanced acc 0.8746 | P 0.8726 | R 0.8746 | F1 0.8714 (weighted F1 0.8761)
- Val: acc 0.8991 | F1 macro 0.8990
- Time: 278.8s training (13.6s/epoch), inference 65.5 ms/batch = 978 img/s model-only (1231 img/s end-to-end over the test loader), peak GPU 785 MB

Per-class test scores:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Alu | 0.9278 | 0.9091 | 0.9184 | 99 |
| Carton | 0.8772 | 0.9259 | 0.9009 | 108 |
| Foam_box | 0.9747 | 0.8851 | 0.9277 | 87 |
| Milk_box | 0.9485 | 0.9109 | 0.9293 | 101 |
| Other | 0.7049 | 0.8600 | 0.7748 | 50 |
| PET | 0.7414 | 0.8431 | 0.7890 | 102 |
| Paper | 0.9278 | 0.8491 | 0.8867 | 106 |
| Paper_cup | 0.8909 | 0.9333 | 0.9116 | 105 |
| Plastic_cup | 0.8602 | 0.7547 | 0.8040 | 106 |

Artefacts: `results/3a_frozen_mobilenet_v2/` (curves.png, confusion_matrix_test.png, classification_report_test.txt, history.csv)

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

### 3B - 3b_frozen_vit_b_16

Freeze all of vit_b_16, train only the MLP head

- Backbone: ViT-B/16 (768-d features)
- Head: MLP(768->512->9, dropout=0.3)
- Trainable: 399,369 / 86,198,025 (0.46%)
- Epochs: 10 ran (best at 4, early stopped)
- Test: acc 0.9317 | balanced acc 0.9330 | P 0.9261 | R 0.9330 | F1 0.9284 (weighted F1 0.9317)
- Val: acc 0.9169 | F1 macro 0.9174
- Time: 129.6s training (10.3s/epoch), inference 95.9 ms/batch = 667 img/s model-only (640 img/s end-to-end over the test loader), peak GPU 770 MB

Per-class test scores:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Alu | 0.9400 | 0.9495 | 0.9447 | 99 |
| Carton | 0.9694 | 0.8796 | 0.9223 | 108 |
| Foam_box | 0.9540 | 0.9540 | 0.9540 | 87 |
| Milk_box | 0.9798 | 0.9604 | 0.9700 | 101 |
| Other | 0.7966 | 0.9400 | 0.8624 | 50 |
| PET | 0.9200 | 0.9020 | 0.9109 | 102 |
| Paper | 0.9018 | 0.9528 | 0.9266 | 106 |
| Paper_cup | 0.9545 | 1.0000 | 0.9767 | 105 |
| Plastic_cup | 0.9192 | 0.8585 | 0.8878 | 106 |

Artefacts: `results/3b_frozen_vit_b_16/` (curves.png, confusion_matrix_test.png, classification_report_test.txt, history.csv)

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

### 3C - 3c_lora_vit_b_16

Frozen vit_b_16 + LoRA adapters, train adapters + MLP head

- Backbone: ViT-B/16 (768-d features)
- Head: MLP(768->512->9, dropout=0.3)
- Trainable: 1,136,649 / 86,935,305 (1.31%)
- LoRA: rank 8, alpha 16.0, 24 adapted layers, 737,280 adapter params
- Epochs: 30 ran (best at 25)
- Test: acc 0.9537 | balanced acc 0.9534 | P 0.9479 | R 0.9534 | F1 0.9501 (weighted F1 0.9538)
- Val: acc 0.9389 | F1 macro 0.9387
- Time: 1450.2s training (42.2s/epoch), inference 142.4 ms/batch = 225 img/s model-only (401 img/s end-to-end over the test loader), peak GPU 2530 MB

Per-class test scores:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Alu | 0.9505 | 0.9697 | 0.9600 | 99 |
| Carton | 0.9810 | 0.9537 | 0.9671 | 108 |
| Foam_box | 0.9405 | 0.9080 | 0.9240 | 87 |
| Milk_box | 0.9902 | 1.0000 | 0.9951 | 101 |
| Other | 0.8421 | 0.9600 | 0.8972 | 50 |
| PET | 0.9588 | 0.9118 | 0.9347 | 102 |
| Paper | 0.9612 | 0.9340 | 0.9474 | 106 |
| Paper_cup | 0.9813 | 1.0000 | 0.9906 | 105 |
| Plastic_cup | 0.9259 | 0.9434 | 0.9346 | 106 |

Artefacts: `results/3c_lora_vit_b_16/` (curves.png, confusion_matrix_test.png, classification_report_test.txt, history.csv)

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

### 3D - 3d_full_finetune_vit_b_16

Fine-tune every weight of vit_b_16 together with the MLP head

- Backbone: ViT-B/16 (768-d features)
- Head: MLP(768->512->9, dropout=0.3)
- Trainable: 86,198,025 / 86,198,025 (100.00%)
- Epochs: 17 ran (best at 12, early stopped)
- Test: acc 0.9468 | balanced acc 0.9492 | P 0.9459 | R 0.9492 | F1 0.9470 (weighted F1 0.9466)
- Val: acc 0.9442 | F1 macro 0.9443
- Time: 795.1s training (43.7s/epoch), inference 98.0 ms/batch = 327 img/s model-only (408 img/s end-to-end over the test loader), peak GPU 3476 MB

Per-class test scores:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Alu | 0.9490 | 0.9394 | 0.9442 | 99 |
| Carton | 0.9623 | 0.9444 | 0.9533 | 108 |
| Foam_box | 1.0000 | 0.9655 | 0.9825 | 87 |
| Milk_box | 0.9709 | 0.9901 | 0.9804 | 101 |
| Other | 0.9074 | 0.9800 | 0.9423 | 50 |
| PET | 0.8692 | 0.9118 | 0.8900 | 102 |
| Paper | 0.9712 | 0.9528 | 0.9619 | 106 |
| Paper_cup | 0.9545 | 1.0000 | 0.9767 | 105 |
| Plastic_cup | 0.9286 | 0.8585 | 0.8922 | 106 |

Artefacts: `results/3d_full_finetune_vit_b_16/` (curves.png, confusion_matrix_test.png, classification_report_test.txt, history.csv)
