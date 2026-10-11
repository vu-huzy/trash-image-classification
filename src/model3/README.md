# Model 3 — Transfer learning với pretrained backbone

Bốn biến thể chuyển giao học tập (transfer learning) trên bộ dữ liệu rác VN đã tiền xử lý
(`data/VN_trash_classification_preprocessing`, 9 lớp — 9.542 train / 1.685 val / 864 test).

| Biến thể | Backbone | Chiến lược | Ý tưởng |
| --- | --- | --- | --- |
| **3A** | VGG16 | `frozen` | Đóng băng toàn bộ VGG16 (giữ 2 lớp fc pretrained → đặc trưng 4096-d), chỉ train MLP head — baseline |
| **3B** | ResNet50, EfficientNet-B0 | `frozen` | Cùng công thức như 3A nhưng backbone mạnh hơn → so sánh chất lượng đặc trưng ImageNet |
| **3C** | backbone thắng ở 3B | `lora` | Backbone vẫn đóng băng, nhưng chèn LoRA adapter để đặc trưng thích nghi với domain rác VN |
| **3D** | backbone thắng ở 3B | `full_finetune` | Mở băng toàn bộ, fine-tune mọi trọng số |

3C và 3D dùng đúng backbone đạt **validation accuracy** cao nhất ở 3B, nên ba chiến lược
(frozen → LoRA → full finetune) được so sánh trên cùng một nền đặc trưng. Tập test chỉ được
dùng một lần ở cuối mỗi run, không tham gia chọn model.

## Cấu trúc thư mục

```
src/model3/
├── config.py          # Bảng thí nghiệm: mọi tham số khác nhau giữa 3A–3D nằm ở đây
├── data.py            # DataLoader từ data/VN_trash_classification_preprocessing
├── backbones.py       # Tải pretrained backbone, bỏ classifier gốc → feature extractor
├── heads.py           # >>> MLP head (dùng chung cho cả 4 biến thể)
├── lora.py            # >>> LoRA tự viết: LoRALinear, LoRAConv2d, inject_lora()
├── model_builder.py   # >>> Nơi 3 chiến lược frozen / lora / full_finetune lộ rõ nhất
├── engine.py          # Vòng train/eval, early stopping, checkpoint, đo thời gian
├── metrics.py         # acc / precision / recall / F1, confusion matrix, learning curve
├── report.py          # Tổng hợp mọi run → results/summary.csv + results/REPORT.md
├── train.py           # CLI chạy 1 thí nghiệm
├── run_all.py         # Chạy tuần tự 3A → 3B(x2) → 3C → 3D rồi sinh báo cáo
├── pretrained/        # Nơi tải weight ImageNet (TORCH_HOME), ~990 MB cho VGG16/ResNet50/EfficientNet-B0 + các weight cũ MobileNetV2/ViT
├── checkpoints/       # <run>.pt — trọng số tốt nhất theo val accuracy
└── results/           # <run>/{metrics.json, history.csv, curves.png, confusion_matrix_test.png,
                       #         classification_report_test.txt, train_log.txt}
                       # + summary.csv, REPORT.md
```

Muốn xem từng khái niệm nằm ở đâu:

- **MLP head** → [`heads.py`](heads.py): `Linear → BatchNorm → ReLU → Dropout → Linear`.
  Giống nhau ở cả 4 biến thể để khác biệt điểm số chỉ đến từ chiến lược backbone.
- **LoRA** → [`lora.py`](lora.py): `y = W₀x + (α/r)·BAx`, `W₀` đóng băng, `B` khởi tạo bằng 0
  nên lúc bắt đầu model y hệt bản pretrained. Có hai loại adapter: `LoRALinear` (transformer)
  và `LoRAConv2d` (CNN, phân rã thành conv rank-r rồi conv 1×1).
- **freeze / finetune** → [`model_builder.py`](model_builder.py): hàm `build_model()` có ba
  nhánh `frozen` / `lora` / `full_finetune`, ghi rõ cái gì bị đóng băng, cái gì chạy `no_grad`,
  và BatchNorm có được cập nhật running stats hay không.

## Cách chạy

Chạy toàn bộ nghiên cứu (30 epoch/biến thể + early stopping patience 5–6):

```bash
.venv/Scripts/python.exe src/model3/run_all.py
```

Chạy riêng một biến thể:

```bash
.venv/Scripts/python.exe src/model3/train.py --experiment 3a
.venv/Scripts/python.exe src/model3/train.py --experiment 3b --backbone efficientnet_b0
.venv/Scripts/python.exe src/model3/train.py --experiment 3c --backbone resnet50
.venv/Scripts/python.exe src/model3/train.py --experiment 3d --backbone resnet50
```

Các cờ hữu ích của `run_all.py`: `--only 3c 3d` (chỉ chạy một số biến thể),
`--winner resnet50` (ép backbone cho 3C/3D thay vì lấy từ 3B), `--skip-existing`
(bỏ qua run đã có `metrics.json`), `--epochs N`, `--batch-size N`, `--no-amp`.

Sinh lại báo cáo từ các run đã có:

```bash
.venv/Scripts/python.exe src/model3/report.py
```

## Kết quả

Chi tiết đầy đủ: [`results/REPORT.md`](results/REPORT.md) (bảng so sánh + từng biến thể +
per-class) và [`results/summary.csv`](results/summary.csv) (máy đọc được).

Mỗi run ghi lại: accuracy, balanced accuracy, precision/recall/F1 ở cả dạng **macro**
(coi 9 lớp ngang nhau) và **weighted** (theo số lượng mẫu), điểm từng lớp, số epoch đã chạy,
epoch tốt nhất, tổng thời gian train, thời gian mỗi epoch, tốc độ inference (img/s),
peak GPU memory, và số tham số trainable so với tổng tham số.

Kết quả (RTX 5070, 5 run, sắp theo test accuracy). **Chỉ 3A (VGG16) được chạy lại** sau khi đổi
thiết kế; 4 run còn lại (3B ×2, 3C, 3D — ResNet50/EfficientNet-B0) là kết quả của lần chạy đầy đủ
trước đó, code của chúng không đổi:

| Biến thể | Backbone | Chiến lược | Trainable | Val acc | **Test acc** | Test F1 macro | Train | Peak GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D | ResNet50 | full finetune | 24,6M | 0.9484 | **0.9606** | **0.9597** | 284s / 10 ep | 1785 MB |
| 3C | ResNet50 | LoRA | 1,57M | 0.9543 | 0.9549 | 0.9536 | 1037s / 29 ep | 630 MB |
| 3B | ResNet50 | frozen | 1,05M | 0.9341 | 0.9294 | 0.9227 | 163s / 13 ep | 851 MB |
| 3B | EfficientNet-B0 | frozen | 662K | 0.9122 | 0.9201 | 0.9132 | 132s / 10 ep | 472 MB |
| 3A | VGG16 | frozen | 2,10M | 0.8617 | 0.8380 | 0.8330 | 138s / 9 ep | 2910 MB |

> **Lịch sử thay đổi thiết kế.** Ban đầu 3A là MobileNetV2 và 3B/3C/3D có thêm ViT-B/16. Cả hai
> đã bị bỏ: 3A đổi sang VGG16, ViT-B/16 bị gỡ khỏi toàn bộ nghiên cứu (3C/3D chỉ còn ResNet50).
> Kết quả cũ được giữ nguyên trong `results/_removed/` (không nằm trong `summary.csv`/`REPORT.md`):
> MobileNetV2 frozen test 0.8750; ViT-B/16 frozen 0.9317, LoRA 0.9537, full finetune 0.9468.

### Nhận xét

1. **VGG16 frozen là baseline yếu nhất: 0.838**, thấp hơn ResNet50 frozen 9 điểm và thấp hơn cả
   MobileNetV2 frozen cũ (0.875) dù trainable gấp ~3 lần (2,1M) và tổng 136M tham số. Nó overfit
   rất sớm: epoch tốt nhất là epoch 3, đến epoch 9 train acc đạt 0.978 trong khi val chỉ 0.853.
   Đặc trưng fc7 4096-d của VGG16 làm head 512 nút học thuộc train nhanh nhưng không tổng quát hoá
   tốt bằng đặc trưng ResNet50/EfficientNet. Điểm yếu nằm ở `Other` (precision 0.59) và `PET`
   (precision 0.68). VGG16 cũng tốn peak GPU nhiều nhất (2910 MB) dù đóng băng.
2. **Đổi backbone là đòn bẩy rẻ nhất.** Giữ nguyên head và công thức train, chỉ thay VGG16 (3A)
   bằng ResNet50/EfficientNet-B0 (3B) đã tăng test accuracy từ 0.838 lên 0.92–0.93.
3. **frozen → LoRA → full finetune tăng dần đúng như kỳ vọng** (ResNet50: 0.929 → 0.955 → 0.961).
   Cho backbone thích nghi với domain rác VN đáng giá ~2,5–3 điểm so với dùng nguyên đặc trưng ImageNet.
4. **LoRA tiết kiệm bộ nhớ và tham số, nhưng KHÔNG tiết kiệm thời gian.** 3C đạt 0.9549 với 1,57M
   tham số trainable (6,4% của 3D) và peak GPU thấp hơn 2,8 lần (630 MB vs 1785 MB) — gần bằng
   full finetune. Nhưng lại train lâu hơn 3,7 lần (1037s vs 284s) vì gradient vẫn phải chảy ngược
   qua toàn bộ backbone y như full finetune, cộng thêm hội tụ chậm hơn (29 epoch vs 10).
   LoRA có lợi khi thiếu **VRAM**, không phải khi thiếu **thời gian**.
5. **Lớp khó nhất là nhóm nhựa/xốp.** Ở model tốt nhất (3D ResNet50), `PET` có precision tuyệt đối
   1.000 nhưng recall thấp nhất 0.902 — bỏ sót 10 ảnh, chủ yếu đoán nhầm sang `Alu` (4 ảnh, đều là
   vỏ chai/lon sáng bóng) và `Plastic_cup` (3 ảnh). Chiều ngược lại, `Plastic_cup` bị nhầm thành
   `Foam_box` 6 ảnh. Muốn cải thiện tiếp nên tập trung vào cụm `PET ↔ Plastic_cup ↔ Foam_box`.
6. **Chênh lệch nhỏ nằm trong khoảng nhiễu.** 3D hơn 3C 0.57 điểm ≈ 5 ảnh trên 864 ảnh test, và
   cùng cấu hình chạy lại có thể lệch ±1–2% (xem mục tái lập bên dưới). Chênh lệch lớn (VGG16 so
   với ResNet50, frozen so với LoRA/finetune) thì đủ lớn để tin.

## Chi tiết kỹ thuật đáng lưu ý

- **Mixed precision bfloat16** — RTX 5070 hỗ trợ bf16 nên không cần `GradScaler`.
- **Backbone đóng băng chạy trong `torch.no_grad()`** (biến thể 3A/3B) để bỏ hẳn graph → nhanh
  hơn đáng kể. Với 3C thì *không* dùng `no_grad`, vì gradient phải chảy qua backbone để tới adapter.
- **BatchNorm bị giữ ở `eval()`** khi backbone đóng băng (3A/3B) và khi dùng LoRA (3C). Nếu không,
  running statistics vẫn âm thầm trôi theo từng epoch và đặc trưng "đóng băng" thực ra không còn đóng băng.
- **LoRA đặt ở đâu**: `nn.MultiheadAttention` của PyTorch truyền `in_proj_weight` và
  `out_proj.weight` trực tiếp như tensor vào `F.multi_head_attention_forward` chứ không gọi
  submodule, nên bọc adapter vào đó sẽ bị bỏ qua hoặc lỗi (liên quan nếu thêm lại ViT: adapter
  phải đặt vào các `Linear` trong MLP của transformer block). Hiện chỉ còn CNN nên adapter chèn vào
  các `Conv2d` ở những stage cuối (xem `lora_target_prefixes` trong [`backbones.py`](backbones.py));
  `LoRALinear` vẫn còn trong `lora.py` nhưng không được dùng. VGG16 chỉ chạy frozen nên
  `lora_target_prefixes=()`.
- **Learning rate**: head luôn `1e-3`. Backbone dùng LR nhỏ hơn — LoRA `1e-3` (adapter khởi tạo 0
  nên chịu được LR lớn), full finetune `1e-4` để không phá đặc trưng pretrained trong vài bước đầu
  (trước đây ViT dùng `1e-5`, nhánh này đã bỏ cùng ViT).
- **Augmentation on-the-fly nhẹ** (flip + color jitter nhỏ) được thêm lên ảnh đã tiền xử lý, vì
  augmentation trong `src/preprocessing.py` đã bị "đóng băng" một lần khi ghi ra đĩa — train 30
  epoch trên đúng một bản augmentation sẽ overfit nhanh hơn.
- **Dữ liệu khá cân bằng** (1.135–1.322 ảnh/lớp ở train gốc) nên không dùng class weights;
  macro F1 vẫn được báo cáo để phát hiện lớp bị bỏ rơi.
- **Về tính tái lập**: seed được cố định (`set_seed(42)`), nhưng `torch.backends.cudnn.benchmark
  = True` và các kernel GPU không tất định khiến thứ tự cộng dồn số thực thay đổi giữa các lần
  chạy. Qua 30 epoch, sai khác nhỏ đó tích luỹ và làm early stopping dừng ở epoch khác nhau →
  **chênh lệch khoảng ±1–2% accuracy giữa hai lần chạy cùng cấu hình**. Số trong `REPORT.md` là
  của một lần chạy cụ thể, không phải trung bình nhiều seed. Muốn tất định hoàn toàn thì đặt
  `cudnn.benchmark = False` + `torch.use_deterministic_algorithms(True)` (đổi lại chậm hơn đáng kể).
