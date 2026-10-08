# Model 2: CNN architectures

Fourteen model variants implemented from scratch in PyTorch for the 9-class VN Trash dataset.
All models accept RGB batches shaped `(N, 3, 224, 224)` and return class logits
shaped `(N, 9)`. The training script uses `CrossEntropyLoss`, so it intentionally
does not apply softmax inside the models.

| Group | Basic model | Advanced model |
| --- | --- | --- |
| CNN | `cnn_kernel3_sequential` | `cnn_parallel` |
| AlexNet | `alexnet_basic` | `alexnet_advanced` |
| Network-in-Network | `nin_basic` | `nin_advanced` |
| GoogLeNet | `googlenet_basic` | `googlenet_advanced` |
| DenseNet | `densenet_basic` | `densenet_advanced` |
| VGG-16 | `vgg16_basic` | `vgg16_advanced` |
| ResNet | `resnet18` | `resnet50_bottleneck` |

Each architecture has its own Python file. The parallel CNN has three
independent 3×3 convolution branches whose feature maps are concatenated.
AlexNet and VGG basic variants use a sequential feature extractor. Their
advanced variants add three parallel, multi-scale input branches before the
sequential stack; VGG-16 keeps the five standard convolution block depths
(2/2/3/3/3) in both variants. NiN uses stacked
1×1 MLPConv layers; GoogLeNet concatenates multi-scale Inception branches; and
DenseNet concatenates each layer's feature maps, with the advanced version
using bottlenecks and compressed transitions. ResNet-18 uses basic residual
blocks; ResNet-50 uses the standard 3/4/6/3 bottleneck stages.

`models.py` is a small registry used by the training script; shared ResNet block
definitions live in `resnet_blocks.py`.

The trainer delegates the standard train/evaluation epoch, seed setup,
checkpoint I/O, numerical metrics, and generic report plots to `src/common/`.
Model 2 keeps its own CLI, optimizer/scheduler policy, and ImageFolder transforms.
Only the selected architecture is instantiated for a run. The previous
`alexnet_sequential`, `alexnet_parallel`, `vgg16_sequential`, and
`vgg16_parallel` names remain accepted as aliases; the CLI lists the basic and
advanced names.

## Train one model

Prepare `data/VN_trash_classification_preprocessing/` first, with `train/`,
`val/`, and `test/` subdirectories organized as ImageFolder class folders.

```powershell
python run.py model2 --model cnn_kernel3_sequential --device cuda --epochs 20
```

Run all 14 registered variants for 20 epochs, sequentially, in PowerShell:

```powershell
$models = @(
  "cnn_kernel3_sequential", "cnn_parallel",
  "alexnet_basic", "alexnet_advanced",
  "nin_basic", "nin_advanced",
  "googlenet_basic", "googlenet_advanced",
  "densenet_basic", "densenet_advanced",
  "vgg16_basic", "vgg16_advanced",
  "resnet18", "resnet50_bottleneck"
)
foreach ($model in $models) {
  python run.py model2 --model $model --device cuda --epochs 20
  if ($LASTEXITCODE -ne 0) { throw "Training failed for $model" }
}
```

`--device auto` (the default) selects CUDA when available and otherwise CPU.
Use `--device cuda` to require a GPU; training stops with an explicit error if
CUDA is unavailable. Select a model name from the table. Options include
`--batch-size`, `--image-size`, `--lr`, `--workers`, `--seed`, and `--data-dir`
for a different dataset location. `--image-size` defaults to 224; reducing it
trades image detail for faster training. Add `--amp` to opt into CUDA bfloat16
mixed precision and `--channels-last` to use the CUDA channels-last memory format.
`--limit-batches N` runs a smoke test without replacing the regular checkpoint
or summary.

For a speed-focused full-dataset run on a CUDA GPU that supports bfloat16:

```powershell
$models = @(
  "cnn_parallel", "alexnet_advanced", "nin_advanced",
  "googlenet_advanced", "densenet_advanced",
  "vgg16_advanced", "resnet50_bottleneck"
)
$batchSizes = @{
  cnn_parallel = 256
  alexnet_advanced = 256
  nin_advanced = 256
  googlenet_advanced = 512
  densenet_advanced = 512
  vgg16_advanced = 256
  resnet50_bottleneck = 384
}
$imageSizes = @{
  cnn_parallel = 32
  alexnet_advanced = 32
  nin_advanced = 32
  googlenet_advanced = 32
  densenet_advanced = 32
  vgg16_advanced = 32
  resnet50_bottleneck = 24
}
foreach ($model in $models) {
  $batchSize = $batchSizes[$model]
  $imageSize = $imageSizes[$model]
  python run.py model2 --model $model --device cuda --amp `
    --channels-last --image-size $imageSize --batch-size $batchSize `
    --workers 0 --epochs 20
  if ($LASTEXITCODE -ne 0) { throw "Training failed for $model" }
}
```

On this workspace's RTX 5050, the following speed-focused configuration
completed a full train+validation epoch for every advanced model in under
20 seconds:

| Model | Image size | Batch size | Epoch seconds |
| --- | ---: | ---: |
| `cnn_parallel` | 32×32 | 256 | 17.9 |
| `alexnet_advanced` | 32×32 | 256 | 17.6 |
| `nin_advanced` | 32×32 | 256 | 18.1 |
| `googlenet_advanced` | 32×32 | 512 | 17.2 |
| `densenet_advanced` | 32×32 | 512 | 18.4 |
| `vgg16_advanced` | 32×32 | 256 | 18.5 |
| `resnet50_bottleneck` | 24×24 | 384 | 19.2 |

These timings used BF16 AMP, channels-last tensors, and `workers=0`; they vary
with GPU load and hardware. Each timing is one complete train+validation pass
in smoke mode (no regular checkpoint or summary was replaced); test evaluation
is not included in the epoch time. These low input resolutions (especially
24×24 for ResNet-50) trade away image detail and make cross-model comparisons
uneven; use a common 128×128 or default 224×224 input when accuracy or fair
comparison matters more than speed. Default input size and training behavior
are unchanged. On Windows systems with limited pagefile memory, `--workers 0`
avoids the memory cost of spawning several PyTorch data-loader processes.
The shared epoch engine accumulates loss/accuracy on the device and copies them
to CPU once per epoch, avoiding per-batch GPU synchronization.

The best validation-accuracy checkpoint is saved to
`src/model2/checkpoints/<model>.pt`. Every run writes a unique directory under
`src/model2/results/<model>/` with `metrics.json`, `history.csv`,
`train_log.txt`, `curves.png`, `confusion_matrix_test.png`, and
`classification_report_test.txt`. The latest full-run metrics are also
available at `src/model2/results/<model>.json`, with the run index in
`src/model2/results/summary.csv`.

## Files

| Model | File |
| --- | --- |
| CNN 3×3 sequential | `cnn_kernel3_sequential.py` |
| CNN parallel | `cnn_parallel.py` |
| AlexNet basic / advanced | `alexnet/alexnet_sequential.py`, `alexnet/alexnet_parallel.py` |
| NiN basic / advanced | `nin/nin_basic.py`, `nin/nin_advanced.py` |
| GoogLeNet basic / advanced | `googlenet/googlenet_basic.py`, `googlenet/googlenet_advanced.py` |
| DenseNet basic / advanced | `densenet/densenet_basic.py`, `densenet/densenet_advanced.py` |
| VGG-16 basic / advanced | `vgg/vgg16_sequential.py`, `vgg/vgg16_parallel.py` |
| ResNet-18 | `resnet18.py` |
| ResNet-50 bottleneck | `resnet50_bottleneck.py` |

## Use a model in Python

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src/model2").resolve()))
from resnet.resnet18 import ResNet18

# Input is a normalized RGB batch shaped (N, 3, 224, 224).
model = ResNet18(num_classes=9)
logits = model(images)  # images: N x 3 x 224 x 224
```
