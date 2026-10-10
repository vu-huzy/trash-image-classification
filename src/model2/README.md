# Model 2: CNN architectures

Five models implemented in PyTorch for the 9-class VN Trash dataset.
All models accept RGB batches shaped `(N, 3, 224, 224)` and return class logits
shaped `(N, 9)`. The training script uses `CrossEntropyLoss`, so it intentionally
does not apply softmax inside the models.

| Group | Basic model | Improved model |
| --- | --- | --- |
| CNN | `cnn_sequential` | `cnn_parallel` |
| VGG | `vgg_sequential` | `vgg_parallel` |
| ResNet | `resnet18` | — |

Each architecture has its own Python file. The parallel CNN has three
independent 3×3 convolution branches whose feature maps are concatenated.
VGG parallel variants use three parallel first-convolution branches
and then continue through their normal sequential stacks. ResNet-18 uses basic
residual blocks.

`models.py` is a small registry used by the training script; shared ResNet block
definitions live in `resnet_blocks.py`. The parallel variants use the same number
of pooling stages as their sequential counterparts so their comparisons are less
biased by feature-map resolution.

## Train one model

Prepare `data/VN_trash_classification_preprocessing/` first, with `train/`,
`val/`, and `test/` subdirectories organized as ImageFolder class folders.

```powershell
.venv/Scripts/python.exe -m model2.train --model cnn_sequential --epochs 40
```

Run the command from the `src/` directory, or set `PYTHONPATH=src`. The trainer
automatically selects CUDA, MPS, or CPU and uses CUDA bfloat16 autocast when
available. Options include `--batch-size`, `--lr`, `--workers`, `--seed`,
`--device`, `--pretrained`, `--class-balanced/--no-class-balanced`, and
`--data-dir`. The default is 30 epochs, batch size 32, cosine learning-rate
decay, and early stopping based on validation accuracy. `--pretrained` uses
ImageNet weights for VGG sequential and ResNet18.
The best validation checkpoint is saved to `src/model2/checkpoints/`; per-epoch
history and final test metrics are written to `src/model2/results/`.

## Files

| Model | File |
| --- | --- |
| CNN sequential | `cnn_sequential.py` |
| CNN parallel | `cnn_parallel.py` |
| VGG sequential | `vgg_sequential.py` |
| VGG parallel | `vgg_parallel.py` |
| ResNet-18 | `resnet18.py` |

## Use a model in Python

```python
from model2.resnet.resnet18 import ResNet18

model = ResNet18(num_classes=9)
logits = model(images)  # images: N x 3 x 224 x 224
```
