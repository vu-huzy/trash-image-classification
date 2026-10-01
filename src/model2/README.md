# Model 2: 10 CNN architectures

Ten models implemented from scratch in PyTorch for the 9-class VN Trash dataset.
All models accept RGB batches shaped `(N, 3, 224, 224)` and return class logits
shaped `(N, 9)`. The training script uses `CrossEntropyLoss`, so it intentionally
does not apply softmax inside the models.

| Group | Basic model | Improved model |
| --- | --- | --- |
| CNN | `cnn_kernel3_sequential` | `cnn_parallel` |
| AlexNet | `alexnet_sequential` | `alexnet_parallel` |
| VGG-16 | `vgg16_sequential` | `vgg16_parallel` |
| VGG-19 | `vgg19_sequential` | `vgg19_parallel` |
| ResNet | `resnet18` | `resnet50_bottleneck` |

Each architecture has its own Python file. The parallel CNN has three
independent 3×3 convolution branches whose feature maps are concatenated.
AlexNet and VGG parallel variants use three parallel first-convolution branches
and then continue through their normal sequential stacks. ResNet-18 uses basic
residual blocks; ResNet-50 uses the standard 3/4/6/3 bottleneck stages.

`models.py` is a small registry used by the training script; shared ResNet block
definitions live in `resnet_blocks.py`.

## Train one model

Prepare `data/VN_trash_classification_preprocessing/` first, with `train/`,
`val/`, and `test/` subdirectories organized as ImageFolder class folders.

```powershell
.venv/Scripts/python.exe "model 2/train.py" --model cnn_kernel3_sequential --epochs 20
```

Select any model name from the table. Options include `--batch-size`, `--lr`,
`--workers`, `--seed`, and `--data-dir` for a different dataset location.
The best validation checkpoint is saved to `model 2/checkpoints/`; per-epoch
history and final test metrics are written to `model 2/results/`.

## Files

| Model | File |
| --- | --- |
| CNN 3×3 sequential | `cnn_kernel3_sequential.py` |
| CNN parallel | `cnn_parallel.py` |
| AlexNet sequential | `alexnet_sequential.py` |
| AlexNet parallel | `alexnet_parallel.py` |
| VGG-16 sequential | `vgg16_sequential.py` |
| VGG-16 parallel | `vgg16_parallel.py` |
| VGG-19 sequential | `vgg19_sequential.py` |
| VGG-19 parallel | `vgg19_parallel.py` |
| ResNet-18 | `resnet18.py` |
| ResNet-50 bottleneck | `resnet50_bottleneck.py` |

## Use a model in Python

```python
from resnet18 import ResNet18

model = ResNet18(num_classes=9)
logits = model(images)  # images: N x 3 x 224 x 224
```
