# Model 2: CNN architectures

Five models implemented in PyTorch for the 9-class VN Trash dataset.
All models accept RGB batches shaped `(N, 3, 224, 224)` and return class logits
shaped `(N, 9)`. The training script uses `CrossEntropyLoss`, so it intentionally
does not apply softmax inside the models.

| Group | Basic model | Improved model |
| --- | --- | --- |
| CNN | `cnn_sequential` | `cnn_parallel` |
| VGG | `vgg_sequential` | `vgg_parallel` |
| ResNet | `resnet` (registry key `resnet18`) | — |

The parallel CNN has three independent 3×3 convolution branches whose feature
maps are concatenated. VGG parallel variants use three parallel first-convolution
branches and then continue through their normal sequential stacks. The ResNet uses
basic residual blocks. The parallel variants use the same number of pooling stages
as their sequential counterparts so their comparisons are less biased by
feature-map resolution.

## Layout

```
src/model2/
  notebook/      5 self-contained notebooks: data -> model -> train -> test -> save
  py/            the same architectures as .py files, plus the training script
    cnn/ vgg/ resnet/   one file per architecture
    models.py           registry used by train.py and Predict.py
    train.py  metrics.py
  checkpoints/   one final weight file per model, older runs in archive/
  results/       json / csv / logs written by py/train.py
  Predict.py     load a checkpoint and check it on data/.../test
```

## Notebooks

`notebook/*.ipynb` (`cnn_sequential`, `cnn_parallel`, `vgg_sequential`,
`vgg_parallel`, `resnet`) are end-to-end pipelines in the same layout as Model 1:
the architecture is defined inside the notebook and nothing is imported from
`py/`. Layer names match the `.py` files, so weights are interchangeable. Training
keeps the weights with the best validation accuracy; the last cell saves them with
`torch.save(model.state_dict(), CHECKPOINT_DIR / '<name>.pt')` into `checkpoints/`.

The notebook is called `resnet`, not `resnet18`, because it is not the standard
ResNet-18: the stem is a 3×3 stride-1 conv without max-pool (standard: 7×7
stride-2 + max-pool) and the head is Dropout-Linear(512, 128)-ReLU-Dropout-Linear
(standard: one Linear). Only the four-stage × two-BasicBlock body is the same.
The `.py` version in `py/resnet/resnet18.py` keeps the old name.

## Checkpoints

`checkpoints/<model>.pt` is what `Predict.py` loads. Each file comes from one
training run of `py/train.py`; every other run was moved to `checkpoints/archive/`
(nothing was deleted).

| File | Source run | Val acc | Test acc |
| --- | --- | ---: | ---: |
| `cnn_sequential.pt` | `cnn_sequential_20261009_171644` (run 4) | 0.7395 | 0.7049 |
| `cnn_parallel.pt` | `cnn_parallel_20261009_180544` (run 1) | 0.7395 | 0.6921 |
| `vgg_sequential.pt` | `vgg_sequential_20261009_233635` (run 2) | 0.7976 | 0.7581 |
| `vgg_parallel.pt` | `vgg_parallel_20261010_002742` (run 1) | 0.8172 | 0.7697 |
| `resnet.pt` | `resnet18_20261011_010301` (run 3) | 0.7537 | 0.7014 |

The picked run is the best validation accuracy among the runs whose file size
equals the latest run of that model, i.e. the ones that most likely match the
current architecture. Two archived runs have a higher validation accuracy but a
different file size, so they were probably trained with an older version of the
architecture (not loaded, so not verified): `vgg_sequential_20261009_205648`
(val 0.8421, test 0.7928) and `resnet18_20261010_113058` (val 0.9347, test 0.9387).

## Check a model

Set `MODEL_NAME` at the top of `Predict.py`, then:

```powershell
.venv/Scripts/python.exe src/model2/Predict.py
```

It prints the true and predicted class of the first 10 test images, then accuracy,
macro precision / recall / F1, weighted F1 and the confusion matrix on the whole
test split. It reads both checkpoint formats: a plain `state_dict` (notebooks) and
`{'model': state_dict, ...}` (`py/train.py`).

## Train one model with the script

Prepare `data/VN_trash_classification_preprocessing/` first, with `train/`,
`val/`, and `test/` subdirectories organized as ImageFolder class folders.

```powershell
.venv/Scripts/python.exe src/model2/py/train.py --model cnn_sequential --epochs 40
```

The trainer automatically selects CUDA, MPS, or CPU and uses CUDA bfloat16 autocast
when available. Options include `--batch-size`, `--lr`, `--workers`, `--seed`,
`--device`, `--pretrained`, `--class-balanced/--no-class-balanced`, and
`--data-dir`. The default is 30 epochs, batch size 32, cosine learning-rate
decay, and early stopping based on validation accuracy. `--pretrained` uses
ImageNet weights for VGG sequential and ResNet18. `--model` takes the registry
names `cnn_sequential`, `cnn_parallel`, `vgg_sequential`, `vgg_parallel`, `resnet18`.
The best validation checkpoint is saved to `checkpoints/<model>_<timestamp>.pt`;
per-epoch history and final test metrics are written to `results/`.

## Files

| Model | File |
| --- | --- |
| CNN sequential | `py/cnn/cnn_sequential.py` |
| CNN parallel | `py/cnn/cnn_parallel.py` |
| VGG sequential | `py/vgg/vgg_sequential.py` |
| VGG parallel | `py/vgg/vgg_parallel.py` |
| ResNet | `py/resnet/resnet18.py` |

## Use a model in Python

```python
import sys
sys.path.insert(0, 'src/model2/py')
from models import build_models

model = build_models(num_classes=9)['resnet18']
logits = model(images)  # images: N x 3 x 224 x 224
```
