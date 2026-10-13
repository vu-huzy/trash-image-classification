# Model 2: CNN architectures

Five models implemented in PyTorch for the 9-class VN Trash dataset.
All models accept RGB batches shaped `(N, 3, 224, 224)` and return class logits
shaped `(N, 9)`. Training uses `CrossEntropyLoss`, so the models intentionally
do not apply softmax inside.

| Group | Basic model | Improved model |
| --- | --- | --- |
| CNN | `cnn_sequential` | `cnn_parallel` |
| VGG | `vgg_sequential` | `vgg_parallel` |
| ResNet | `resnet` | — |

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
  py/metrics.py  calculate_metrics(), the only .py file the notebooks import
  checkpoints/   one final weight file per model, older runs in archive/
  results/       json / csv / logs written by the old train.py (no longer produced)
  Predict.py     load a checkpoint and check it on data/.../test
```

## Notebooks

`notebook/*.ipynb` (`cnn_sequential`, `cnn_parallel`, `vgg_sequential`,
`vgg_parallel`, `resnet`) are end-to-end pipelines in the same layout as Model 1:
the architecture is defined inside the notebook; the only import from the project
is `calculate_metrics` from `py/metrics.py`.

- **Paths** are relative to the notebook's own folder (open the notebook and
  press Run, no setup): data in `../../../data/VN_trash_classification_preprocessing`,
  weights in `../checkpoints`, metrics in `../py`.
- **Preprocessing**: train uses Resize(224, 224), RandomHorizontalFlip, ColorJitter
  (brightness/contrast/saturation 0.1), ToTensor and ImageNet Normalize. Validation
  and test use Resize, ToTensor and the same Normalize.
- **Training**: up to 30 epochs, batch 32 (`resnet`: batch 24, because batch 32 ran out of GPU memory
  and slowed to ~17 min/epoch), AdamW + cosine schedule, class-weighted
  loss. Early stopping monitors validation accuracy with `PATIENCE = 7`. The weights with the best validation accuracy are kept in RAM and loaded
  back into the model after the loop.
- **Test**: `calculate_metrics` prints accuracy, macro precision / recall / F1
  and weighted F1; then `classification_report` prints precision, recall, F1 and
  support for every class. There is no confusion matrix.
- **Save**: the last cell is
  `torch.save(model.state_dict(), CHECKPOINT_DIR / '<name>.pt')`.

The notebook is called `resnet`, not `resnet18`, because it is not the standard
ResNet-18: the stem is a 3×3 stride-1 conv without max-pool (standard: 7×7
stride-2 + max-pool) and the head is Dropout-Linear(512, 128)-ReLU-Dropout-Linear
(standard: one Linear). Only the four-stage × two-BasicBlock body is the same.

## Checkpoints

`checkpoints/<model>.pt` is what `Predict.py` loads. Running a notebook
overwrites its file with that run's best-validation weights; the files that were
there before (and every other older run) are in `checkpoints/archive/`, nothing
was deleted.

The first four rows were rerun on 2026-10-13 with the current notebooks (ImageNet normalize,
HFlip + ColorJitter, early stopping with `PATIENCE = 7`; RTX 5070, 8 DataLoader workers). The
`resnet` row is the older 2026-10-11 run (batch 32, no early stopping) because its rerun was
stopped; the notebook is now set to batch 24 and has not been rerun. The 2026-10-11 files that
the rerun overwrote were kept in a backup folder outside the repo (`backup_checkpoints_20261013`).
Per-class precision / recall / F1 / support for each model is in
[../../RESULTS.md](../../RESULTS.md).

| File | Val acc (best epoch) | Test acc | Macro F1 | Epoch | Time | Old `train.py` pick (test acc) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cnn_sequential.pt` | 69.38% | 65.97% | 0.6526 | 30 / 30 | 16.1 min | 70.49% |
| `cnn_parallel.pt` | 70.74% | 68.17% | 0.6754 | 30 / 30 | 17.8 min | 69.21% |
| `vgg_sequential.pt` | 79.70% | 72.34% | 0.7109 | 25 / 30 | 29.1 min | 75.81% |
| `vgg_parallel.pt` | 82.55% | **76.04%** | **0.7477** | 28 / 30 | 37.4 min | 76.97% |
| `resnet.pt` (old run) | 75.07% | 69.33% | 0.6813 | 30 / 30 | 90.7 min | 70.14% |

- The best epoch of the four rerun models is 30, 30, 25 and 28 of 30 and early stopping
  never triggered, so none of them stopped improving before the 30-epoch limit; whether
  more epochs would help was not tested.
- The notebook runs are 0.8–4.5 points of test accuracy below the old `train.py`
  picks. The two pipelines differ (the old script used CUDA bfloat16 autocast and
  its own run selection), so the gap is not attributed to a single cause.
- `vgg_parallel` has the best score and a file less than half the size of
  `vgg_sequential` (20.6 MB vs 56.5 MB), but it is a single run per model, so the
  difference between the two VGG variants is not established.
- The class with the lowest F1 is `Foam_box` for `cnn_sequential`, `cnn_parallel` and
  `vgg_parallel` (recall 0.54, 0.51 and 0.46), `Other` for `vgg_sequential`, and `PET` for
  the old `resnet` run.

## Check a model

Set `MODEL_NAME` at the top of `Predict.py`, then:

```powershell
.venv/Scripts/python.exe src/model2/Predict.py
```

It loads `checkpoints/<MODEL_NAME>.pt` (a plain `state_dict` saved by the notebook), runs the
whole test split and prints the accuracy and the per-class `classification_report`
(precision / recall / F1, macro avg, weighted avg). The files in `checkpoints/archive/` use an
older `{'model': state_dict, ...}` format and are not loaded by `Predict.py`.
