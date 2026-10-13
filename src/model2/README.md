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
- **Training**: up to 30 epochs, batch 32 (`resnet`: batch 16, because batch 32 ran out of GPU memory
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

The results below predate the preprocessing and early-stopping update.
All five `<model>.pt` files below come from the notebook run of 2026-10-11 (RTX 5070,
8 DataLoader workers). Per-class precision / recall / F1 / support for each model is
in [../../RESULTS.md](../../RESULTS.md).

| File | Val acc (best epoch) | Test acc | Macro F1 | Epoch | Time | Old `train.py` pick (test acc) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cnn_sequential.pt` | 68.25% | 64.12% | 0.6320 | 28 / 30 | 16.0 min | 70.49% |
| `cnn_parallel.pt` | 69.73% | 64.12% | 0.6339 | 27 / 30 | 18.2 min | 69.21% |
| `vgg_sequential.pt` | 80.71% | 73.38% | 0.7217 | 29 / 30 | 29.3 min | 75.81% |
| `vgg_parallel.pt` | 82.08% | **76.04%** | **0.7487** | 28 / 30 | 45.3 min | 76.97% |
| `resnet.pt` | 75.07% | 69.33% | 0.6813 | 30 / 30 | 90.7 min | 70.14% |

- The best epoch of every model is 27–30 of 30, so none of them stopped improving
  before the 30-epoch limit; whether more epochs would help was not tested.
- The notebook runs are 0.8–6.4 points of test accuracy below the old `train.py`
  picks. The two pipelines differ (the old script used CUDA bfloat16 autocast and
  its own run selection), so the gap is not attributed to a single cause.
- `vgg_parallel` has the best score and a file less than half the size of
  `vgg_sequential` (20.6 MB vs 56.5 MB), but it is a single run per model, so the
  difference between the two VGG variants is not established.
- The class with the lowest F1 is `Foam_box` for `vgg_sequential` and `vgg_parallel`
  (recall only 0.47 and 0.38) and `PET` for the other three models.

## Check a model

Set `MODEL_NAME` at the top of `Predict.py`, then:

```powershell
.venv/Scripts/python.exe src/model2/Predict.py
```

It prints the true and predicted class of the first 10 test images, then accuracy,
macro precision / recall / F1, weighted F1 and the per-class `classification_report`
on the whole test split. It reads both checkpoint formats: a plain `state_dict`
(notebooks) and `{'model': state_dict, ...}` (files in `checkpoints/archive/`).
