import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def append_experiment(path: Path, row: dict):
    """Append one compact experiment row without overwriting past runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
