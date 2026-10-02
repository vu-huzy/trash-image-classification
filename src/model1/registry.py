"""Ánh xạ trường ``architecture`` trong JSON sang module định nghĩa model tương ứng."""

from importlib import import_module
from types import ModuleType


MODEL_MODULES = {
    "m1_01": "model1.models.m1_01",
    "m1_02": "model1.models.m1_02",
    "m1_03": "model1.models.m1_03",
    "m1_04": "model1.models.m1_04",
}


def get_model_module(architecture: str) -> ModuleType:
    """Trả về module có ``build_model`` và ``run`` cho kiến trúc được cấu hình."""
    try:
        return import_module(MODEL_MODULES[architecture])
    except KeyError as error:
        available = ", ".join(MODEL_MODULES)
        raise ValueError(f"Unknown Model 1 architecture: {architecture}. Available: {available}") from error
