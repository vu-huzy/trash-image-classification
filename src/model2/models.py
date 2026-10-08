"""Registry for the fourteen standalone model variants in this directory."""

from collections.abc import Callable

from torch import nn

from alexnet.alexnet_parallel import AlexNetAdvanced
from alexnet.alexnet_sequential import AlexNetBasic
from cnn.cnn_kernel3_sequential import CNNKernel3Sequential
from cnn.cnn_parallel import CNNParallel
from densenet.densenet_advanced import DenseNetAdvanced
from densenet.densenet_basic import DenseNetBasic
from googlenet.googlenet_advanced import GoogLeNetAdvanced
from googlenet.googlenet_basic import GoogLeNetBasic
from nin.nin_advanced import NINAdvanced
from nin.nin_basic import NINBasic
from resnet.resnet18 import ResNet18
from resnet.resnet50_bottleneck import ResNet50Bottleneck
from vgg.vgg16_parallel import VGG16Advanced
from vgg.vgg16_sequential import VGG16Basic


MODEL_FACTORIES: dict[str, Callable[[int], nn.Module]] = {
    "cnn_kernel3_sequential": CNNKernel3Sequential,
    "cnn_parallel": CNNParallel,
    "alexnet_basic": AlexNetBasic,
    "alexnet_advanced": AlexNetAdvanced,
    "nin_basic": NINBasic,
    "nin_advanced": NINAdvanced,
    "googlenet_basic": GoogLeNetBasic,
    "googlenet_advanced": GoogLeNetAdvanced,
    "densenet_basic": DenseNetBasic,
    "densenet_advanced": DenseNetAdvanced,
    "vgg16_basic": VGG16Basic,
    "vgg16_advanced": VGG16Advanced,
    "resnet18": ResNet18,
    "resnet50_bottleneck": ResNet50Bottleneck,
}
MODEL_NAMES = tuple(MODEL_FACTORIES)
MODEL_ALIASES = {
    "alexnet_sequential": "alexnet_basic",
    "alexnet_parallel": "alexnet_advanced",
    "vgg16_sequential": "vgg16_basic",
    "vgg16_parallel": "vgg16_advanced",
}


def build_model(name: str, num_classes: int = 9) -> nn.Module:
    """Instantiate only the requested architecture."""
    name = MODEL_ALIASES.get(name, name)
    try:
        model_factory = MODEL_FACTORIES[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}"
        ) from error
    return model_factory(num_classes)


def build_models(num_classes: int = 9) -> dict[str, nn.Module]:
    """Instantiate every architecture, keyed by its CLI name."""
    return {name: build_model(name, num_classes) for name in MODEL_NAMES}
