"""Registry for the ten standalone model files in this directory."""

from torch import nn

from alexnet_parallel import AlexNetParallel
from alexnet_sequential import AlexNetSequential
from cnn_kernel3_sequential import CNNKernel3Sequential
from cnn_parallel import CNNParallel
from resnet18 import ResNet18
from resnet50_bottleneck import ResNet50Bottleneck
from vgg16_parallel import VGG16Parallel
from vgg16_sequential import VGG16Sequential


def build_models(num_classes: int = 9) -> dict[str, nn.Module]:
    """Instantiate all ten model variants, keyed by their CLI names."""
    return {
        "cnn_kernel3_sequential": CNNKernel3Sequential(num_classes),
        "cnn_parallel": CNNParallel(num_classes),
        "alexnet_sequential": AlexNetSequential(num_classes),
        "alexnet_parallel": AlexNetParallel(num_classes),
        "vgg16_sequential": VGG16Sequential(num_classes),
        "vgg16_parallel": VGG16Parallel(num_classes),
        "resnet18": ResNet18(num_classes),
        "resnet50_bottleneck": ResNet50Bottleneck(num_classes),
    }
