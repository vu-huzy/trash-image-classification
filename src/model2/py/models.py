"""Registry for the model implementations in :mod:`model2`."""

from torch import nn

from cnn.cnn_sequential import CNNSequential
from cnn.cnn_parallel import CNNParallel
from resnet.resnet18 import ResNet18
from vgg.vgg_parallel import VGGParallel
from vgg.vgg_sequential import VGGSequential


def build_models(
    num_classes: int = 9,
    pretrained: bool = False,
) -> dict[str, nn.Module]:
    """Instantiate all available model variants, keyed by their CLI names."""
    return {
        "cnn_sequential": CNNSequential(num_classes),
        "cnn_parallel": CNNParallel(num_classes),
        "vgg_sequential": VGGSequential(num_classes, pretrained),
        "vgg_parallel": VGGParallel(num_classes),
        "resnet18": ResNet18(num_classes, pretrained),
    }
