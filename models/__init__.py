from .resnet import ResNet20, ResNet18
from .pre_resnet import PreResNet110
from .densenet import DenseNet121


def get_supported_models_names():
    """Returns the list of names of supported models"""

    return ["resnet20", "resnet18", "preresnet110", "densenet121"]


def get_cifar10_model(model_name, num_classes):
    """Factory function to return model based on name"""

    models = {
        "resnet20": lambda: ResNet20(num_classes=num_classes),
        "resnet18": lambda: ResNet18(num_classes=num_classes),
        "preresnet110": lambda: PreResNet110(num_classes=num_classes),
        "densenet121": lambda: DenseNet121(num_classes=num_classes),
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )

    return models[model_name]()
