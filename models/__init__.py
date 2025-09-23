from .resnet import ResNet20


def get_cifar10_model(model_name, num_classes):
    """Factory function to return model based on name"""

    models = {
        "resnet20": lambda: ResNet20(num_classes=num_classes),
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )

    return models[model_name]()
