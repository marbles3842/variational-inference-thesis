import grain.python as grain
from datasets import load_dataset
import numpy as np

MNIST_LENGTH = 60000
MNIST_IMAGE_SHAPE = (32, 32, 1)  # additional padding is used


class _Normalize(grain.MapTransform):
    def map(self, element):
        img = element.get("image", element.get("img"))
        img = img.astype(np.float32) / 255.0
        img = img + np.random.rand(*img.shape) / 255.0
        img = (img - 0.5) * 2.0
        return img.flatten()


class _Reshape(grain.MapTransform):
    def map(self, element):
        return element.reshape(28, 28, 1)


class _Pad(grain.MapTransform):
    def map(self, element):
        return np.pad(element, ((2, 2), (2, 2), (0, 0)))


def get_mnist_dataset(
    train_batch_size: int,
    num_epochs: int,
    seed: int,
    train_worker_count: int = 4,
):
    ds = load_dataset("mnist")
    ds = ds.with_format("numpy")["train"]

    train_loader = grain.load(
        ds,
        num_epochs=num_epochs,
        shuffle=True,
        seed=seed,
        drop_remainder=True,
        batch_size=train_batch_size,
        worker_count=train_worker_count,
        transformations=[_Normalize(), _Reshape(), _Pad()],
    )
    return train_loader
