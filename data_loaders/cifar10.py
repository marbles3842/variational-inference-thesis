import numpy as np
import grain.python as grain
from datasets import load_dataset

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class _Augment(grain.RandomMapTransform):
    def random_map(self, element, rng):
        img = element["img"]
        if rng.random() > 0.5:
            img = np.fliplr(img)
        padded = np.pad(img, ((4, 4), (4, 4), (0, 0)), mode="reflect")
        y = rng.integers(0, 8)
        x = rng.integers(0, 8)
        img = padded[y : y + 32, x : x + 32]
        return {"image": img, "label": element["label"]}


class _Normalize(grain.MapTransform):
    def map(self, element):
        img = element.get("image", element.get("img"))
        img = img / 255.0
        img = (img - np.array(_CIFAR10_MEAN)) / np.array(_CIFAR10_STD)
        label = element["label"]
        return {"image": img, "label": label}


def get_cifar10_train_val_loaders(
    train_batch_size: int,
    val_batch_size: int,
    num_epochs: int,
    seed: int,
    val_split: float = 0.1,
):
    """Returns CIFAR-10 train and validation data loaders."""

    ds = load_dataset("cifar10")
    ds = ds.with_format("numpy")

    split_dataset = ds["train"].train_test_split(test_size=val_split, seed=seed)
    train_ds = split_dataset["train"]
    val_ds = split_dataset["test"]

    return grain.load(
        train_ds,
        num_epochs=num_epochs,
        shuffle=True,
        seed=seed,
        drop_remainder=True,
        batch_size=train_batch_size,
        worker_count=3,
        transformations=[_Augment(), _Normalize()],
    ), grain.load(
        val_ds,
        shuffle=False,
        num_epochs=1,
        batch_size=val_batch_size,
        worker_count=1,
        transformations=[_Normalize()],
    )


def get_cifar10_test_loader(batch_size: int):
    """Returns CIFAR-10 test data loader."""

    ds = load_dataset("cifar10")
    ds = ds.with_format("numpy")

    test_ds = ds["test"]
    test_loader = grain.load(
        test_ds,
        shuffle=False,
        num_epochs=1,
        batch_size=batch_size,
        worker_count=1,
        transformations=[_Normalize()],
    )

    return test_loader
