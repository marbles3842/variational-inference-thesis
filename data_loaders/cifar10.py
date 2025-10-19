import numpy as np
import grain.python as grain
from datasets import load_dataset


class CIFAR10Info:
    num_classes = 10
    shape = (1, 32, 32, 3)
    counts = {"train": 50000, "test": 10000}
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)


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
        img = (img - np.array(CIFAR10Info.mean)) / np.array(CIFAR10Info.std)
        label = element["label"]
        return {"image": img, "label": label}


def get_cifar10_train_val_loaders(
    train_batch_size: int,
    val_batch_size: int,
    num_epochs: int,
    seed: int,
    train_worker_count: int = 4,
    val_worker_count: int = 1,
    val_split: float = 0.1,
):
    """Returns CIFAR-10 train and validation data loaders.

    If val_split is 0, returns (train_loader, None) with all training data.
    """

    ds = load_dataset("cifar10")
    ds = ds.with_format("numpy")

    if val_split > 0:
        split_dataset = ds["train"].train_test_split(test_size=val_split, seed=seed)
        train_ds = split_dataset["train"]
        val_ds = split_dataset["test"]
    else:
        train_ds = ds["train"]
        val_ds = None

    train_loader = grain.load(
        train_ds,
        num_epochs=num_epochs,
        shuffle=True,
        seed=seed,
        drop_remainder=True,
        batch_size=train_batch_size,
        worker_count=train_worker_count,
        transformations=[_Augment(), _Normalize()],
    )

    val_loader = None
    if val_ds is not None:
        val_loader = grain.load(
            val_ds,
            shuffle=False,
            num_epochs=1,
            batch_size=val_batch_size,
            worker_count=val_worker_count,
            transformations=[_Normalize()],
        )

    return train_loader, val_loader


def get_cifar10_test_loader(batch_size: int, worker_count: int = 4):
    """Returns CIFAR-10 test data loader."""

    ds = load_dataset("cifar10")
    ds = ds.with_format("numpy")

    test_ds = ds["test"]
    test_loader = grain.load(
        test_ds,
        shuffle=False,
        num_epochs=1,
        batch_size=batch_size,
        worker_count=worker_count,
        transformations=[_Normalize()],
    )

    return test_loader
