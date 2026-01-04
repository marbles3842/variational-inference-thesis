import os
import math
import yaml
import jax
import matplotlib.pyplot as plt
from typing import Optional


_DIFFUSION_TRAIN_CONFIG_FILE = "diffusion_train_config.yaml"


def load_diffusion_ivon_config(filename: str = _DIFFUSION_TRAIN_CONFIG_FILE):
    config_path = os.path.join(os.path.dirname(__file__), filename)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        return config["ivon"]


def save_samples(samples, filename: str, entropy: Optional[jax.Array] = None):
    n_samples = len(samples)

    n_cols = math.ceil(math.sqrt(n_samples))
    n_rows = math.ceil(n_samples / n_cols)

    fig_size = (n_cols * 1.25, n_rows * 1.25)

    _, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_samples:
            ax.imshow(samples[i].squeeze(), cmap="gray")
            if entropy is not None:
                ax.set_title(f"H={entropy[i]:.3f}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
