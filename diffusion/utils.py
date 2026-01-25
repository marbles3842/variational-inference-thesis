import os
import yaml


_DIFFUSION_TRAIN_CONFIG_FILE = "diffusion_train_config.yaml"


def load_diffusion_ivon_config(filename: str = _DIFFUSION_TRAIN_CONFIG_FILE):
    config_path = os.path.join(os.path.dirname(__file__), filename)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        return config["ivon"]
