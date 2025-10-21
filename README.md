# Variational Inference for Bayesian Neural Networks

The main focus of this thesis is variational inference for Bayesian neural networks. Bayesian methods naturally allow to quantify the uncertainty, but usually they are difficult to implement efficiently especially for larger datasets or complex models.

[![Python](https://img.shields.io/badge/python-3.12.11-blue.svg)](https://docs.python.org/3.12/)
[![JAX](https://img.shields.io/badge/JAX-CUDA12-orange.svg)](https://jax.readthedocs.io/)
[![Flax](https://img.shields.io/badge/Flax-Neural%20Networks-green.svg)](https://flax-linen.readthedocs.io/en/latest/)
[![Optax](https://img.shields.io/badge/Optax-Optimization-red.svg)](https://optax.readthedocs.io/)
[![PyYAML](https://img.shields.io/badge/PyYAML-Config-yellow.svg)](https://pyyaml.org/wiki/PyYAMLDocumentation)
[![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243.svg)](https://numpy.org/doc/)
[![Datasets](https://img.shields.io/badge/Datasets-Data%20Loading-lightblue.svg)](https://huggingface.co/docs/datasets/)
[![Grain](https://img.shields.io/badge/Grain-Data%20Pipelines-brown.svg)](https://google-grain.readthedocs.io/en/latest/)

## Table of Contents

- [Project Structure](#project-structure)
- [Baseline Models](#baseline-models)
- [Running Baseline Experiments](#running-baseline-experiments)
  - [Training](#training)
    - [Training with SGD](#training-with-sgd)
    - [Training with IVON](#training-with-ivon)
  - [Testing](#testing)
    - [Testing SGD or IVON@mean](#testing-sgd-or-ivonmean)
    - [Testing IVON with Sampling](#testing-ivon-with-sampling)

## Project Structure

```
├── baseline/         # Baseline implementations from the paper
├── checkpoints/      # Model checkpoints and saved states
├── core/             # Base classes and methods (e.g IVON)
├── data/             # Raw datasets and processed data
├── data_loaders/     # Data loading utilities and pipelines
├── logger/           # Logging configuration and utilities
├── models/           # Neural network model definitions
├── out/              # Output logs and results
└── trainer/          # Training utils
```
*Note: The `checkpoints/`, `data/`, and `out/` directories are included in `.gitignore` and will be created during runtime.*

## Baseline Models

The following models are used for baseline reproduction on CIFAR-10:

- **[ResNet](models/resnet.py)** - Residual Networks with variants ResNet18 and ResNet20. Based on [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al., 2015).

- **[PreResNet](models/pre_resnet.py)** - Pre-activation ResNet with PreResNet110 variant. Based on [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (He et al., 2016).

- **[DenseNet](models/densenet.py)** - Densely Connected Convolutional Networks with DenseNet121 variant. Based on [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (Huang et al., 2017).

- **[Filter Response Normalization](models/filter_response_norm.py)** - Normalization layer used in the baseline models. Based on [Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks](https://arxiv.org/abs/1911.09737) (Singh & Krishnan, 2020).

All models are implemented in Flax and use Filter Response Normalization by default.

## Running Baseline Experiments

### Training

Training hyperparameters are configured in [baseline/train_cifar10_config.yaml](baseline/train_cifar10_config.yaml).

Training scripts will generate a CSV file with metrics (loss, accuracy, etc.) for each epoch.

#### Training with SGD

Use [baseline/train_sgd.py](baseline/train_sgd.py) to train models with standard SGD optimizer:

```bash
python -m baseline.train_sgd --seed 0 --model-name resnet18
python -m baseline.train_sgd --seed 1 --model-name preresnet110 --val-split 0.1
```

#### Training with IVON

Use [baseline/train_ivon.py](baseline/train_ivon.py) to train models with the IVON optimizer:

```bash
python -m baseline.train_ivon \
  --seed 0 \
  --model-name resnet18 \
  --hessian 0.5 \
  --logdir out/ivon \
  --checkpoints-dir checkpoints/ivon/resnet18

python -m baseline.train_ivon \
  --seed 1 \
  --model-name preresnet110 \
  --hessian 0.5 \
  --val-samples 8 \
  --val-split 0.1 \
  --logdir out/ivon \
  --checkpoints-dir checkpoints/ivon/preresnet110
```

### Testing

Testing scripts will generate a report with results averaged over 5 random seeds.

#### Testing SGD or IVON@mean

Use [baseline/test.py](baseline/test.py) to evaluate models trained with SGD or IVON (using mean parameters):

```bash
python -m baseline.test \
  --model-name resnet18 \
  --optimizer sgd \
  --checkpoints-dir checkpoints/sgd/resnet18

python -m baseline.test \
  --model-name resnet18 \
  --optimizer ivon-mean \
  --checkpoints-dir checkpoints/ivon/resnet18
```

#### Testing IVON with Sampling

Use [baseline/test_ivon.py](baseline/test_ivon.py) to evaluate IVON models with uncertainty quantification via sampling:

```bash
python -m baseline.test_ivon \
  --model-name resnet18 \
  --checkpoints-dir checkpoints/ivon/resnet18 \
  --n-samples 64 \
  --hessian 0.5 \
  --logdir out/ivon
```