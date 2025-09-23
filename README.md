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