from typing import Any
from flax.training import train_state
from flax.core import FrozenDict

from .metrics import Metrics


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict[str, Any]
    metrics: Metrics


def create_train_state(model, rng, x0, optimizer):
    """Create training state with model parameters, optimizer, and batch stats.

    Args:
        model: Flax model
        rng: Random key for initialization
        x0: Sample input for shape inference
        optimizer: Optax optimizer

    Returns:
        TrainState with initialized parameters and empty metrics
    """
    variables = model.init(rng, x0)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        metrics=Metrics.empty(),
        batch_stats=batch_stats,
    )
