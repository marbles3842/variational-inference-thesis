from typing import Any
import optax
from clu import metrics
from flax.training import train_state
from flax.core import FrozenDict
from flax import struct


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


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
    variables = model.init(rng, x0, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        metrics=Metrics.empty(),
        batch_stats=batch_stats,
    )


def create_eval_state(model, rng, x0, last_checkpoint_data):
    """Create evaluation state with model parameters and an identity optimizer.

    Args:
        model: Flax model
        rng: Random key for initialization
        x0: Sample input for shape inference,
        last_checkpoint_data: Data from the last checkpoint

    Returns:
        TrainState with initialized parameters and identity optimizer
    """
    model.init(rng, x0, train=False)
    batch_stats = last_checkpoint_data["batch_stats"]
    return TrainState.create(
        apply_fn=model.apply,
        params=last_checkpoint_data["params"],
        tx=optax.identity(),
        metrics=Metrics.empty(),
        batch_stats=batch_stats,
    )
