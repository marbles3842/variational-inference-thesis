import jax
from clu import metrics
from flax.struct import dataclass
from flax.training.train_state import TrainState


@dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    hess_mean: metrics.Average.from_output("hess_mean")
    hess_min: metrics.Average.from_output("hess_min")


class TrainState(TrainState):
    metrics: Metrics


def create_train_state(
    model, rng: jax.random.PRNGKey, x0: jax.Array, t0: jax.Array, optimizer
):
    variables = model.init(rng, x0, t0)
    params = variables["params"]
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer, metrics=Metrics.empty()
    )
