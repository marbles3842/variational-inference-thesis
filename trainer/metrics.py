import jax
import jax.numpy as jnp
import optax

from jax.nn import softmax
from clu import metrics
from flax.struct import dataclass


@dataclass
class BrierScore(metrics.Average):
    @classmethod
    def from_model_output(
        cls, *, logits: jnp.ndarray, labels: jnp.ndarray, **kwargs
    ) -> metrics.Average:

        probabilities = softmax(logits, axis=-1)
        targets = jnp.eye(logits.shape[-1])[labels]
        brier = jnp.sum((probabilities - targets) ** 2, axis=-1).mean()
        return super(BrierScore, cls).from_model_output(values=brier)


@dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")
    brier_score: BrierScore


def cross_entropy_loss(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return jnp.mean(loss)


@jax.jit
def compute_metrics(*, state, batch):
    variables = {"params": state.params}
    logits = state.apply_fn(variables, batch["image"])
    loss = cross_entropy_loss(logits, batch["label"])
    metric_update = state.metrics.single_from_model_output(
        logits=logits, labels=batch["label"], loss=loss
    )
    metrics = state.metrics.merge(metric_update)
    return state.replace(metrics=metrics)
