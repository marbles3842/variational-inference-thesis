import jax
import jax.numpy as jnp
import optax

from jax import vmap
from jax.nn import softmax
from clu import metrics
from flax.struct import dataclass

_DEFAULT_NUM_OF_BINS = 15


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
class ECE(metrics.Average):
    @classmethod
    def from_model_output(
        cls,
        *,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        num_bins: int = _DEFAULT_NUM_OF_BINS,
        **kwargs
    ) -> metrics.Average:

        probabilities = softmax(logits, axis=-1)
        confidences = jnp.max(probabilities, axis=-1)
        predictions = jnp.argmax(probabilities, axis=-1)
        accuracies = (predictions == labels).astype(jnp.float32)

        bin_indices = jnp.clip(
            (confidences * num_bins).astype(jnp.int32), 0, num_bins - 1
        )

        num = confidences.shape[0]

        def compute_bin_contribution(bin):
            in_bin = bin_indices == bin
            count = jnp.sum(in_bin)

            avg_accuracy = jnp.sum(accuracies * in_bin) / jnp.maximum(count, 1)
            avg_confidence = jnp.sum(confidences * in_bin) / jnp.maximum(count, 1)

            weight = count / num
            return weight * jnp.abs(avg_accuracy - avg_confidence)

        ece = jnp.sum(vmap(compute_bin_contribution)(jnp.arange(num_bins)))

        return super(ECE, cls).from_model_output(values=ece)


@dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")
    brier_score: BrierScore
    ece: ECE


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
