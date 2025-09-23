import jax
import jax.numpy as jnp
import optax


def cross_entropy_loss(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return jnp.mean(loss)


@jax.jit
def compute_metrics(*, state, batch):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, batch["image"], train=False)
    loss = cross_entropy_loss(logits, batch["label"])
    metric_update = state.metrics.single_from_model_output(
        logits=logits, labels=batch["label"], loss=loss
    )
    metrics = state.metrics.merge(metric_update)
    return state.replace(metrics=metrics)
