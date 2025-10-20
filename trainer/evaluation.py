import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp
from functools import partial

from .metrics import Metrics
from core.ivon import sample_parameters


def _compute_sampled_logits(model, opt_params, opt_state, images, rng, n_samples):
    def single_sample_logits(sample_rng):
        sampled_params, _ = sample_parameters(sample_rng, opt_params, opt_state)
        return model.apply({"params": sampled_params}, images)

    sample_rngs = jax.random.split(rng, n_samples)
    all_logits = vmap(single_sample_logits)(sample_rngs)

    return all_logits


@partial(jax.jit, static_argnames=("num_classes", "n_samples"))
def _compute_bayesian_metrics(all_logits, all_labels, num_classes, n_samples):
    log_probs = jax.nn.log_softmax(all_logits, axis=2)
    log_bayesian_probs = logsumexp(log_probs, b=1.0 / n_samples, axis=0)

    labels_one_hot = jax.nn.one_hot(all_labels, num_classes)
    mean_loss = -jnp.mean(jnp.sum(labels_one_hot * log_bayesian_probs, axis=1))

    return log_bayesian_probs, mean_loss


def evaluate_ivon_with_sampling(
    model, opt_params, opt_state, dataset, seed, n_samples, num_classes: int
):
    rng = jax.random.key(seed)

    all_logits = []
    all_labels = []

    for batch in dataset:
        batch = jax.device_put(batch)

        rng, batch_rng = jax.random.split(rng)

        batch_logits = _compute_sampled_logits(
            model, opt_params, opt_state, batch["image"], batch_rng, n_samples
        )

        all_logits.append(batch_logits)
        all_labels.append(batch["label"])

    all_logits = jnp.concatenate(all_logits, axis=1)
    all_labels = jnp.concatenate(all_labels, axis=0)

    bayesian_probs, mean_loss = _compute_bayesian_metrics(
        all_logits, all_labels, num_classes, n_samples
    )

    metrics = Metrics.empty()
    metric_update = metrics.single_from_model_output(
        logits=bayesian_probs,
        labels=all_labels,
        loss=mean_loss,
    )

    return metric_update.compute()
