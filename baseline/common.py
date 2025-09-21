import jax
import jax.numpy as jnp
import optax


def cross_entropy_loss(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return jnp.mean(loss)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            batch['image'],
            train=True,
            mutable=['batch_stats'], 
        )
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, new_model_state

    (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(batch_stats=new_model_state['batch_stats'])
    return new_state


@jax.jit
def compute_metrics(*, state, batch):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch['image'], train=False) 
    loss = cross_entropy_loss(logits, batch['label'])
    metric_update = state.metrics.single_from_model_output(
        logits=logits,
        labels=batch['label'],
        loss=loss
    )
    metrics = state.metrics.merge(metric_update)
    return state.replace(metrics=metrics)
