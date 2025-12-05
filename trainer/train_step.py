import jax
import optax

from core.ivon import sample_parameters, accumulate_gradients
from .metrics import cross_entropy_loss


@jax.jit
def train_step(state, batch):
    """
    Train step that can be used for AdamW or SGD
    """

    def loss_fn(params):
        variables = {"params": params}
        logits = state.apply_fn(variables, batch["image"])
        loss = cross_entropy_loss(logits=logits, labels=batch["label"])
        return loss

    grads = jax.grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state


@jax.jit
def train_step_ivon(state, batch, rng_key, train_mcsamples=1):
    """
    Train step that must be used with IVON
    """

    def loss_fn(params):
        variables = {"params": params}
        logits = state.apply_fn(variables, batch["image"])
        loss = cross_entropy_loss(logits=logits, labels=batch["label"])
        return loss

    rng_key, *mc_keys = jax.random.split(rng_key, train_mcsamples + 1)
    opt_state = state.opt_state
    params = state.params
    for i, key in enumerate(mc_keys):
        psample, opt_state = sample_parameters(key, params, opt_state)

        updates = jax.grad(loss_fn)(psample)

        if i == train_mcsamples - 1:
            updates, opt_state = state.tx.update(updates, opt_state, params)
        else:
            opt_state = accumulate_gradients(updates, opt_state)

    new_params = optax.apply_updates(params, updates)
    return state.replace(params=new_params, opt_state=opt_state)
