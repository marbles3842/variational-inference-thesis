from typing import Tuple, Callable, Any

import optax
import jax
import jmp
import jax.numpy as jnp
import jax.random as jr
from jax.lax import scan, cond
import flax.linen as nn
from flax.core import FrozenDict
from flax.struct import dataclass, field
from flax.training.train_state import TrainState

from core.ivon import sample_parameters, accumulate_gradients


Dtype = Any


_policy = jmp.Policy(
    param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32
)


def _linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    dtype: Dtype = jnp.float32,
):
    return jnp.linspace(beta_start, beta_end, timesteps, dtype=dtype)


@dataclass
class DiffusionModelSchedule:
    beta: jax.Array
    alpha: jax.Array
    alpha_cumprod: jax.Array
    timesteps: int = field(pytree_node=False)

    @classmethod
    def create(cls, timesteps: int, dtype: Dtype = jnp.float32):
        beta = _linear_beta_schedule(timesteps=timesteps, dtype=dtype)
        alpha = 1.0 - beta
        alpha_cumprod = jnp.cumprod(alpha)
        return cls(
            beta=beta,
            alpha=alpha,
            alpha_cumprod=alpha_cumprod,
            timesteps=timesteps,
        )


def _negative_elbo(
    batch: jax.Array,
    params: FrozenDict,
    apply_fn: Callable,
    schedule: DiffusionModelSchedule,
    key_t: jr.PRNGKey,
    key_eps: jr.PRNGKey,
):
    params, batch = _policy.cast_to_compute((params, batch))

    batch_size = batch.shape[0]
    t = jr.randint(key=key_t, minval=0, maxval=schedule.timesteps, shape=(batch_size,))
    epsilon = jr.normal(key_eps, shape=batch.shape, dtype=_policy.compute_dtype)

    alpha_bar_t = (
        schedule.alpha_cumprod[t]
        .astype(_policy.compute_dtype)
        .reshape(batch_size, 1, 1, 1)
    )
    x_t = jnp.sqrt(alpha_bar_t) * batch + jnp.sqrt(1 - alpha_bar_t) * epsilon
    variables = {"params": params}
    epsilon_theta = apply_fn(variables, x_t, t)
    loss = (epsilon_theta - epsilon) ** 2

    loss_weights = (1 - schedule.alpha_cumprod[t]).reshape(batch_size, 1, 1, 1)

    loss_weights = loss_weights.astype(_policy.compute_dtype)

    loss = loss * loss_weights
    return _policy.cast_to_output(loss.mean())


@jax.jit
def diffusion_train_step(
    state: TrainState,
    batch: jax.Array,
    key: jr.PRNGKey,
    schedule: DiffusionModelSchedule,
):
    """
    Base training step that can be used with Adam or SGD
    Implements algorithm 1 from https://arxiv.org/pdf/2006.11239
    """
    key_t, key_eps, key = jr.split(key, num=3)

    def loss(params, key_t, key_eps):
        return _negative_elbo(
            batch=batch,
            params=params,
            apply_fn=state.apply_fn,
            schedule=schedule,
            key_t=key_t,
            key_eps=key_eps,
        )

    loss, grads = jax.value_and_grad(loss)(state.params, key_t, key_eps)
    new_state = state.apply_gradients(grads=grads)
    metric_update = new_state.metrics.single_from_model_output(loss=loss)
    metrics = new_state.metrics.merge(metric_update)
    return new_state.replace(metrics=metrics), key


@jax.jit
def _compute_hessian_metrics(hessian):
    all_hess_values = jnp.concatenate([jnp.ravel(h) for h in jax.tree.leaves(hessian)])
    return jnp.mean(all_hess_values), jnp.min(all_hess_values)


@jax.jit
def diffusion_train_step_with_ivon(
    state: TrainState,
    batch: jax.Array,
    train_key: jr.PRNGKey,
    schedule: DiffusionModelSchedule,
    mc_keys: jax.Array,
):
    """
    Training step with MC sampling for IVON
    Implements algorithm 1 from https://arxiv.org/pdf/2006.11239
    """

    def loss(params, key_t, key_eps):
        return _negative_elbo(
            batch=batch,
            params=params,
            apply_fn=state.apply_fn,
            schedule=schedule,
            key_t=key_t,
            key_eps=key_eps,
        )

    params = state.params
    opt_state_dtypes = jax.tree.map(lambda x: x.dtype, state.opt_state)

    def mc_step(opt_state, key):
        key_t, key_eps, key = jr.split(key, num=3)
        psample, opt_state = sample_parameters(key, params, opt_state)
        updates = jax.grad(loss)(psample, key_t, key_eps)
        updates = jax.tree.map(lambda x: x.astype(jnp.float32), updates)
        opt_state = accumulate_gradients(updates, opt_state)
        opt_state = jax.tree.map(
            lambda x, dtype: x.astype(dtype), opt_state, opt_state_dtypes
        )

        return opt_state, None

    opt_state, _ = scan(mc_step, state.opt_state, mc_keys[:-1])

    key_t, key_eps, key = jr.split(mc_keys[-1], num=3)
    psample, opt_state = sample_parameters(key, params, opt_state)
    updates = jax.grad(loss)(psample, key_t, key_eps)
    updates, opt_state = state.tx.update(updates, opt_state, params)

    new_params = optax.apply_updates(params, updates)
    new_state = state.replace(params=new_params, opt_state=opt_state)
    key_t, key_eps, train_key = jr.split(train_key, num=3)
    loss = _negative_elbo(
        batch=batch,
        params=new_params,
        apply_fn=new_state.apply_fn,
        schedule=schedule,
        key_t=key_t,
        key_eps=key_eps,
    )
    hess_mean, hess_min = _compute_hessian_metrics(opt_state[0].hess)
    metric_update = new_state.metrics.single_from_model_output(
        loss=loss, hess_mean=hess_mean, hess_min=hess_min
    )
    metrics = new_state.metrics.merge(metric_update)
    return new_state.replace(metrics=metrics), train_key


def diffusion_sample(
    model: nn.Module,
    variables: FrozenDict,
    shape: Tuple,
    key: jr.PRNGKey,
    schedule: DiffusionModelSchedule,
    dtype: Dtype = jnp.float32,
):
    batch_size = shape[0]

    key, subkey = jr.split(key)

    x_t = jr.normal(key=subkey, shape=shape, dtype=dtype)
    variables_compute = _policy.cast_to_compute(variables)

    def reverse_step(carry, t):
        x_t, key = carry

        t_arr = jnp.full((batch_size,), t, dtype=dtype)
        key, subkey = jr.split(key)

        z = cond(
            t > 0,
            lambda _: jr.normal(key=subkey, shape=shape, dtype=dtype),
            lambda _: jnp.zeros(shape, dtype=dtype),
            None,
        )

        x_t_compute, t_arr_compute = _policy.cast_to_compute((x_t, t_arr))
        predicted_noise = model.apply(variables_compute, x_t_compute, t_arr_compute)
        predicted_noise = _policy.cast_to_output(predicted_noise)

        alpha_t = schedule.alpha[t]
        alpha_t_bar = schedule.alpha_cumprod[t]
        beta_t = schedule.beta[t]

        x_t = (
            1
            / jnp.sqrt(alpha_t)
            * (x_t - predicted_noise * (1 - alpha_t) / jnp.sqrt(1 - alpha_t_bar))
        )

        sigma_t = jnp.sqrt(beta_t)
        x_t = x_t + sigma_t * z
        return (x_t, key), None

    (x_t, key), _ = scan(
        reverse_step, (x_t, key), jnp.arange(schedule.timesteps - 1, -1, -1)
    )

    return x_t
