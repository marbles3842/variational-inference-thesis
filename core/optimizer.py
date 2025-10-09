import optax
from jax.tree_util import tree_structure, tree_flatten_with_path, tree_unflatten

from .ivon import ivon


def create_warmup_cosine_schedule(
    init_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    end_lr: float = 0.0,
) -> optax.Schedule:
    """
    Creates a learning rate schedule that combines linear warmup with cosine annealing.

    Args:
        init_lr: Peak learning rate (reached after warmup)
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of optimization steps per epoch
        end_lr: Final learning rate after cosine decay

    Returns:
        Optax schedule function
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    cosine_steps = total_steps

    warmup_schedule = optax.linear_schedule(
        init_value=init_lr / warmup_epochs,
        end_value=init_lr,
        transition_steps=warmup_steps,
    )

    cosine_schedule = optax.cosine_decay_schedule(
        init_value=init_lr, decay_steps=cosine_steps, alpha=end_lr / init_lr
    )

    schedule = optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule], boundaries=[warmup_steps]
    )

    return schedule


def _weight_decay_mask_fn(params):
    """Only apply weight decay to 'kernel' parameters (conv/dense weights)"""

    def should_decay(path):
        param_name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])

        # Only apply weight decay to kernel/weight parameters
        # Exclude: bias, gamma, beta, tau, scale, shift, etc.
        return param_name in ["kernel", "weight"]

    flat_params = tree_flatten_with_path(params)[0]
    mask_flat = [should_decay(path) for path, _ in flat_params]

    return tree_unflatten(tree_structure(params), mask_flat)


def create_cifar_sgd_optimizer(
    learning_rate: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    momentum: float,
    weight_decay: float,
):
    """
    Creates SGD optimizer with warmup + cosine annealing schedule.

    Args:
        params: Model parameters
        learning_rate: Peak learning rate
        warmup_epochs: Warmup period
        total_epochs: Total training epochs
        steps_per_epoch: Steps per epoch (dataset_size // batch_size)
        momentum: SGD momentum
        weight_decay: L2 regularization strength
    """

    lr_schedule = create_warmup_cosine_schedule(
        init_lr=learning_rate,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        end_lr=0.0,
    )

    optimizer = optax.chain(
        optax.add_decayed_weights(
            weight_decay=weight_decay, mask=_weight_decay_mask_fn
        ),
        optax.sgd(
            learning_rate=lr_schedule,
            momentum=momentum,
            nesterov=False,
        ),
    )

    return optimizer


def create_cifar_ivon_optimizer(
    learning_rate: float,
    warmup_epochs: int,
    total_epochs: int,
    hess_init: float,
    steps_per_epoch: int,
    momentum: float,
    momentum_hess: float,
    ess: float,
    weight_decay: float,
):
    """
    Creates IVON optimizer with warmup + cosine annealing schedule.

    Args:
        learning_rate: Peak learning rate
        warmup_epochs: Warmup period
        total_epochs: Total training epochs
        hess_init: Initial Hessian diagonal estimate
        steps_per_epoch: Steps per epoch (dataset_size // batch_size)
        momentum: Gradient momentum (beta1)
        momentum_hess: Hessian momentum (beta2)
        ess: Effective sample size parameter
        weight_decay: L2 regularization strength
    """

    lr_schedule = create_warmup_cosine_schedule(
        init_lr=learning_rate,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        end_lr=0.0,
    )

    optimizer = ivon(
        learning_rate=lr_schedule,
        ess=ess,
        hess_init=hess_init,
        weight_decay=weight_decay,
        beta1=momentum,
        beta2=momentum_hess,
    )

    return optimizer
