import optax


def create_warmup_cosine_schedule(
    init_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    end_lr: float = 0.0
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
    cosine_steps = total_steps - warmup_steps
    
    warmup_schedule = optax.linear_schedule(
        init_value=init_lr / warmup_epochs,  
        end_value=init_lr,                 
        transition_steps=warmup_steps
    )
    
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=init_lr,
        decay_steps=cosine_steps,
        alpha=end_lr / init_lr 
    )
    
    schedule = optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps]
    )
    
    return schedule



def create_cifar_sgd_optimizer(
    learning_rate: float = 0.2,
    warmup_epochs: int = 5,
    total_epochs: int = 200,
    steps_per_epoch: int = 1000,
    momentum: float = 0.9,
    weight_decay: float = 2e-4
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
        end_lr=0.0
    )
    
    optimizer = optax.sgd(
        learning_rate=lr_schedule,
        momentum=momentum,
        nesterov=False 
    )
    
    if weight_decay > 0:
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optimizer
        )
    
    return optimizer
