import os
import jax
import jax.numpy as jnp
import optax
import yaml


from .cifar_dataset import get_cifar10_train_val_loaders
from .metrics_logger import MetricsLogger
from .train_state import create_state
from .optimizer import create_cifar_sgd_optimizer
from models.resnet import ResNet20


NUM_CLASSES = 10
CIFAR10_NUM_FILTERS = 16


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


if __name__ == '__main__':
    
    config_path = os.path.join(os.path.dirname(__file__), "train_cifar10_config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = config['cifar10']['sgd']


    model = ResNet20(num_classes=NUM_CLASSES, num_filters=CIFAR10_NUM_FILTERS)
    

    train_ds, val_ds = get_cifar10_train_val_loaders(
        train_batch_size=config['train_batch_size'], 
        val_batch_size=config['val_batch_size'],
        seed=43, 
        num_epochs=config['num_epochs']
    )

    num_steps_per_epoch = jnp.ceil(train_ds._data_source.__len__()/config['train_batch_size']).astype(jnp.int32)

    init_rng = jax.random.key(0)
    
    optimizer = create_cifar_sgd_optimizer(
        learning_rate=config['learning_rate'],
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['num_epochs'],
        steps_per_epoch=num_steps_per_epoch,
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

  
    state = create_state(model=model, rng=init_rng, x0 = jnp.ones([1, 32, 32, 3]), optimizer=optimizer)

    del init_rng

    logdir = img_dir = os.path.join(os.path.dirname(__file__), "..", "out")
    metrics_log_path = os.path.join(logdir, "metrics-sgd.csv")

    with MetricsLogger(metrics_log_path) as logger:
        
        for step, batch in enumerate(train_ds):
            
            batch = jax.device_put(batch)
            
            state = train_step(state, batch)
            state = compute_metrics(state=state, batch=batch)
            
            if(step +1) % num_steps_per_epoch == 0:
                for metric, value in state.metrics.compute().items():
                    logger.update('train', metric, value)

                state = state.replace(metrics=state.metrics.empty())
                
                val_state = state
                
                for val_batch in val_ds:
                    val_batch = jax.device_put(val_batch)
                    val_state = compute_metrics(state=val_state, batch=val_batch)
                
                for metric, value in val_state.metrics.compute().items():
                    logger.update('val', metric, value)
                
                logger.end_epoch()