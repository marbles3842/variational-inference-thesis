import argparse
import os
import jax
import jax.numpy as jnp
import yaml


from .cifar_dataset import get_cifar10_train_val_loaders
from .metrics_logger import MetricsLogger
from .train_state import create_state
from .optimizer import create_cifar_sgd_optimizer
from .common import train_step, compute_metrics
from models.resnet import ResNet20


NUM_CLASSES = 10
CIFAR10_NUM_FILTERS = 16


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Seed for the initialization")
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(__file__), "train_cifar10_config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = config['cifar10']['sgd']


    model = ResNet20(num_classes=NUM_CLASSES, num_filters=CIFAR10_NUM_FILTERS)
    
    init_rng = jax.random.key(args.seed)
    

    train_ds, val_ds = get_cifar10_train_val_loaders(
        train_batch_size=config['train_batch_size'], 
        val_batch_size=config['val_batch_size'],
        seed=args.seed, 
        num_epochs=config['num_epochs']
    )

    num_steps_per_epoch = jnp.ceil(train_ds._data_source.__len__()/config['train_batch_size']).astype(jnp.int32)
    
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