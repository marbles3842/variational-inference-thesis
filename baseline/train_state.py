from clu import metrics
from flax.training import train_state
from flax.core import FrozenDict
from flax import struct
from typing import Any

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
    batch_stats: FrozenDict[str, Any]
    metrics: Metrics
    
def create_state(model, rng, x0, optimizer):
    variables = model.init(rng, x0, train=True)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        metrics = Metrics.empty(),
        batch_stats = batch_stats
    )