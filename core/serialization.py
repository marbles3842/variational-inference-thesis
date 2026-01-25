import jax
from flax import serialization
from flax.core import FrozenDict
from .ivon import IVONState


def save_params(path: str, params: FrozenDict | IVONState):
    """Save model parameters or optimizer state to disk.

    Args:
        path: File path to save to
        params: Parameters or state to save
    """
    bytes_output = serialization.to_bytes(params)

    with open(path, "wb") as f:
        f.write(bytes_output)


def load_model_params(path: str, params: FrozenDict):
    """Load model parameters from disk.

    Args:
        path: File path to load from
        params: Template parameters with correct structure

    Returns:
        Loaded parameters
    """
    with open(path, "rb") as f:
        opt_params = serialization.from_bytes(params, f.read())

    return opt_params


def load_ivon_state(path: str, params: FrozenDict):
    """Load IVON optimizer state from disk.

    Args:
        path: File path to load from
        params: Template parameters for creating state structure

    Returns:
        Loaded IVON state
    """
    target = IVONState(
        ess=0.0,
        beta1=0.0,
        beta2=0.0,
        weight_decay=0.0,
        momentum=jax.tree.map(lambda x: jax.numpy.zeros_like(x), params),
        hess=jax.tree.map(lambda x: jax.numpy.zeros_like(x), params),
        axis_name=None,
        current_step=0,
        grad_acc=None,
        nxg_acc=None,
        noise=None,
        acc_count=0,
    )
    with open(path, "rb") as f:
        opt_state = serialization.from_bytes(target, f.read())
    return opt_state
