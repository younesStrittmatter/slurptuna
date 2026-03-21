from .api import OptimizeResult, optimize
from .registry import LossDefinition, get_registered_loss, get_registered_losses, loss, register_loss

__all__ = [
    "LossDefinition",
    "loss",
    "register_loss",
    "get_registered_loss",
    "get_registered_losses",
    "OptimizeResult",
    "optimize",
]
