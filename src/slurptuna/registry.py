from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional

LossRaw = object
SeedLossFn = Callable[[dict[str, float], int, dict[str, object]], LossRaw]


@dataclass(frozen=True)
class LossDefinition:
    name: str
    description: str
    parameter_space: dict[str, tuple[float, float]]
    seed_loss_fn: SeedLossFn
    default_num_chunks: int = 10
    default_chunk_size: int = 100
    seed_start: int = 0
    source_file: Optional[str] = field(default=None, compare=False, hash=False)


_REGISTRY: dict[str, LossDefinition] = {}


def register_loss(loss: LossDefinition, *, overwrite: bool = True) -> LossDefinition:
    if (not overwrite) and loss.name in _REGISTRY:
        raise KeyError(f"Loss '{loss.name}' is already registered")
    _REGISTRY[loss.name] = loss
    return loss


def loss(
    *,
    name: str,
    description: str,
    parameter_space: dict[str, tuple[float, float]],
    default_num_chunks: int = 10,
    default_chunk_size: int = 100,
    seed_start: int = 0,
):
    def _wrap(fn: SeedLossFn) -> LossDefinition:
        try:
            src = inspect.getfile(fn)
        except (TypeError, OSError):
            src = None
        return register_loss(
            LossDefinition(
                name=name,
                description=description,
                parameter_space=parameter_space,
                seed_loss_fn=fn,
                default_num_chunks=default_num_chunks,
                default_chunk_size=default_chunk_size,
                seed_start=seed_start,
                source_file=src,
            )
        )

    return _wrap


def get_registered_loss(name: str) -> LossDefinition:
    return _REGISTRY[name]


def get_registered_losses() -> list[LossDefinition]:
    return list(_REGISTRY.values())


def maybe_get_registered_loss(name: str) -> Optional[LossDefinition]:
    return _REGISTRY.get(name)
