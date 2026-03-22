from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional

from .params import ParamValue, ParamSpec, SearchParam, normalize_parameter_space

LossRaw = object
SeedLossFn = Callable[[dict[str, ParamValue], int, dict[str, object]], LossRaw]


@dataclass(frozen=True)
class LossDefinition:
    name: str
    description: str
    parameter_space: dict[str, SearchParam]
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
    name: str | None = None,
    description: str | None = None,
    parameter_space: dict[str, ParamSpec] | None = None,
    default_num_chunks: int = 10,
    default_chunk_size: int = 100,
    seed_start: int = 0,
):
    missing_fields: list[str] = []
    if not name:
        missing_fields.append("name")
    if not description:
        missing_fields.append("description")
    if not parameter_space:
        missing_fields.append("parameter_space")

    if missing_fields:
        missing_str = ", ".join(missing_fields)
        raise ValueError(
            "@loss requires non-empty metadata fields: "
            f"{missing_str}. Example: "
            "@loss(name='my_loss', description='...', parameter_space={'x': (0.0, 1.0)})"
        )

    def _wrap(fn: SeedLossFn) -> LossDefinition:
        try:
            src = inspect.getfile(fn)
        except (TypeError, OSError):
            src = None
        return register_loss(
            LossDefinition(
                name=name,
                description=description,
                parameter_space=normalize_parameter_space(parameter_space),
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
