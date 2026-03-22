from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .params import ParamValue, ParamSpec, SearchParam, normalize_parameter_space

LossRaw = object
SeedLossFn = Callable[..., LossRaw]
SeedLossCallMode = Literal["two_positional", "three_positional", "keyword_context"]


def _resolve_seed_loss_call_mode(fn: SeedLossFn) -> SeedLossCallMode:
    signature = inspect.signature(fn)

    try:
        signature.bind({}, 0, {})
    except TypeError:
        pass
    else:
        return "three_positional"

    try:
        signature.bind({}, 0, context={})
    except TypeError:
        pass
    else:
        return "keyword_context"

    try:
        signature.bind({}, 0)
    except TypeError as exc:
        raise TypeError(
            "@loss functions must accept either (params, seed) or (params, seed, context)"
        ) from exc
    return "two_positional"


@dataclass(frozen=True)
class LossDefinition:
    name: str
    description: str
    parameter_space: dict[str, SearchParam]
    seed_loss_fn: SeedLossFn
    seed_loss_call_mode: SeedLossCallMode
    default_num_chunks: int = 10
    default_chunk_size: int = 100
    seed_start: int = 0
    source_file: Optional[str] = field(default=None, compare=False, hash=False)

    def evaluate_seed_loss(
        self,
        params: dict[str, ParamValue],
        seed: int,
        context: dict[str, object],
    ) -> LossRaw:
        if self.seed_loss_call_mode == "three_positional":
            return self.seed_loss_fn(params, seed, context)
        if self.seed_loss_call_mode == "keyword_context":
            return self.seed_loss_fn(params, seed, context=context)
        return self.seed_loss_fn(params, seed)


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
        call_mode = _resolve_seed_loss_call_mode(fn)
        return register_loss(
            LossDefinition(
                name=name,
                description=description,
                parameter_space=normalize_parameter_space(parameter_space),
                seed_loss_fn=fn,
                seed_loss_call_mode=call_mode,
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
