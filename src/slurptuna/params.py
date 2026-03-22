from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import optuna

ParamDType = Literal["float", "int", "str", "bool"]
ParamValue = float | int | str | bool
RangeSpec = tuple[float, float]


@dataclass(frozen=True)
class SearchParam:
    range: tuple[float, float] | None = None
    allowed: tuple[ParamValue, ...] | None = None
    dtype: ParamDType | None = None


ParamSpec = RangeSpec | SearchParam


def search_param(
    *,
    range: tuple[float, float] | None = None,
    allowed: list[ParamValue] | tuple[ParamValue, ...] | None = None,
    dtype: ParamDType | None = None,
) -> SearchParam:
    values: tuple[ParamValue, ...] | None = tuple(allowed) if allowed is not None else None
    return SearchParam(range=range, allowed=values, dtype=dtype)


def normalize_parameter_space(parameter_space: dict[str, ParamSpec]) -> dict[str, SearchParam]:
    normalized: dict[str, SearchParam] = {}
    for name, spec in parameter_space.items():
        normalized[name] = normalize_param_spec(name, spec)
    return normalized


def normalize_param_spec(name: str, spec: ParamSpec) -> SearchParam:
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError(f"Parameter '{name}' range tuple must have exactly 2 values")
        low, high = spec
        return _validate_search_param(name, SearchParam(range=(float(low), float(high)), dtype="float"))

    if not isinstance(spec, SearchParam):
        raise ValueError(
            f"Parameter '{name}' must be a (min, max) tuple or SearchParam(...)"
        )
    return _validate_search_param(name, spec)


def suggest_param(trial: optuna.trial.Trial, name: str, spec: SearchParam) -> ParamValue:
    if spec.allowed is not None:
        values = list(spec.allowed)
        return trial.suggest_categorical(name, values)

    assert spec.range is not None
    low, high = spec.range
    if spec.dtype == "int":
        return trial.suggest_int(name, int(low), int(high))
    return trial.suggest_float(name, float(low), float(high))


def _validate_search_param(name: str, spec: SearchParam) -> SearchParam:
    has_range = spec.range is not None
    has_allowed = spec.allowed is not None

    if has_range == has_allowed:
        raise ValueError(
            f"Parameter '{name}' must define exactly one of range or allowed"
        )

    if has_range:
        assert spec.range is not None
        low, high = spec.range
        if low > high:
            raise ValueError(f"Parameter '{name}' range low must be <= high")
        dtype = spec.dtype or "float"
        if dtype not in {"float", "int"}:
            raise ValueError(
                f"Parameter '{name}' with range only supports dtype='float' or 'int'"
            )
        return SearchParam(range=(float(low), float(high)), allowed=None, dtype=dtype)

    assert spec.allowed is not None
    if len(spec.allowed) == 0:
        raise ValueError(f"Parameter '{name}' allowed must not be empty")

    inferred = _infer_allowed_dtype(spec.allowed)
    dtype = spec.dtype or inferred
    if dtype != inferred:
        raise ValueError(
            f"Parameter '{name}' allowed values do not match dtype='{dtype}'"
        )

    return SearchParam(range=None, allowed=spec.allowed, dtype=dtype)


def _infer_allowed_dtype(values: tuple[ParamValue, ...]) -> ParamDType:
    if all(isinstance(v, bool) for v in values):
        return "bool"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "int"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return "float"
    if all(isinstance(v, str) for v in values):
        return "str"
    raise ValueError("allowed values must all share a compatible type")