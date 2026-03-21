from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import optuna

from .evaluate import normalize_seed_loss, summarize_rows
from .registry import LossDefinition, register_loss


@dataclass(frozen=True)
class OptimizeResult:
    loss_name: str
    best_value: float
    best_params: dict[str, float]
    n_trials: int
    study_name: str


def optimize(
    loss: LossDefinition,
    *,
    n_trials: int = 10,
    seeds: Iterable[int] | None = None,
    random_seed: int = 123,
    entry_id: str | None = None,
    direction: str = "minimize",
) -> OptimizeResult:
    """Optimize a registered loss locally with Optuna.

    This is the minimal high-level API: define a loss, then optimize(loss).
    """

    register_loss(loss, overwrite=True)

    active_seeds = (
        list(seeds)
        if seeds is not None
        else list(
            range(
                loss.seed_start,
                loss.seed_start + (loss.default_num_chunks * loss.default_chunk_size),
            )
        )
    )
    if not active_seeds:
        raise ValueError("seeds must not be empty")

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction=direction, sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            name: trial.suggest_float(name, bounds[0], bounds[1])
            for name, bounds in loss.parameter_space.items()
        }

        rows = [
            normalize_seed_loss(
                seed,
                loss.seed_loss_fn(
                    params,
                    seed,
                    {
                        "entry_id": entry_id,
                    },
                ),
            )
            for seed in active_seeds
        ]
        summary = summarize_rows(rows)
        trial.set_user_attr("n_seeds", summary["n_seeds"])
        return float(summary["mean_total_loss"])

    study.optimize(objective, n_trials=n_trials)

    return OptimizeResult(
        loss_name=loss.name,
        best_value=float(study.best_value),
        best_params={k: float(v) for k, v in study.best_params.items()},
        n_trials=n_trials,
        study_name=study.study_name,
    )
