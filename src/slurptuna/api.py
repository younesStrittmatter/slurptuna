from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
from enum import Enum
from typing import Iterable
from pathlib import Path

import optuna

from .evaluate import normalize_seed_loss, summarize_rows
from .params import ParamValue, suggest_param
from .registry import LossDefinition, register_loss
from .slurm_backend import SlurmConfig, submit_trial, wait_for_summary


class ExecutionMode(Enum):
    SINGLE = "single"
    DISTRIBUTED = "distributed"


@dataclass(frozen=True)
class OptimizeResult:
    loss_name: str
    best_value: float
    best_params: dict[str, ParamValue]
    n_trials: int
    study_name: str
    run_dir: str | None = None
    mode: ExecutionMode = ExecutionMode.SINGLE


@dataclass(frozen=True)
class MultiOptimizeResult:
    loss_name: str
    n_entries: int
    entries: list[str]
    results_by_entry: dict[str, OptimizeResult]
    best_values_by_entry: dict[str, float]
    best_params_by_entry: dict[str, dict[str, ParamValue]]
    mode: ExecutionMode = ExecutionMode.SINGLE
    run_dir: str | None = None


def _coerce_mode(mode: ExecutionMode | str) -> ExecutionMode:
    if isinstance(mode, ExecutionMode):
        return mode
    aliases = {
        "single": ExecutionMode.SINGLE,
        "local": ExecutionMode.SINGLE,
        "distributed": ExecutionMode.DISTRIBUTED,
        "slurm": ExecutionMode.DISTRIBUTED,
    }
    try:
        return aliases[str(mode).lower()]
    except KeyError as exc:
        raise ValueError("mode must be one of: single, distributed") from exc


def _resolve_seed_layout(
    *,
    loss: LossDefinition,
    seeds: Iterable[int] | None,
    n_seeds: int | None,
    chunk_size: int | None,
    num_chunks: int | None,
) -> tuple[list[int], int, int]:
    resolved_chunk_size = chunk_size or loss.default_chunk_size
    if resolved_chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if seeds is not None:
        active_seeds = list(seeds)
        if not active_seeds:
            raise ValueError("seeds must not be empty")
    else:
        if n_seeds is not None:
            if n_seeds <= 0:
                raise ValueError("n_seeds must be > 0")
            total_seeds = n_seeds
        else:
            resolved_num_chunks = num_chunks or loss.default_num_chunks
            if resolved_num_chunks <= 0:
                raise ValueError("num_chunks must be > 0")
            total_seeds = resolved_chunk_size * resolved_num_chunks

        active_seeds = list(range(loss.seed_start, loss.seed_start + total_seeds))

    derived_num_chunks = (len(active_seeds) + resolved_chunk_size - 1) // resolved_chunk_size
    if num_chunks is not None and num_chunks != derived_num_chunks:
        raise ValueError(
            f"num_chunks={num_chunks} does not match seeds/chunk_size layout ({derived_num_chunks})"
        )
    return active_seeds, resolved_chunk_size, derived_num_chunks


def _next_versioned_run_name(run_root: Path, loss_name: str) -> str:
    pat = re.compile(rf"^{re.escape(loss_name)}_v(\d{{4}})$")
    max_version = 0
    if run_root.exists():
        for p in run_root.iterdir():
            if not p.is_dir():
                continue
            m = pat.match(p.name)
            if m:
                max_version = max(max_version, int(m.group(1)))
    return f"{loss_name}_v{max_version + 1:04d}"


def _objective_local(
    *,
    trial: optuna.trial.Trial,
    loss: LossDefinition,
    active_seeds: list[int],
    entry_id: str | None,
) -> float:
    params = {name: suggest_param(trial, name, spec) for name, spec in loss.parameter_space.items()}

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


def optimize_run(
    loss: LossDefinition,
    *,
    n_trials: int = 10,
    seeds: Iterable[int] | None = None,
    n_seeds: int | None = None,
    chunk_size: int | None = None,
    num_chunks: int | None = None,
    random_seed: int = 123,
    entry_id: str | None = None,
    direction: str = "minimize",
    mode: ExecutionMode = ExecutionMode.SINGLE,
    run_root: str | Path = "runs",
    run_name: str | None = None,
    loss_module: str | None = None,
    slurm_poll_seconds: int = 15,
    slurm_timeout_minutes: int = 120,
    cpus_per_task: int = 1,
    max_concurrent_trials: int = 1,
    array_parallelism_limit: int | None = None,
    worker_parallelism: int = 1,
    worker_time_limit: timedelta = timedelta(hours=1),
    trial_retry_attempts: int = 1,
) -> OptimizeResult:
    """Single public orchestration API: define a loss and optimize it.

    mode:
    - single: execute objective directly in-process
    - distributed: submit chunk/reduce jobs via sbatch arrays for each trial

    seed/chunk controls:
    - seeds: explicit iterable of seed IDs (highest priority)
    - n_seeds: generate contiguous seeds from loss.seed_start
    - chunk_size / num_chunks: controls Slurm array shape and default seed count
    """

    register_loss(loss, overwrite=True)
    resolved_mode = _coerce_mode(mode)
    if max_concurrent_trials <= 0:
        raise ValueError("max_concurrent_trials must be > 0")
    if worker_parallelism <= 0:
        raise ValueError("worker_parallelism must be > 0")
    if array_parallelism_limit is not None and array_parallelism_limit <= 0:
        raise ValueError("array_parallelism_limit must be > 0 when provided")
    if worker_time_limit.total_seconds() <= 0:
        raise ValueError("worker_time_limit must be > 0 seconds")

    active_seeds, resolved_chunk_size, resolved_num_chunks = _resolve_seed_layout(
        loss=loss,
        seeds=seeds,
        n_seeds=n_seeds,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
    )

    run_root_path = Path(run_root)
    chosen_run_name = run_name or _next_versioned_run_name(run_root_path, loss.name)
    run_dir = run_root_path / chosen_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at_utc = datetime.now(timezone.utc)

    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "loss_name": loss.name,
                "run_name": chosen_run_name,
                "run_dir": str(run_dir),
                "mode": resolved_mode.value,
                "n_trials": n_trials,
                "entry_id": entry_id,
                "started_at_utc": started_at_utc.isoformat(),
                "seed_start": active_seeds[0],
                "seed_end": active_seeds[-1],
                "n_seeds": len(active_seeds),
                "chunk_size": resolved_chunk_size,
                "num_chunks": resolved_num_chunks,
                "max_concurrent_trials": max_concurrent_trials,
                "array_parallelism_limit": array_parallelism_limit,
                "worker_parallelism": worker_parallelism,
                "worker_time_limit_seconds": int(worker_time_limit.total_seconds()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    storage = f"sqlite:///{(run_dir / 'optuna.db').resolve()}"
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
        study_name=loss.name,
    )

    if resolved_mode == ExecutionMode.SINGLE:
        def objective_local(trial: optuna.trial.Trial) -> float:
            return _objective_local(
                trial=trial,
                loss=loss,
                active_seeds=active_seeds,
                entry_id=entry_id,
            )

        study.optimize(objective_local, n_trials=n_trials, n_jobs=max_concurrent_trials)
    elif resolved_mode == ExecutionMode.DISTRIBUTED:
        resolved_loss_module = loss_module or getattr(loss.seed_loss_fn, "__module__", "")
        if (not resolved_loss_module) or resolved_loss_module == "__main__":
            resolved_loss_module = loss.source_file or ""
        if not resolved_loss_module:
            raise ValueError(
                "For mode=ExecutionMode.DISTRIBUTED, provide loss_module or define loss in an importable module"
            )

        if len(active_seeds) % resolved_chunk_size != 0:
            raise ValueError("For mode=ExecutionMode.DISTRIBUTED, number of seeds must be divisible by chunk_size")

        slurm_cfg = SlurmConfig(
            poll_seconds=slurm_poll_seconds,
            timeout_minutes=slurm_timeout_minutes,
            cpus_per_task=cpus_per_task,
            array_parallelism_limit=array_parallelism_limit,
            worker_time_limit=worker_time_limit,
        )

        project_root = Path.cwd()
        python_executable = sys.executable

        def objective_slurm(trial: optuna.trial.Trial) -> float:
            params = {
                name: suggest_param(trial, name, spec)
                for name, spec in loss.parameter_space.items()
            }

            last_error: Exception | None = None
            for attempt in range(trial_retry_attempts + 1):
                summary_path = submit_trial(
                    project_root=project_root,
                    run_dir=run_dir,
                    trial_number=trial.number,
                    loss_module=resolved_loss_module,
                    loss_name=loss.name,
                    params=params,
                    entry_id=entry_id,
                    seed_start=active_seeds[0],
                    num_chunks=resolved_num_chunks,
                    chunk_size=resolved_chunk_size,
                    worker_parallelism=worker_parallelism,
                    config=slurm_cfg,
                    python_executable=python_executable,
                )

                try:
                    summary = wait_for_summary(summary_path, config=slurm_cfg)
                    trial.set_user_attr("n_seeds", summary["n_seeds"])
                    trial.set_user_attr("retry_attempts_used", attempt)
                    return float(summary["mean_total_loss"])
                except TimeoutError as exc:
                    last_error = exc
                    trial.set_user_attr("last_timeout_attempt", attempt)
                    continue

            raise TimeoutError(
                f"Trial {trial.number} failed after {trial_retry_attempts + 1} attempts"
            ) from last_error

        study.optimize(objective_slurm, n_trials=n_trials, n_jobs=max_concurrent_trials)
    else:
        raise ValueError("mode must be one of: single, distributed")

    completed_at_utc = datetime.now(timezone.utc)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "loss_name": loss.name,
                "run_name": chosen_run_name,
                "run_dir": str(run_dir),
                "mode": resolved_mode.value,
                "n_trials": n_trials,
                "entry_id": entry_id,
                "started_at_utc": started_at_utc.isoformat(),
                "completed_at_utc": completed_at_utc.isoformat(),
                "duration_seconds": (completed_at_utc - started_at_utc).total_seconds(),
                "seed_start": active_seeds[0],
                "seed_end": active_seeds[-1],
                "n_seeds": len(active_seeds),
                "chunk_size": resolved_chunk_size,
                "num_chunks": resolved_num_chunks,
                "max_concurrent_trials": max_concurrent_trials,
                "array_parallelism_limit": array_parallelism_limit,
                "worker_parallelism": worker_parallelism,
                "worker_time_limit_seconds": int(worker_time_limit.total_seconds()),
                "best_value": float(study.best_value),
                "best_params": dict(study.best_params),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_value": float(study.best_value),
                "best_params": dict(study.best_params),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return OptimizeResult(
        loss_name=loss.name,
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        n_trials=n_trials,
        study_name=study.study_name,
        run_dir=str(run_dir),
        mode=resolved_mode,
    )


def optimize(
    loss: LossDefinition,
    *,
    n_trials: int = 10,
    seeds: Iterable[int] | None = None,
    n_seeds: int | None = None,
    chunk_size: int | None = None,
    num_chunks: int | None = None,
    random_seed: int = 123,
    entry_id: str | None = None,
    direction: str = "minimize",
) -> OptimizeResult:
    """Backward-compatible local optimizer helper."""
    return optimize_run(
        loss,
        n_trials=n_trials,
        seeds=seeds,
        n_seeds=n_seeds,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        random_seed=random_seed,
        entry_id=entry_id,
        direction=direction,
        mode=ExecutionMode.SINGLE,
    )


def _sanitize_entry_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return label or "entry"


def optimize_entries(
    loss: LossDefinition,
    *,
    entry_ids: Iterable[str],
    n_trials: int = 10,
    seeds: Iterable[int] | None = None,
    n_seeds: int | None = None,
    chunk_size: int | None = None,
    num_chunks: int | None = None,
    random_seed: int = 123,
    direction: str = "minimize",
    mode: ExecutionMode = ExecutionMode.SINGLE,
    run_root: str | Path = "runs",
    run_name_prefix: str | None = None,
    loss_module: str | None = None,
    slurm_poll_seconds: int = 15,
    slurm_timeout_minutes: int = 120,
    cpus_per_task: int = 1,
    max_concurrent_trials: int = 1,
    array_parallelism_limit: int | None = None,
    worker_parallelism: int = 1,
    max_concurrent_entries: int | None = None,
    worker_time_limit: timedelta = timedelta(hours=1),
    trial_retry_attempts: int = 1,
) -> MultiOptimizeResult:
    """Optimize one independent fit per entry_id and return n best parameter sets.

    This is a modeling-parallel mode: each entry gets its own optimization study.
    Cluster acceleration is still available by setting mode=ExecutionMode.DISTRIBUTED.
    """

    entry_list = [str(x) for x in entry_ids]
    if not entry_list:
        raise ValueError("entry_ids must not be empty")

    sanitized = [_sanitize_entry_label(entry) for entry in entry_list]
    if len(set(sanitized)) != len(sanitized):
        raise ValueError("entry_ids collapse to duplicate run labels after sanitization")
    if max_concurrent_entries is not None and max_concurrent_entries <= 0:
        raise ValueError("max_concurrent_entries must be > 0 when provided")

    resolved_mode = _coerce_mode(mode)

    run_root_path = Path(run_root)
    if run_name_prefix is not None:
        parent_name = run_name_prefix
    else:
        parent_name = _next_versioned_run_name(run_root_path, f"{loss.name}_entries")
    parent_dir = run_root_path / parent_name
    parent_dir.mkdir(parents=True, exist_ok=True)

    def _run_one(idx: int, entry: str) -> tuple[str, OptimizeResult]:
        result = optimize_run(
            loss,
            n_trials=n_trials,
            seeds=seeds,
            n_seeds=n_seeds,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            random_seed=random_seed + idx,
            entry_id=entry,
            direction=direction,
            mode=resolved_mode,
            run_root=parent_dir,
            run_name=sanitized[idx],
            loss_module=loss_module,
            slurm_poll_seconds=slurm_poll_seconds,
            slurm_timeout_minutes=slurm_timeout_minutes,
            cpus_per_task=cpus_per_task,
            max_concurrent_trials=max_concurrent_trials,
            array_parallelism_limit=array_parallelism_limit,
            worker_parallelism=worker_parallelism,
            worker_time_limit=worker_time_limit,
            trial_retry_attempts=trial_retry_attempts,
        )
        return entry, result

    workers = min(len(entry_list), max_concurrent_entries or len(entry_list))
    results_by_entry: dict[str, OptimizeResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, idx, entry) for idx, entry in enumerate(entry_list)]
        for fut in as_completed(futures):
            entry, result = fut.result()
            results_by_entry[entry] = result

    best_params_by_entry = {k: dict(v.best_params) for k, v in results_by_entry.items()}
    best_values_by_entry = {k: v.best_value for k, v in results_by_entry.items()}

    (parent_dir / "summary.json").write_text(
        json.dumps(
            {
                "entries": entry_list,
                "best_params_by_entry": best_params_by_entry,
                "best_values_by_entry": best_values_by_entry,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return MultiOptimizeResult(
        loss_name=loss.name,
        n_entries=len(entry_list),
        entries=entry_list,
        results_by_entry=results_by_entry,
        best_values_by_entry=best_values_by_entry,
        best_params_by_entry=best_params_by_entry,
        mode=resolved_mode,
        run_dir=str(parent_dir),
    )
