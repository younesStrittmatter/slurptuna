from __future__ import annotations

import argparse
import importlib
import json
from datetime import timedelta
from pathlib import Path

from .api import ExecutionMode, optimize_entries, optimize_run
from .registry import get_registered_loss


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run slurptuna optimization")
    parser.add_argument("--loss-module", required=True, type=str)
    parser.add_argument("--loss-name", required=True, type=str)
    parser.add_argument("--mode", default="single", choices=["single", "entries"])
    parser.add_argument("--execution-mode", default="single", choices=["single", "distributed"])
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--entry-id", type=str, default=None)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--slurm-poll-seconds", type=int, default=15)
    parser.add_argument("--slurm-timeout-minutes", type=int, default=120)
    parser.add_argument("--slurm-qos", type=str, default="short")
    parser.add_argument("--trial-retry-attempts", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem-per-cpu", type=str, default="2G")
    parser.add_argument("--max-concurrent-trials", type=int, default=1)
    parser.add_argument("--array-parallelism-limit", type=int, default=None)
    parser.add_argument("--worker-parallelism", type=int, default=1)
    parser.add_argument("--entry-ids", nargs="*", default=None)
    parser.add_argument("--max-concurrent-entries", type=int, default=None)
    parser.add_argument("--run-name-prefix", type=str, default=None)
    parser.add_argument("--worker-time-limit-seconds", type=int, default=7200)
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    return parser


def _seconds_to_timedelta(value: int) -> timedelta:
    if value <= 0:
        raise ValueError("--worker-time-limit-seconds must be > 0")
    return timedelta(seconds=value)


def main() -> None:
    args = build_parser().parse_args()

    importlib.import_module(args.loss_module)
    loss = get_registered_loss(args.loss_name)
    execution_mode = ExecutionMode(args.execution_mode)
    worker_time_limit = _seconds_to_timedelta(args.worker_time_limit_seconds)
    slurm_qos = args.slurm_qos if args.slurm_qos else None

    if args.mode == "single":
        result = optimize_run(
            loss,
            mode=execution_mode,
            n_trials=args.n_trials,
            n_seeds=args.n_seeds,
            chunk_size=args.chunk_size,
            num_chunks=args.num_chunks,
            random_seed=args.random_seed,
            entry_id=args.entry_id,
            run_root=args.run_root,
            run_name=args.run_name,
            loss_module=args.loss_module,
            slurm_poll_seconds=args.slurm_poll_seconds,
            slurm_timeout_minutes=args.slurm_timeout_minutes,
            trial_retry_attempts=args.trial_retry_attempts,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            max_concurrent_trials=args.max_concurrent_trials,
            array_parallelism_limit=args.array_parallelism_limit,
            worker_parallelism=args.worker_parallelism,
            worker_time_limit=worker_time_limit,
            slurm_qos=slurm_qos,
        )

        payload = {
            "loss_name": result.loss_name,
            "best_value": result.best_value,
            "best_params": result.best_params,
            "n_trials": result.n_trials,
            "study_name": result.study_name,
            "run_dir": result.run_dir,
            "mode": result.mode.value,
        }
    else:
        if not args.entry_ids:
            raise ValueError("--entry-ids is required when --mode entries")

        result = optimize_entries(
            loss,
            entry_ids=args.entry_ids,
            mode=execution_mode,
            n_trials=args.n_trials,
            n_seeds=args.n_seeds,
            chunk_size=args.chunk_size,
            num_chunks=args.num_chunks,
            random_seed=args.random_seed,
            run_root=args.run_root,
            run_name_prefix=args.run_name_prefix,
            loss_module=args.loss_module,
            slurm_poll_seconds=args.slurm_poll_seconds,
            slurm_timeout_minutes=args.slurm_timeout_minutes,
            trial_retry_attempts=args.trial_retry_attempts,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            max_concurrent_trials=args.max_concurrent_trials,
            array_parallelism_limit=args.array_parallelism_limit,
            worker_parallelism=args.worker_parallelism,
            max_concurrent_entries=args.max_concurrent_entries,
            worker_time_limit=worker_time_limit,
            slurm_qos=slurm_qos,
        )

        payload = {
            "loss_name": result.loss_name,
            "n_entries": result.n_entries,
            "entries": result.entries,
            "best_values_by_entry": result.best_values_by_entry,
            "best_params_by_entry": result.best_params_by_entry,
            "mode": result.mode.value,
            "run_dirs_by_entry": {
                entry: result.results_by_entry[entry].run_dir for entry in result.entries
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
