from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import importlib.util

from .evaluate import normalize_seed_loss, summarize_rows
from .registry import get_registered_loss


_CHUNK_LOSS = None
_CHUNK_PARAMS: dict[str, object] | None = None
_CHUNK_ENTRY_ID: str | None = None


def _chunk_eval_seed(seed: int) -> dict[str, object]:
    if _CHUNK_LOSS is None or _CHUNK_PARAMS is None:
        raise RuntimeError("chunk worker state not initialized")
    return normalize_seed_loss(
        seed,
        _CHUNK_LOSS.evaluate_seed_loss(
            _CHUNK_PARAMS,
            seed,
            {"entry_id": _CHUNK_ENTRY_ID},
        ),
    )


def _import_loss_module(module_str: str) -> None:
    """Import by module name or by file path (ends in .py)."""
    if module_str.endswith(".py"):
        spec = importlib.util.spec_from_file_location("_slurptuna_loss", module_str)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load loss module from file: {module_str}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        __import__(module_str)


def _chunk_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("slurptuna-worker chunk")
    p.add_argument("--loss-module", required=True)
    p.add_argument("--loss-name", required=True)
    p.add_argument("--params-json", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed-start", required=True, type=int)
    p.add_argument("--chunk-size", required=True, type=int)
    p.add_argument("--workers", default=1, type=int)
    p.add_argument("--use-processes", action="store_true", default=False)
    p.add_argument("--entry-id", default="")
    return p


def _reduce_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("slurptuna-worker reduce")
    p.add_argument("--chunks-dir", required=True, type=Path)
    p.add_argument("--expected-chunks", required=True, type=int)
    p.add_argument("--out-json", required=True, type=Path)
    return p


def run_chunk(args: argparse.Namespace) -> None:
    _import_loss_module(args.loss_module)
    loss = get_registered_loss(args.loss_name)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    start = args.seed_start + (task_id * args.chunk_size)
    end = start + args.chunk_size
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")

    params = json.loads(args.params_json.read_text(encoding="utf-8"))

    def _eval_seed(seed: int) -> dict[str, object]:
        return normalize_seed_loss(
            seed,
            loss.evaluate_seed_loss(
                params,
                seed,
                {
                    "entry_id": args.entry_id or None,
                },
            ),
        )

    seeds = range(start, end)
    if args.workers == 1:
        rows = [_eval_seed(seed) for seed in seeds]
    elif args.use_processes:
        global _CHUNK_LOSS, _CHUNK_PARAMS, _CHUNK_ENTRY_ID
        _CHUNK_LOSS = loss
        _CHUNK_PARAMS = params
        _CHUNK_ENTRY_ID = args.entry_id or None
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=args.workers) as pool:
            rows = pool.map(_chunk_eval_seed, seeds)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_eval_seed, seeds))

    summary = summarize_rows(rows)
    summary["seed_start"] = start
    summary["seed_end"] = end
    summary["task_id"] = task_id
    summary["params"] = params

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"chunk_{task_id:05d}.json"
    out_path.write_text(json.dumps(summary), encoding="utf-8")


def run_reduce(args: argparse.Namespace) -> None:
    chunk_paths = sorted(args.chunks_dir.glob("chunk_*.json"))
    if len(chunk_paths) != args.expected_chunks:
        raise RuntimeError(
            f"Expected {args.expected_chunks} chunks, found {len(chunk_paths)} in {args.chunks_dir}"
        )

    # Aggregate chunk-level statistics directly to avoid building per-seed rows in memory.
    total_n = 0
    weighted_mean_sum = 0.0
    chunk_stats: list[tuple[int, float, float]] = []

    for path in chunk_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunk_n = int(payload["n_seeds"])
        chunk_mean = float(payload["mean_total_loss"])
        chunk_std = float(payload["std_total_loss"])
        chunk_var = chunk_std * chunk_std

        total_n += chunk_n
        weighted_mean_sum += chunk_n * chunk_mean
        chunk_stats.append((chunk_n, chunk_mean, chunk_var))

    if total_n <= 0:
        raise RuntimeError("No seeds found while reducing chunk summaries")

    global_mean = weighted_mean_sum / total_n
    global_var = sum(
        chunk_n * (chunk_var + (chunk_mean - global_mean) ** 2)
        for chunk_n, chunk_mean, chunk_var in chunk_stats
    ) / total_n
    global_std = global_var ** 0.5

    summary = {
        "n_seeds": total_n,
        "mean_total_loss": float(global_mean),
        "std_total_loss": float(global_std),
        "stderr_total_loss": float(global_std / (total_n ** 0.5)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("slurptuna-worker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("chunk", parents=[_chunk_parser()], add_help=False)
    sub.add_parser("reduce", parents=[_reduce_parser()], add_help=False)

    args = parser.parse_args()
    if args.cmd == "chunk":
        run_chunk(args)
        return
    run_reduce(args)


if __name__ == "__main__":
    main()
