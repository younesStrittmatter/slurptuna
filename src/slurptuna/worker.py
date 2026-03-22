from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import importlib.util

from .evaluate import normalize_seed_loss, summarize_rows
from .registry import get_registered_loss


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
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_eval_seed, seeds))

    summary = summarize_rows(rows)
    summary["seed_start"] = start
    summary["seed_end"] = end
    summary["task_id"] = task_id

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"chunk_{task_id:05d}.json"
    out_path.write_text(json.dumps(summary), encoding="utf-8")


def run_reduce(args: argparse.Namespace) -> None:
    chunk_paths = sorted(args.chunks_dir.glob("chunk_*.json"))
    if len(chunk_paths) != args.expected_chunks:
        raise RuntimeError(
            f"Expected {args.expected_chunks} chunks, found {len(chunk_paths)} in {args.chunks_dir}"
        )

    rows = []
    for path in chunk_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        totals = payload.get("total_losses", [])
        seed_start = int(payload["seed_start"])
        for i, value in enumerate(totals):
            rows.append({
                "seed": seed_start + i,
                "total_loss": float(value),
                "components": {},
            })

    summary = summarize_rows(rows)
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
