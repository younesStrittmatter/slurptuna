from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .api import optimize
from .registry import get_registered_loss


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run slurptuna optimization")
    parser.add_argument("--loss-module", required=True, type=str)
    parser.add_argument("--loss-name", required=True, type=str)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--entry-id", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()

    importlib.import_module(args.loss_module)
    loss = get_registered_loss(args.loss_name)

    result = optimize(
        loss,
        n_trials=args.n_trials,
        random_seed=args.random_seed,
        entry_id=args.entry_id,
    )

    payload = {
        "loss_name": result.loss_name,
        "best_value": result.best_value,
        "best_params": result.best_params,
        "n_trials": result.n_trials,
        "study_name": result.study_name,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
