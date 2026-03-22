from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

from slurptuna import execution_mode, loss
from slurptuna.api import optimize_entries


DATA_CSV = Path(__file__).with_name("participant_truth.csv")


def load_participants(csv_path: Path) -> dict[str, dict[str, float]]:
    participants: dict[str, dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row["participant_id"])
            participants[pid] = {
                "alpha": float(row["alpha_true"]),
                "beta": float(row["beta_true"]),
            }
    if not participants:
        raise ValueError("CSV has no participant rows")
    return participants


PARTICIPANTS = load_participants(DATA_CSV)


@loss(
    name="participant_from_csv_demo",
    description="Fit one alpha/beta set per participant loaded from CSV",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def participant_from_csv_demo(params, seed, context):
    entry_id = str(context.get("entry_id"))
    truth = PARTICIPANTS[entry_id]
    seed_term = (seed % 23) * 1e-4
    return abs(params["alpha"] - truth["alpha"]) + abs(params["beta"] - truth["beta"]) + seed_term


if __name__ == "__main__":
    print("loaded participants from:", DATA_CSV)
    print("participant ids:", list(PARTICIPANTS.keys()))

    # LOCAL / single-mode example.
    # result = optimize_entries(
    #     participant_from_csv_demo,
    #     entry_ids=PARTICIPANTS.keys(),
    #     mode=execution_mode("single"),
    #     n_trials=20,
    #     n_seeds=200,
    #     run_name_prefix="participant_from_csv_single",
    # )

    # DISTRIBUTED example.
    result = optimize_entries(
        participant_from_csv_demo,
        entry_ids=PARTICIPANTS.keys(),
        mode=execution_mode("distributed"),
        n_trials=4,
        n_seeds=80,
        chunk_size=20,
        run_name_prefix="participant_from_csv_distributed",
        max_concurrent_entries=4,
        max_concurrent_trials=2,
        worker_parallelism=2,
        cpus_per_task=2,
        worker_time_limit=timedelta(minutes=30),
    )

    print("mode:", result.mode.value)
    print("best params by participant:")
    for pid in result.entries:
        print(pid, result.best_params_by_entry[pid])
