from __future__ import annotations

from datetime import timedelta

from slurptuna import ExecutionMode, loss, optimize_entries

PARTICIPANTS = {
    "p01": {"alpha": 0.20, "beta": 0.80},
    "p02": {"alpha": 0.35, "beta": 0.55},
    "p03": {"alpha": 0.60, "beta": 0.25},
    "p04": {"alpha": 0.45, "beta": 0.65},
}


@loss(
    name="participant_individual_demo",
    description="Fit participant-wise: separate optimization per participant",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def participant_individual_demo(params, seed, context):
    entry_id = str(context.get("entry_id"))
    truth = PARTICIPANTS[entry_id]
    seed_term = (seed % 23) * 1e-4
    return abs(params["alpha"] - truth["alpha"]) + abs(params["beta"] - truth["beta"]) + seed_term


if __name__ == "__main__":
    # LOCAL / single-mode example.
    # result = optimize_entries(
    #     participant_individual_demo,
    #     entry_ids=PARTICIPANTS.keys(),
    #     mode=ExecutionMode.SINGLE,
    #     n_trials=20,
    #     n_seeds=200,
    #     run_name_prefix="participant_individual_single",
    # )

    # DISTRIBUTED example (one study per participant, each using Slurm chunk arrays).
    result = optimize_entries(
        participant_individual_demo,
        entry_ids=PARTICIPANTS.keys(),
        mode=ExecutionMode.DISTRIBUTED,
        n_trials=6,
        n_seeds=120,
        chunk_size=20,
        run_name_prefix="participant_individual_distributed",
        max_concurrent_entries=4,
        max_concurrent_trials=2,
        worker_parallelism=2,
        cpus_per_task=2,
        worker_time_limit=timedelta(minutes=30),
    )

    print("mode:", result.mode.value)
    print("best params by participant:")
    for pid, params in result.best_params_by_entry.items():
        print(pid, params)
