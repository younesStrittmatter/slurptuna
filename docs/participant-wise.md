# Participant-wise Fitting

Use `optimize_entries` when you want **one independent best parameter set per
participant** (or condition, subject, item — any discrete ID).

Each entry gets its own Optuna study. Studies run concurrently via
`max_concurrent_entries`.

## Example

```python
from datetime import timedelta
from slurptuna import execution_mode, loss, optimize_entries

PARTICIPANTS = {
    "p01": {"alpha": 0.20, "beta": 0.80},
    "p02": {"alpha": 0.35, "beta": 0.55},
    "p03": {"alpha": 0.60, "beta": 0.25},
}

@loss(
    name="my_model",
  description="Participant-wise fit",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def my_model(params, seed, context):
    pid = context["entry_id"]          # ← entry_id is injected automatically
    truth = PARTICIPANTS[pid]
    return abs(params["alpha"] - truth["alpha"]) + abs(params["beta"] - truth["beta"])

if __name__ == "__main__":
    result = optimize_entries(
        my_model,
        entry_ids=PARTICIPANTS.keys(),
      mode=execution_mode("distributed"),
        n_trials=20,
        n_seeds=400,
        chunk_size=20,
        run_name_prefix="my_model_participants",
        max_concurrent_entries=3,      # all three run at the same time
        max_concurrent_trials=2,
        worker_parallelism=2,
        cpus_per_task=2,
        worker_time_limit=timedelta(minutes=30),
    )

    for pid, params in result.best_params_by_entry.items():
        print(pid, params)
```

## Output structure

Per-participant runs are stored as **subfolders** of the prefix directory:

```
runs/my_model_participants/
  summary.json        ← all participants' best params in one place
  p01/
    summary.json
    meta.json
    optuna.db
    trials/
  p02/
    ...
  p03/
    ...
```

The top-level `summary.json` is the convenient final output:

```json
{
  "entries": ["p01", "p02", "p03"],
  "best_params_by_entry": {
    "p01": { "alpha": 0.201, "beta": 0.799 },
    "p02": { "alpha": 0.353, "beta": 0.548 },
    "p03": { "alpha": 0.598, "beta": 0.251 }
  },
  "best_values_by_entry": {
    "p01": 0.003,
    "p02": 0.007,
    "p03": 0.009
  }
}
```

## Loading participants from a CSV

```python
import csv
from pathlib import Path

def load_participants(path):
    with Path(path).open() as f:
        return {row["id"]: {"alpha": float(row["alpha"]), "beta": float(row["beta"])}
                for row in csv.DictReader(f)}

PARTICIPANTS = load_participants("participants.csv")
```

Then pass `entry_ids=PARTICIPANTS.keys()` as usual.

## Difference from `optimize_run` with a dict loss

`optimize_run` with a loss that returns a `dict` averages across participants to
find **one shared parameter set** that fits all participants simultaneously.
`optimize_entries` finds **separate** best params per participant.
Use whichever matches your modeling assumption.
