# Distributed Mode

When you set `mode=execution_mode("distributed")`, slurptuna runs each Optuna trial
as a pair of Slurm jobs instead of in-process.

## How a trial works

For each trial Optuna proposes, slurptuna:

1. Writes `params.json` with the candidate parameters
2. Submits a **chunk array job** — one array task per `chunk_size` seeds
3. Each task runs your loss over its seed range and writes a `chunk_XXXXX.json`
4. Submits a dependent **reduce job** that aggregates chunks into `summary.json`
5. Reads the mean loss from `summary.json` and reports it back to Optuna

The controller sits in a polling loop between steps 4 and 5.

## Key parameters

| Parameter | What it controls |
|---|---|
| `n_trials` | Number of Optuna trials (candidate parameter sets to evaluate) |
| `n_seeds` | Total seeds evaluated per trial |
| `chunk_size` | Seeds per array task |
| `max_concurrent_trials` | Trials running in parallel inside one study |
| `worker_parallelism` | Threads per chunk task |
| `cpus_per_task` | CPUs allocated to each chunk task on Slurm |
| `worker_time_limit` | Wall time for chunk/reduce jobs |
| `array_parallelism_limit` | Max simultaneous array tasks (`--array %N`) |
| `trial_retry_attempts` | Retries on timeout before failing a trial |

```python
from slurptuna import execution_mode, optimize_run

result = optimize_run(
    my_model,
    mode=execution_mode("distributed"),
    n_trials=50,
    n_seeds=800,
    chunk_size=40,           # → 20 array tasks per trial
    max_concurrent_trials=4,
    worker_parallelism=4,
    cpus_per_task=4,
    worker_time_limit=timedelta(minutes=30),
    array_parallelism_limit=80,
)
```

## Run naming and versioning

If you omit `run_name`, slurptuna auto-creates versioned names:

```
runs/my_model_v0001/
runs/my_model_v0002/
```

Pass `run_name` explicitly to resume an existing run:

```python
optimize_run(my_model, run_name="my_model_v0001", ...)
```

Existing trial outputs are reused automatically — only missing chunks are resubmitted.

## Fault tolerance

- **Timeout**: a trial that times out is retried up to `trial_retry_attempts` times.
- **Partial completion**: existing `chunk_XXXXX.json` files are reused on retry.
- **Resume**: re-running with the same `run_name` picks up from the existing Optuna DB.
