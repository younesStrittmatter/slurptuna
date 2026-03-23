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
| `slurm_qos` | Optional Slurm QoS passed as `--qos` (default `"short"`) |
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
    worker_time_limit=timedelta(hours=2),
    slurm_qos="short",
    array_parallelism_limit=80,
)
```

Defaults in distributed mode:
- `worker_time_limit=timedelta(hours=2)`
- `slurm_qos="short"`

## Dynamic Losses And argv

Distributed workers import your loss module in a separate process and look up the
requested loss by name.

What works:
- Dynamic loss names derived from `sys.argv` at module import time.
- Multiple runs of the same script with different argv values.
- Repeated launches do not leak argv state across runs (worker import temporarily
    overrides argv and restores it immediately after import).

What does not work:
- Registering the only needed `@loss` inside `if __name__ == "__main__":`.
    Worker imports do not execute that block.

When using `optimize_run(..., mode=execution_mode("distributed"))`, slurptuna
forwards the launcher argv to workers by default (`forward_sys_argv_to_workers=True`).
Disable this only if you explicitly do not want argv-dependent registration.

QoS names are cluster-specific. If your cluster does not define `short`, set
`slurm_qos` to the local value or `None`. Slurptuna retries once without
`--qos` if submission fails with the configured QoS.

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

## Performance

Benchmarked on a real Slurm cluster (`short` QoS, 4-CPU tasks):

| Case | Mode | Seeds | Wall time | Seeds/s | Speedup |
|---|---|---|---|---|---|
| 1 worker (baseline) | single | 100,000 | 84 s | 1,191 | 1× |
| 4 processes | single | 400,000 | 89 s | 4,494 | 3.8× |
| **100 tasks × 4 processes** | **distributed** | **40,000,000** | **399 s** | **100,251** | **84×** |

The distributed case evaluates 40 million seeds in ~7 minutes — work that would take ~9 hours sequentially.
The 84× speedup comes from Slurm parallelism (100 tasks) combined with per-task multiprocessing (4 processes each).

See [`benchmark/`](https://github.com/younesstrittmatter/slurptuna/tree/main/benchmark) for the full setup.
