# slurptuna – Run Optuna on Slurm (HPC hyperparameter optimization made simple)
> Run Optuna hyperparameter optimization on Slurm clusters without writing sbatch scripts or managing distributed workers.

Running Optuna on a Slurm cluster (HPC) is not straightforward. `slurptuna` provides a simple way to run Optuna on Slurm with minimal setup.

In practice, running Optuna on Slurm clusters usually means:
- writing and managing `sbatch` job arrays
- coordinating distributed Optuna trials 
- aggregating results across workers

While Optuna supports distributed optimization, integrating it with Slurm
typically requires custom orchestration.

`slurptuna` removes that overhead by handling job submission, parallel execution,
and result aggregation automatically.

## Install

```bash
pip install slurptuna
```

Or with uv:

```bash
uv add slurptuna
```

## Usage

Here is a minimal example of running Optuna on Slurm using `slurptuna`:

### (1) Write your loss function in a script

```python
# my_model.py
from datetime import timedelta
from slurptuna import execution_mode, loss, optimize_run

@loss(
    name="my_model",
    description="Fit alpha/beta",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def my_model(params, seed):
    return abs(params["alpha"] - 0.3) + abs(params["beta"] - 0.7)

if __name__ == "__main__":
    result = optimize_run(
        my_model,
        mode=execution_mode("distributed"),
        n_trials=20,
        n_seeds=400,
        chunk_size=20,
        worker_time_limit=timedelta(hours=2),
        slurm_qos="short",
    )
    print(result.best_params)
    # best params and best value are also written to runs/my_model_v0001/summary.json
```

Distributed defaults are tuned for common short-queue clusters:
- `worker_time_limit=timedelta(hours=2)`
- `slurm_qos="short"`

If your cluster uses a different QoS (or none), set `slurm_qos` accordingly. `slurptuna`
automatically retries submission without `--qos` if the specified QoS is rejected.

### Parameter space

Loss functions may accept either `(params, seed)` or `(params, seed, context)`.
Most users can ignore `context`; it is reserved for framework-provided metadata.

Tuple shorthand is interpreted as `(min, max)`:

```python
parameter_space={"alpha": (0.0, 1.0)}
```

You can also use explicit specs when needed:

```python
from slurptuna import search_param

parameter_space={
    "alpha": search_param(range=(0.0, 1.0)),
    "steps": search_param(range=(1, 10), dtype="int"),
    "mode": search_param(allowed=["fast", "slow"]),
}
```

### (2) Submit your script as a long-running controller job on Slurm:

```bash
sbatch run_controller.sh my_model.py
```

`run_controller.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=slurptuna-controller
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source .venv/bin/activate
python "$1"
```

The controller submits and monitors chunk/reduce array jobs automatically —
you just wait for the result.

### Dynamic script styles (`sys.argv`) in distributed mode

`slurptuna` forwards the launcher's `sys.argv` to worker imports by default, so
argv-driven dynamic loss naming works across Slurm jobs.

Important: the loss still must be registered when the module is imported. A loss
defined only inside `if __name__ == "__main__":` is not visible to workers.

## Performance

On a real Slurm cluster (`short` QoS, 4-CPU tasks), slurptuna scales seed throughput dramatically:

| Case | Seeds | Wall time | Seeds/s | Speedup |
|---|---|---|---|---|
| single, 1 worker (baseline) | 100,000 | 84 s | 1,191 | 1× |
| single, 4 processes | 400,000 | 89 s | 4,494 | 3.8× |
| **distributed, 100 tasks × 4 processes** | **40,000,000** | **399 s** | **100,251** | **84×** |

The distributed case evaluates 40 million seeds in ~7 minutes — work that would take ~9 hours sequentially.

See [`benchmark/`](benchmark/) for the full benchmark setup and loss function.

## Docs

[younesstrittmatter.github.io/slurptuna](https://younesstrittmatter.github.io/slurptuna)
