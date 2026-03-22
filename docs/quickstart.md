# Quickstart

## 1. Write your loss function

Create a Python script with a `@loss`-decorated function.
The function must accept `params` (the current candidate) and `seed` (an integer).
It may also accept an optional `context` dict when you need framework-provided
metadata.
Return a scalar loss — lower is better.

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
    # Replace this with your real model evaluation.
    return abs(params["alpha"] - 0.3) + abs(params["beta"] - 0.7)

if __name__ == "__main__":
    result = optimize_run(
        my_model,
    mode=execution_mode("distributed"),
        n_trials=20,
        n_seeds=400,
        chunk_size=20,
        worker_time_limit=timedelta(minutes=30),
    )
    print(result.best_params)
    print(result.best_value)
```

If you need framework-provided metadata, add a third argument:

```python
def my_model(params, seed, context):
    entry_id = context.get("entry_id")
    ...
```

### Parameter space options
  
The tuple shorthand is interpreted as `(min, max)`:

```python
parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)}
```

You can use explicit specs when needed:

```python
from slurptuna import search_param
parameter_space={
  "alpha": search_param(range=(0.0, 1.0)),
  "steps": search_param(range=(1, 10), dtype="int"),
  "mode": search_param(allowed=["fast", "slow"]),
}
```


## 2. Create a controller script

```bash
# run_controller.sh
#!/bin/bash
#SBATCH --job-name=slurptuna-controller
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source .venv/bin/activate
python "$1"
```

## 3. Submit

```bash
sbatch run_controller.sh my_model.py
```

The controller job runs for the full duration of the optimization.
It submits chunk and reduce jobs per trial, collects results, and updates
the Optuna study. You can monitor progress with `squeue --me`.

## Output

Results land in `runs/my_model_v0001/` (auto-versioned):

```
runs/my_model_v0001/
  summary.json     ← best_value + best_params
  meta.json        ← run config + timing
  optuna.db        ← full Optuna study
  trials/
    trial_00000/
      params.json
      summary.json
      chunks/
```

`summary.json` at the top level contains the final answer:

```json
{
  "best_value": 0.012,
  "best_params": { "alpha": 0.301, "beta": 0.698 }
}
```

## Test locally first

Use `execution_mode("single")` to run everything in-process before submitting:

```python
result = optimize_run(
    my_model,
    mode=execution_mode("single"),
    n_trials=5,
    n_seeds=20,
)
```
