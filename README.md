# slurptuna

Optuna hyperparameter optimization on Slurm, without the boilerplate.

## Install

```bash
uv sync
```

## Usage

Write your loss function in a script:

```python
# my_model.py
from datetime import timedelta
from slurptuna import ExecutionMode, loss, optimize_run

@loss(
    name="my_model",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def my_model(params, seed, context):
    return abs(params["alpha"] - 0.3) + abs(params["beta"] - 0.7)

if __name__ == "__main__":
    result = optimize_run(
        my_model,
        mode=ExecutionMode.DISTRIBUTED,
        n_trials=20,
        n_seeds=400,
        chunk_size=20,
        worker_time_limit=timedelta(minutes=30),
    )
    print(result.best_params)
    # best params and best value are also written to runs/my_model_v0001/summary.json
```

Submit your script as a long-running controller job on Slurm:

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

## Docs

See the [docs/](docs/) folder, or run locally:

```bash
uv run mkdocs serve
```
