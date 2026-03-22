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
from slurptuna import ExecutionMode, loss, optimize_run

@loss(
    name="my_model",
    description="Fit alpha/beta",
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

## Docs

[younesstrittmatter.github.io/slurptuna](https://younesstrittmatter.github.io/slurptuna)
