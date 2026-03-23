# slurptuna

Run Optuna hyperparameter optimization on Slurm (HPC clusters) without boilerplate.

slurptuna is a simple way to run Optuna on Slurm clusters without writing sbatch scripts or managing distributed workers.

Running Optuna in distributed mode on a Slurm cluster (HPC) normally requires writing
custom job scripts, managing shared storage, and coordinating distributed workers manually.
slurptuna wraps this behind a single function call.

## How slurptuna runs Optuna on Slurm

You write a **loss function** decorated with `@loss`, then call `optimize_run`.
slurptuna handles:

- Submitting per-trial chunk array jobs via `sbatch`
- Collecting and aggregating seed-level results
- Storing Optuna trial state in a local SQLite database
- Writing a `summary.json` with the best result when done

## Install

```bash
pip install slurptuna
```

Or with uv:

```bash
uv add slurptuna
```

## Next steps

- [Quickstart](quickstart.md) — write your first loss and run it
- [Distributed Mode](distributed.md) — how chunk/reduce jobs work, what knobs to turn, and benchmark numbers
- [Participant-wise Fitting](participant-wise.md) — fitting one parameter set per participant

Full docs: [younesstrittmatter.github.io/slurptuna](https://younesstrittmatter.github.io/slurptuna)
