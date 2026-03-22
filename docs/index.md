# slurptuna

Optuna hyperparameter optimization on Slurm, without the boilerplate.

Running Optuna in distributed mode on a Slurm cluster normally requires writing
custom job scripts, managing shared storage, and coordinating workers manually.
slurptuna wraps all of that behind a single function call.

## How it works

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
- [Distributed Mode](distributed.md) — how chunk/reduce jobs work and what knobs to turn
- [Participant-wise Fitting](participant-wise.md) — fitting one parameter set per participant

Full docs: [younesstrittmatter.github.io/surptuna](https://younesstrittmatter.github.io/surptuna)
