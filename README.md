# slurptuna

Simple optimization orchestration for cluster-style objective functions.

## Why

`slurptuna` keeps the user API tiny:

- define a loss
- call `optimize(loss)`
- optionally return condition/participant-wise losses as dicts or lists

## Install

```bash
uv sync
```

## Quick Start

```python
from slurptuna import loss, optimize

@loss(
    name="participantwise_loss",
    description="Example participant-wise objective",
    parameter_space={
        "alpha": (0.0, 1.0),
        "beta": (0.0, 1.0),
    },
    default_num_chunks=1,
    default_chunk_size=8,
)
def participantwise_loss(params, seed, context):
    # Return scalar, dict, or list.
    # Dict -> mean of values becomes total loss.
    return {
        "participant_01": abs(params["alpha"] - 0.2) + (seed % 3) * 0.01,
        "participant_02": abs(params["beta"] - 0.8) + (seed % 5) * 0.01,
    }

result = optimize(participantwise_loss, n_trials=20)
print(result.best_value, result.best_params)
```

## CLI

```bash
uv run slurptuna --loss-module slurptuna.example_losses --loss-name toy_conditions --n-trials 20
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```
