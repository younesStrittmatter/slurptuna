from __future__ import annotations

from .registry import loss


@loss(
    name="toy_conditions",
    description="Toy loss over two conditions",
    parameter_space={
        "alpha": (0.0, 1.0),
        "beta": (0.0, 1.0),
    },
    default_num_chunks=1,
    default_chunk_size=8,
)
def toy_conditions(params: dict[str, float], seed: int, context: dict[str, object]):
    _ = context
    return {
        "condition_a": abs(params["alpha"] - 0.3) + (seed % 3) * 0.01,
        "condition_b": abs(params["beta"] - 0.7) + (seed % 5) * 0.01,
    }
