"""Shared loss definitions used by benchmark scripts.

Four losses are provided across two axes — compute style and cost per seed:

* ``cheap_loss``        – pure-Python arithmetic, ~microseconds per seed.
                         Threading will NOT help (GIL never released).
* ``slow_python_loss``  – pure-Python loop, ~5 ms per seed.
                         Processes WILL help (no GIL); threads will not.
* ``numpy_loss``        – numpy simulation, ~1 ms per seed.
                         Threading helps (numpy releases GIL).
* ``heavy_numpy_loss``  – numpy linear-algebra solve, ~5 ms per seed.
                         Threading helps; processes also work but add overhead.

The 2×2 comparison (executor_comparison benchmark) uses ``slow_python_loss``
and ``heavy_numpy_loss`` to demonstrate when processes beat threads and vice
versa.

For a benchmark where distributed is expected to win wall-clock time, use
``very_slow_python_loss`` and ``very_heavy_numpy_loss`` with many seeds.
"""

from __future__ import annotations

import numpy as np

from slurptuna import loss

# ---------------------------------------------------------------------------
# Cheap (pure-Python) loss
# ---------------------------------------------------------------------------

TRUE_CHEAP = {"alpha": 0.3, "beta": 0.7}


@loss(
    name="bench_cheap",
    description="Pure-Python arithmetic loss – used to show threading does NOT help",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def cheap_loss(params, seed):
    """Recovers alpha/beta with a tiny seed-dependent jitter."""
    base = abs(params["alpha"] - TRUE_CHEAP["alpha"]) + abs(params["beta"] - TRUE_CHEAP["beta"])
    jitter = (seed % 17) * 1e-4
    return base + jitter


# ---------------------------------------------------------------------------
# Heavy (numpy) loss
# ---------------------------------------------------------------------------

TRUE_NUMPY = {"mu": 0.3, "sigma": 0.5}


@loss(
    name="bench_numpy",
    description="Numpy-heavy loss – used to show threading DOES help (GIL released)",
    parameter_space={"mu": (-2.0, 2.0), "sigma": (0.01, 2.0)},
)
def numpy_loss(params, seed):
    """Fits mu/sigma of a Normal distribution using a simulated dataset."""
    rng = np.random.default_rng(seed)
    data = rng.normal(TRUE_NUMPY["mu"], TRUE_NUMPY["sigma"], size=5_000)
    residuals = data - params["mu"]
    return float(np.mean(residuals ** 2) + abs(np.std(data) - params["sigma"]))


# ---------------------------------------------------------------------------
# Slow pure-Python loss  (processes beat threads here)
# ---------------------------------------------------------------------------

TRUE_SLOW = {"alpha": 0.3, "beta": 0.7}


@loss(
    name="bench_slow_python",
    description="Pure-Python loop ~5 ms/seed – processes win, threads blocked by GIL",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def slow_python_loss(params, seed):
    """CPU-bound pure-Python loop with no numpy — GIL is held the entire time."""
    acc = 0.0
    for i in range(80_000):
        acc += (i % 17) * 1e-6
    base = abs(params["alpha"] - TRUE_SLOW["alpha"]) + abs(params["beta"] - TRUE_SLOW["beta"])
    return base + acc * 0.0


# ---------------------------------------------------------------------------
# Heavy numpy loss  (threads beat processes here)
# ---------------------------------------------------------------------------

TRUE_HEAVY = {"scale": 1.0}


@loss(
    name="bench_heavy_numpy",
    description="Numpy linalg solve ~5 ms/seed – threads win via GIL release, no pickle overhead",
    parameter_space={"scale": (0.1, 3.0)},
)
def heavy_numpy_loss(params, seed):
    """Solves a random linear system — heavy numpy, releases GIL throughout."""
    rng = np.random.default_rng(seed)
    size = 150
    A = rng.normal(0, 1, (size, size))
    A = A @ A.T + np.eye(size) * params["scale"]
    b = rng.normal(0, 1, size)
    x = np.linalg.solve(A, b)
    return float(np.sum(x ** 2))


# ---------------------------------------------------------------------------
# Extra-heavy losses (distributed-favoring profile)
# ---------------------------------------------------------------------------

TRUE_VERY_SLOW = {"alpha": 0.3, "beta": 0.7}


@loss(
    name="bench_very_slow_python",
    description="Pure-Python loop ~0.2-0.5 s/seed on typical CPUs; favors distributed runs",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def very_slow_python_loss(params, seed):
    """Pure-Python CPU work with substantial per-seed runtime."""
    acc = 0.0
    for i in range(3_000_000):
        acc += ((i + seed) % 17) * 1e-9
    base = abs(params["alpha"] - TRUE_VERY_SLOW["alpha"]) + abs(params["beta"] - TRUE_VERY_SLOW["beta"])
    return base + acc * 0.0


TRUE_VERY_HEAVY = {"scale": 1.0}


@loss(
    name="bench_very_heavy_numpy",
    description="Large numpy linalg solve; heavier per-seed compute to amortize scheduler overhead",
    parameter_space={"scale": (0.1, 3.0)},
)
def very_heavy_numpy_loss(params, seed):
    """Heavier numpy workload than heavy_numpy_loss for distributed comparisons."""
    rng = np.random.default_rng(seed)
    size = 350
    A = rng.normal(0, 1, (size, size))
    A = A @ A.T + np.eye(size) * params["scale"]
    b = rng.normal(0, 1, size)
    x = np.linalg.solve(A, b)
    return float(np.sum(x ** 2))


# ---------------------------------------------------------------------------
# Scaling demo loss (single function for apples-to-apples throughput demos)
# ---------------------------------------------------------------------------

TRUE_SCALING = {"alpha": 0.37, "beta": 0.72}


@loss(
    name="bench_scaling_demo",
    description="Probabilistic loss used for parallel/distributed scaling demos",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
)
def scaling_demo_loss(params, seed):
    """Noisy loss: each seed draws a random sample so more seeds → lower variance.

    Simulates a participant-level model fit. The noise term is large enough that
    individual seeds are unreliable; only averaging over many seeds gives a
    stable estimate, which is the whole reason slurptuna exists.
    """
    rng = __import__("random").Random(seed)
    noise = rng.gauss(0.0, 0.5)
    # Burn some CPU so parallelism is clearly visible in wall time
    acc = 0.0
    for i in range(10_000):
        acc += ((i + seed) % 17) * 1e-8
    true_alpha = TRUE_SCALING["alpha"]
    true_beta = TRUE_SCALING["beta"]
    signal = (params["alpha"] - true_alpha) ** 2 + (params["beta"] - true_beta) ** 2
    return signal + noise
