# benchmark/

Demonstrates the parallel scaling power of slurptuna across three execution
setups using a single probabilistic loss function.

## The loss

`bench_scaling_demo` (`loss_definitions.py`) — a noisy two-parameter loss where
each seed models one participant/observation.  The per-seed signal is swamped by
Gaussian noise (`σ=0.5`), so you need a large number of seeds to recover a
stable mean loss.  This is the canonical use case slurptuna is built for.

## Files

| File | Purpose |
|---|---|
| `loss_definitions.py` | All `@loss` definitions including `scaling_demo_loss` |
| `run_parallel_scaling_demo.py` | Three-case parallel scaling benchmark |
| `run_parallel_scaling_demo.sh` | SLURM submission script |

## Quick start

From the repo root:

```bash
sbatch benchmark/run_parallel_scaling_demo.sh
```

Or directly on a compute node (4 CPUs):

```bash
PYTHONPATH=. python benchmark/run_parallel_scaling_demo.py
```

## The three cases

| Case | Mode | Seeds | Workers |
|---|---|---|---|
| `single_100k_threads1` | single | 100,000 | 1 (sequential baseline) |
| `single_400k_process4` | single | 400,000 | 4 processes (4× seeds, ~same wall time) |
| `distributed_40M_array100_process4` | distributed | 40,000,000 | 100 Slurm tasks × 4 processes |

## Expected output

Actual results on a Slurm cluster (`short` QoS, 4-CPU tasks):

```
case                                  mode         seeds    wall_s   seeds/s  vs_base_tp  proj_base_s
single_100k                           single       100000       84      1191        1.00          84
single_400k                           single       400000       89      4494        3.78         336
distributed_40M_array100_process4     distributed 40000000      399    100251       84.21       33613
```

- `vs_base_tp` — throughput speedup relative to the sequential baseline.
- `proj_base_s` — projected wall time if the same seed count ran sequentially at baseline throughput.
