from pathlib import Path
import json

from slurptuna import ExecutionMode, loss, optimize, optimize_entries, optimize_run


@loss(
    name="toy_conditions",
    description="Toy loss over two conditions",
    parameter_space={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
    default_num_chunks=1,
    default_chunk_size=8,
)
def toy_conditions(params, seed, context):
    return {
        "condition_a": abs(params["alpha"] - 0.3) + (seed % 3) * 0.01,
        "condition_b": abs(params["beta"] - 0.7) + (seed % 5) * 0.01,
    }


def test_optimize_toy_loss_runs():
    result = optimize(toy_conditions, n_trials=2, seeds=[0, 1, 2], random_seed=7)
    assert result.loss_name == "toy_conditions"
    assert result.n_trials == 2
    assert isinstance(result.best_value, float)
    assert "alpha" in result.best_params
    assert "beta" in result.best_params


def test_optimize_run_local_creates_run_dir(tmp_path: Path):
    result = optimize_run(
        toy_conditions,
        mode=ExecutionMode.SINGLE,
        n_trials=2,
        seeds=[0, 1, 2],
        random_seed=7,
        run_root=tmp_path,
        run_name="test_run",
    )
    assert result.mode == ExecutionMode.SINGLE
    assert result.run_dir is not None
    run_dir = Path(result.run_dir)
    assert run_dir.exists()
    assert (run_dir / "optuna.db").exists()
    assert (run_dir / "meta.json").exists()


@loss(
    name="toy_entries",
    description="Toy loss with independent entry targets",
    parameter_space={"alpha": (0.0, 1.0)},
    default_num_chunks=1,
    default_chunk_size=4,
)
def toy_entries(params, seed, context):
    _ = seed
    targets = {
        "p01": 0.2,
        "p02": 0.8,
    }
    entry_id = str(context.get("entry_id"))
    return abs(params["alpha"] - targets[entry_id])


def test_optimize_entries_returns_one_best_set_per_entry(tmp_path: Path):
    result = optimize_entries(
        toy_entries,
        entry_ids=["p01", "p02"],
        mode=ExecutionMode.SINGLE,
        n_trials=15,
        seeds=[0, 1],
        random_seed=11,
        run_root=tmp_path,
        run_name_prefix="entry_fit",
    )

    assert result.n_entries == 2
    assert result.entries == ["p01", "p02"]
    assert set(result.results_by_entry.keys()) == {"p01", "p02"}

    alpha_1 = result.best_params_by_entry["p01"]["alpha"]
    alpha_2 = result.best_params_by_entry["p02"]["alpha"]
    assert alpha_1 < alpha_2

    run_dir_1 = Path(result.results_by_entry["p01"].run_dir or "")
    run_dir_2 = Path(result.results_by_entry["p02"].run_dir or "")
    assert run_dir_1.exists()
    assert run_dir_2.exists()

    meta_1 = json.loads((run_dir_1 / "meta.json").read_text(encoding="utf-8"))
    meta_2 = json.loads((run_dir_2 / "meta.json").read_text(encoding="utf-8"))
    assert meta_1["entry_id"] == "p01"
    assert meta_2["entry_id"] == "p02"
