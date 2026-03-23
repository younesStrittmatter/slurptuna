from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurptuna import registry as registry_mod
from slurptuna.worker import _import_loss_module, run_chunk


@pytest.fixture(autouse=True)
def _clear_loss_registry_between_tests():
    registry_mod._REGISTRY.clear()
    yield
    registry_mod._REGISTRY.clear()


def _write_dynamic_loss_module(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import sys",
                "from slurptuna import loss",
                "",
                "TASK = sys.argv[1] if len(sys.argv) > 1 else 'fallback'",
                "",
                "@loss(",
                "    name=f'fit_to_average_person_{TASK}',",
                "    description='dynamic argv-driven loss',",
                "    parameter_space={'alpha': (0.0, 1.0)},",
                ")",
                "def fit_to_people(params, seed):",
                "    return float(seed)",
            ]
        ),
        encoding="utf-8",
    )


def test_import_loss_module_restores_sys_argv(tmp_path: Path):
    module_path = tmp_path / "dynamic_loss_module.py"
    _write_dynamic_loss_module(module_path)

    original_argv = list(sys.argv)
    _import_loss_module(str(module_path), module_argv=["fit.py", "revaluation"])
    assert sys.argv == original_argv


def test_run_chunk_supports_argv_driven_loss_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module_path = tmp_path / "dynamic_loss_module.py"
    _write_dynamic_loss_module(module_path)

    params_json = tmp_path / "params.json"
    params_json.write_text(json.dumps({"alpha": 0.5}), encoding="utf-8")

    module_argv_json = tmp_path / "module_argv.json"
    module_argv_json.write_text(json.dumps(["fit_script.py", "revaluation"]), encoding="utf-8")

    out_dir = tmp_path / "chunks"
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")

    args = SimpleNamespace(
        loss_module=str(module_path),
        loss_name="fit_to_average_person_revaluation",
        params_json=params_json,
        module_argv_json=module_argv_json,
        out_dir=out_dir,
        seed_start=0,
        chunk_size=2,
        workers=1,
        use_processes=False,
        entry_id="",
    )

    run_chunk(args)

    out_path = out_dir / "chunk_00000.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["n_seeds"] == 2
    assert payload["mean_total_loss"] == pytest.approx(0.5)


def test_run_chunk_without_forwarded_argv_may_fail_dynamic_style(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module_path = tmp_path / "dynamic_loss_module.py"
    _write_dynamic_loss_module(module_path)

    params_json = tmp_path / "params.json"
    params_json.write_text(json.dumps({"alpha": 0.5}), encoding="utf-8")

    out_dir = tmp_path / "chunks"
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")

    args = SimpleNamespace(
        loss_module=str(module_path),
        loss_name="fit_to_average_person_revaluation",
        params_json=params_json,
        module_argv_json=None,
        out_dir=out_dir,
        seed_start=0,
        chunk_size=1,
        workers=1,
        use_processes=False,
        entry_id="",
    )

    with pytest.raises(RuntimeError, match="Requested loss was not registered"):
        run_chunk(args)


def test_run_chunk_missing_loss_error_contains_debug_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module_path = tmp_path / "dynamic_loss_module.py"
    _write_dynamic_loss_module(module_path)

    params_json = tmp_path / "params.json"
    params_json.write_text(json.dumps({"alpha": 0.5}), encoding="utf-8")

    module_argv_json = tmp_path / "module_argv.json"
    module_argv_json.write_text(json.dumps(["fit_script.py", "two_step"]), encoding="utf-8")

    out_dir = tmp_path / "chunks"
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")

    args = SimpleNamespace(
        loss_module=str(module_path),
        loss_name="fit_to_average_person_revaluation",
        params_json=params_json,
        module_argv_json=module_argv_json,
        out_dir=out_dir,
        seed_start=0,
        chunk_size=1,
        workers=1,
        use_processes=False,
        entry_id="",
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_chunk(args)

    msg = str(exc_info.value)
    assert "requested='fit_to_average_person_revaluation'" in msg
    assert "forwarded_argv=['fit_script.py', 'two_step']" in msg
    assert "available=[fit_to_average_person_two_step]" in msg


def test_run_chunk_repeated_runs_with_different_argv_do_not_conflict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module_path = tmp_path / "dynamic_loss_module.py"
    _write_dynamic_loss_module(module_path)

    params_json = tmp_path / "params.json"
    params_json.write_text(json.dumps({"alpha": 0.5}), encoding="utf-8")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")

    out_dir_one = tmp_path / "chunks_one"
    argv_one = tmp_path / "argv_one.json"
    argv_one.write_text(json.dumps(["fit_script.py", "two_step"]), encoding="utf-8")
    args_one = SimpleNamespace(
        loss_module=str(module_path),
        loss_name="fit_to_average_person_two_step",
        params_json=params_json,
        module_argv_json=argv_one,
        out_dir=out_dir_one,
        seed_start=0,
        chunk_size=1,
        workers=1,
        use_processes=False,
        entry_id="",
    )
    run_chunk(args_one)

    out_dir_two = tmp_path / "chunks_two"
    argv_two = tmp_path / "argv_two.json"
    argv_two.write_text(json.dumps(["fit_script.py", "revaluation"]), encoding="utf-8")
    args_two = SimpleNamespace(
        loss_module=str(module_path),
        loss_name="fit_to_average_person_revaluation",
        params_json=params_json,
        module_argv_json=argv_two,
        out_dir=out_dir_two,
        seed_start=0,
        chunk_size=1,
        workers=1,
        use_processes=False,
        entry_id="",
    )
    run_chunk(args_two)

    payload_one = json.loads((out_dir_one / "chunk_00000.json").read_text(encoding="utf-8"))
    payload_two = json.loads((out_dir_two / "chunk_00000.json").read_text(encoding="utf-8"))
    assert payload_one["n_seeds"] == 1
    assert payload_two["n_seeds"] == 1
