from __future__ import annotations

from pathlib import Path
import time
import subprocess

import pytest

from slurptuna.slurm_backend import ChunkExecutionError, SlurmConfig, find_missing_chunks, submit_trial, wait_for_summary


def test_find_missing_chunks(tmp_path: Path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()

    (chunks / "chunk_00000.json").write_text("{}", encoding="utf-8")
    (chunks / "chunk_00002.json").write_text("{}", encoding="utf-8")

    missing = find_missing_chunks(chunks, 4)
    assert missing == [1, 3]


def test_wait_for_summary_times_out_when_jobs_have_not_reported_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summary_path = tmp_path / "summary.json"
    config = SlurmConfig(timeout_minutes=0, poll_seconds=0, fail_on_chunk_error=True)

    monkeypatch.setattr("slurptuna.slurm_backend._get_job_states", lambda job_id: [])
    monkeypatch.setattr(time, "time", lambda: 0.0)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    with pytest.raises(TimeoutError, match="Timed out waiting"):
        wait_for_summary(
            summary_path,
            config=config,
            chunk_job_id="123",
            reduce_job_id="456",
        )


def test_wait_for_summary_fails_on_terminal_chunk_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summary_path = tmp_path / "summary.json"
    config = SlurmConfig(timeout_minutes=1, poll_seconds=0, fail_on_chunk_error=True)

    monkeypatch.setattr(
        "slurptuna.slurm_backend._get_job_states",
        lambda job_id: ["FAILED"] if job_id == "123" else [],
    )

    with pytest.raises(ChunkExecutionError, match="chunk_job_id=123"):
        wait_for_summary(
            summary_path,
            config=config,
            chunk_job_id="123",
            reduce_job_id="456",
        )


def test_submit_trial_passes_qos_to_chunk_and_reduce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run_cmd(cmd: list[str]) -> str:
        calls.append(cmd)
        return "123"

    monkeypatch.setattr("slurptuna.slurm_backend._run_cmd", fake_run_cmd)

    config = SlurmConfig(qos="short")
    submitted = submit_trial(
        project_root=tmp_path,
        run_dir=tmp_path,
        trial_number=0,
        loss_module="tests.test_api",
        loss_name="toy_conditions",
        params={"alpha": 0.1},
        entry_id=None,
        seed_start=0,
        num_chunks=1,
        chunk_size=1,
        worker_parallelism=1,
        use_processes=False,
        config=config,
        python_executable="python",
    )

    assert submitted.chunk_job_id == "123"
    assert submitted.reduce_job_id == "123"
    assert len(calls) == 2
    assert "--qos=short" in calls[0]
    assert "--qos=short" in calls[1]


def test_submit_trial_retries_without_qos_when_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run_cmd(cmd: list[str]) -> str:
        calls.append(cmd)
        if any(part == "--qos=short" for part in cmd):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return "456"

    monkeypatch.setattr("slurptuna.slurm_backend._run_cmd", fake_run_cmd)

    config = SlurmConfig(qos="short")
    submitted = submit_trial(
        project_root=tmp_path,
        run_dir=tmp_path,
        trial_number=1,
        loss_module="tests.test_api",
        loss_name="toy_conditions",
        params={"alpha": 0.2},
        entry_id=None,
        seed_start=0,
        num_chunks=1,
        chunk_size=1,
        worker_parallelism=1,
        use_processes=False,
        config=config,
        python_executable="python",
    )

    assert submitted.chunk_job_id == "456"
    assert submitted.reduce_job_id == "456"
    assert len(calls) == 4
    assert "--qos=short" in calls[0]
    assert "--qos=short" not in calls[1]
    assert "--qos=short" in calls[2]
    assert "--qos=short" not in calls[3]


def test_submit_trial_passes_module_argv_json_to_chunk_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run_cmd(cmd: list[str]) -> str:
        calls.append(cmd)
        return "789"

    monkeypatch.setattr("slurptuna.slurm_backend._run_cmd", fake_run_cmd)

    config = SlurmConfig(qos="short")
    submit_trial(
        project_root=tmp_path,
        run_dir=tmp_path,
        trial_number=2,
        loss_module="tests.test_api",
        loss_name="toy_conditions",
        params={"alpha": 0.3},
        entry_id=None,
        seed_start=0,
        num_chunks=1,
        chunk_size=1,
        worker_parallelism=1,
        use_processes=False,
        config=config,
        python_executable="python",
        module_argv=["fit.py", "revaluation"],
    )

    assert len(calls) == 2
    assert "--module-argv-json" in " ".join(calls[0])
