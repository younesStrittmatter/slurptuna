from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


@dataclass(frozen=True)
class SlurmConfig:
    poll_seconds: int = 15
    timeout_minutes: int = 120
    cpus_per_task: int = 1
    array_parallelism_limit: int | None = None
    worker_time_limit: timedelta = timedelta(hours=1)  # --time passed to chunk/reduce sbatch jobs
    sbatch_executable: str = "sbatch"


def _to_slurm_time_limit(value: timedelta) -> str:
    total_seconds = int(value.total_seconds())
    if total_seconds <= 0:
        raise ValueError("worker_time_limit must be > 0 seconds")
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _extract_job_id(sbatch_output: str) -> str:
    parts = sbatch_output.strip().split()
    if not parts:
        raise RuntimeError(f"Could not parse sbatch output: {sbatch_output!r}")
    return parts[-1]


def _run_cmd(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, text=True)
    return out.strip()


def find_missing_chunks(chunks_dir: Path, num_chunks: int) -> list[int]:
    missing: list[int] = []
    for idx in range(num_chunks):
        if not (chunks_dir / f"chunk_{idx:05d}.json").exists():
            missing.append(idx)
    return missing


def _array_spec(indices: list[int]) -> str:
    if not indices:
        raise ValueError("indices must not be empty")
    if len(indices) == 1:
        return str(indices[0])

    spans: list[str] = []
    start = indices[0]
    prev = indices[0]
    for cur in indices[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        spans.append(f"{start}-{prev}" if start != prev else str(start))
        start = cur
        prev = cur
    spans.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(spans)


def submit_trial(
    *,
    project_root: Path,
    run_dir: Path,
    trial_number: int,
    loss_module: str,
    loss_name: str,
    params: dict[str, float],
    entry_id: str | None,
    seed_start: int,
    num_chunks: int,
    chunk_size: int,
    worker_parallelism: int,
    config: SlurmConfig,
    python_executable: str,
) -> Path:
    trial_dir = run_dir / "trials" / f"trial_{trial_number:05d}"
    chunks_dir = trial_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "slurm_logs" / f"trial_{trial_number:05d}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    params_json = trial_dir / "params.json"
    params_json.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")

    summary_path = trial_dir / "summary.json"
    slurm_time_limit = _to_slurm_time_limit(config.worker_time_limit)

    chunk_cmd = (
        f"cd {shlex.quote(str(project_root))} && "
        f"{shlex.quote(python_executable)} -m slurptuna.worker chunk "
        f"--loss-module {shlex.quote(loss_module)} "
        f"--loss-name {shlex.quote(loss_name)} "
        f"--params-json {shlex.quote(str(params_json))} "
        f"--out-dir {shlex.quote(str(chunks_dir))} "
        f"--seed-start {seed_start} "
        f"--chunk-size {chunk_size} "
        f"--workers {worker_parallelism} "
        f"--entry-id {shlex.quote(entry_id or '')}"
    )

    missing_chunks = find_missing_chunks(chunks_dir, num_chunks)
    chunk_job_id: str | None = None
    if missing_chunks:
        array_spec = _array_spec(missing_chunks)
        if config.array_parallelism_limit is not None:
            array_spec = f"{array_spec}%{config.array_parallelism_limit}"

        chunk_submit = [
            config.sbatch_executable,
            f"--array={array_spec}",
            f"--cpus-per-task={config.cpus_per_task}",
            f"--time={slurm_time_limit}",
            "--output",
            str(logs_dir / "chunk_%A_%a.out"),
            "--error",
            str(logs_dir / "chunk_%A_%a.err"),
            "--parsable",
            "--wrap",
            chunk_cmd,
        ]
        chunk_job_id = _extract_job_id(_run_cmd(chunk_submit))

    reduce_cmd = (
        f"cd {shlex.quote(str(project_root))} && "
        f"{shlex.quote(python_executable)} -m slurptuna.worker reduce "
        f"--chunks-dir {shlex.quote(str(chunks_dir))} "
        f"--expected-chunks {num_chunks} "
        f"--out-json {shlex.quote(str(summary_path))}"
    )

    reduce_submit = [config.sbatch_executable]
    if chunk_job_id is not None:
        reduce_submit.append(f"--dependency=afterok:{chunk_job_id}")
    reduce_submit.extend([
        f"--cpus-per-task={config.cpus_per_task}",
        f"--time={slurm_time_limit}",
        "--output",
        str(logs_dir / "reduce_%j.out"),
        "--error",
        str(logs_dir / "reduce_%j.err"),
        "--parsable",
        "--wrap",
        reduce_cmd,
    ])
    _run_cmd(reduce_submit)

    return summary_path


def wait_for_summary(summary_path: Path, *, config: SlurmConfig) -> dict[str, object]:
    deadline = time.time() + (config.timeout_minutes * 60)
    while True:
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for {summary_path}")
        time.sleep(config.poll_seconds)
