from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


TERMINAL_FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


class ChunkExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SlurmConfig:
    poll_seconds: int = 15
    timeout_minutes: int = 120
    cpus_per_task: int = 1
    mem_per_cpu: str = "2G"  # --mem-per-cpu passed to each chunk sbatch task
    array_parallelism_limit: int | None = None
    worker_time_limit: timedelta = timedelta(hours=2)  # --time passed to chunk/reduce sbatch jobs
    qos: str | None = "short"  # Optional --qos passed to chunk/reduce sbatch jobs
    sbatch_executable: str = "sbatch"
    fail_on_chunk_error: bool = True  # Fail immediately if any chunk fails instead of continuing


@dataclass(frozen=True)
class SubmittedTrial:
    summary_path: Path
    chunk_job_id: str | None
    reduce_job_id: str


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


def _run_sbatch_with_optional_qos_fallback(cmd: list[str], *, qos: str | None) -> str:
    try:
        return _run_cmd(cmd)
    except subprocess.CalledProcessError:
        # Some clusters reject unknown QoS names; retry once without --qos.
        if qos is None:
            raise
        cmd_without_qos = [part for part in cmd if not part.startswith("--qos=")]
        if len(cmd_without_qos) == len(cmd):
            raise
        return _run_cmd(cmd_without_qos)


def _normalize_slurm_state(raw_state: str) -> str:
    state = raw_state.strip().split()[0]
    return state.rstrip("+")


def _get_job_states(job_id: str) -> list[str]:
    try:
        out = _run_cmd(["sacct", "-n", "-P", "-j", job_id, "--format=State"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    states: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        raw_state = line.split("|", 1)[0]
        if raw_state:
            states.append(_normalize_slurm_state(raw_state))
    return states


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
    params: dict[str, object],
    entry_id: str | None,
    seed_start: int,
    num_chunks: int,
    chunk_size: int,
    worker_parallelism: int,
    use_processes: bool,
    config: SlurmConfig,
    python_executable: str,
    module_argv: list[str] | None = None,
) -> SubmittedTrial:
    trial_dir = run_dir / "trials" / f"trial_{trial_number:05d}"
    chunks_dir = trial_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "slurm_logs" / f"trial_{trial_number:05d}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    params_json = trial_dir / "params.json"
    params_json.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")

    module_argv_json: Path | None = None
    if module_argv is not None:
        module_argv_json = trial_dir / "module_argv.json"
        module_argv_json.write_text(json.dumps(module_argv), encoding="utf-8")

    summary_path = trial_dir / "summary.json"
    slurm_time_limit = _to_slurm_time_limit(config.worker_time_limit)

    chunk_cmd = (
        f"cd {shlex.quote(str(project_root))} && "
        f"{shlex.quote(python_executable)} -m slurptuna.worker chunk "
        f"--loss-module {shlex.quote(loss_module)} "
        f"--loss-name {shlex.quote(loss_name)} "
        f"--params-json {shlex.quote(str(params_json))} "
        + (
            f"--module-argv-json {shlex.quote(str(module_argv_json))} "
            if module_argv_json is not None
            else ""
        )
        +
        f"--out-dir {shlex.quote(str(chunks_dir))} "
        f"--seed-start {seed_start} "
        f"--chunk-size {chunk_size} "
        f"--workers {worker_parallelism} "
        + ("--use-processes " if use_processes else "")
        + f"--entry-id {shlex.quote(entry_id or '')}"
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
        ]
        if config.qos:
            chunk_submit.append(f"--qos={config.qos}")
        chunk_submit.append(f"--mem-per-cpu={config.mem_per_cpu}")
        chunk_submit.extend([
            "--output",
            str(logs_dir / "chunk_%A_%a.out"),
            "--error",
            str(logs_dir / "chunk_%A_%a.err"),
            "--parsable",
            "--wrap",
            chunk_cmd,
        ])
        chunk_job_id = _extract_job_id(_run_sbatch_with_optional_qos_fallback(chunk_submit, qos=config.qos))

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
        "--cpus-per-task=1",
        f"--time={slurm_time_limit}",
        f"--mem-per-cpu={config.mem_per_cpu}",
    ])
    if config.qos:
        reduce_submit.append(f"--qos={config.qos}")
    reduce_submit.extend([
        "--output",
        str(logs_dir / "reduce_%j.out"),
        "--error",
        str(logs_dir / "reduce_%j.err"),
        "--parsable",
        "--wrap",
        reduce_cmd,
    ])
    reduce_job_id = _extract_job_id(_run_sbatch_with_optional_qos_fallback(reduce_submit, qos=config.qos))

    return SubmittedTrial(
        summary_path=summary_path,
        chunk_job_id=chunk_job_id,
        reduce_job_id=reduce_job_id,
    )


def wait_for_summary(
    summary_path: Path,
    *,
    config: SlurmConfig,
    chunk_job_id: str | None = None,
    reduce_job_id: str | None = None,
) -> dict[str, object]:
    deadline = time.time() + (config.timeout_minutes * 60)
    while True:
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))

        if config.fail_on_chunk_error:
            if chunk_job_id is not None:
                chunk_states = _get_job_states(chunk_job_id)
                failed_chunk_states = sorted({state for state in chunk_states if state in TERMINAL_FAILURE_STATES})
                if failed_chunk_states:
                    raise ChunkExecutionError(
                        "Chunk job failed before producing a summary. "
                        f"chunk_job_id={chunk_job_id}, states={failed_chunk_states}."
                    )

            if reduce_job_id is not None:
                reduce_states = _get_job_states(reduce_job_id)
                failed_reduce_states = sorted({state for state in reduce_states if state in TERMINAL_FAILURE_STATES})
                if failed_reduce_states:
                    raise ChunkExecutionError(
                        "Reduce job failed before producing a summary. "
                        f"reduce_job_id={reduce_job_id}, states={failed_reduce_states}."
                    )

        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for {summary_path}")
        time.sleep(config.poll_seconds)
