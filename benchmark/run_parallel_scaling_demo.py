"""Fair parallel scaling demo using one controller that spawns sbatch case jobs.

This script supports two modes:

1) Controller mode (default): submits three case jobs via sbatch and reports wall
   time from each case job's Slurm start/end timestamps.
2) Case mode (--run-case): executes exactly one optimize_run call.

Cases:
- single_100k (single mode, 100k seeds, 1 worker)
- single_400k (single mode, 400k seeds, 4 processes)
- distributed_40M_array100_process4 (distributed mode, 40M seeds, chunk_size=400k, 4 proc/chunk)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import optuna

from benchmark.loss_definitions import scaling_demo_loss
from slurptuna import optimize_run

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class CaseSpec:
    label: str
    mode: str
    n_seeds: int
    worker_parallelism: int
    use_processes: bool
    controller_cpus: int
    controller_mem: str
    chunk_size: int | None = None


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        label="single_100k",
        mode="single",
        n_seeds=100_000,
        worker_parallelism=1,
        use_processes=False,
        controller_cpus=1,
        controller_mem="4G",
    ),
    CaseSpec(
        label="single_400k",
        mode="single",
        n_seeds=400_000,
        worker_parallelism=4,
        use_processes=True,
        controller_cpus=4,
        controller_mem="8G",
    ),
    CaseSpec(
        label="distributed_40M_array100_process4",
        mode="distributed",
        n_seeds=40_000_000,
        worker_parallelism=4,
        use_processes=True,
        chunk_size=400_000,
        controller_cpus=1,
        controller_mem="4G",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="benchmark/runs", help="Output root")
    parser.add_argument("--loss-module", default="benchmark.loss_definitions", help="Importable loss module")
    parser.add_argument("--n-trials", type=int, default=1, help="Optuna trials per case")
    parser.add_argument("--slurm-timeout-minutes", type=int, default=180, help="Timeout for distributed trial")
    parser.add_argument("--distributed-cpus-per-task", type=int, default=4, help="Chunk-task CPUs")
    parser.add_argument("--mem-per-cpu", type=str, default="2G", help="Chunk-task mem-per-cpu")
    parser.add_argument("--case-time", type=str, default="03:00:00", help="SBATCH --time for each case job")
    parser.add_argument("--controller-poll-seconds", type=int, default=5, help="Polling interval for sbatch jobs")
    parser.add_argument("--workspace", type=str, default=str(Path.cwd()), help="Workspace root for sbatch --wrap")

    parser.add_argument("--run-case", type=str, default=None, choices=[c.label for c in CASES])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--result-json", type=str, default=None)
    return parser.parse_args()


def _run_command(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _squeue_contains(job_id: str) -> bool:
    out = _run_command(["squeue", "--noheader", "--jobs", job_id, "--format", "%i"])
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return any(line == job_id for line in lines)


def _wait_for_completion(job_id: str, poll_seconds: int) -> None:
    while _squeue_contains(job_id):
        time.sleep(max(1, poll_seconds))


def _get_job_timing(job_id: str, retries: int = 24, retry_sleep: int = 5) -> tuple[str, str, int]:
    """Return (state, start, elapsed_raw_seconds) for the exact job row."""
    cmd = [
        "sacct",
        "-X",
        "--jobs",
        job_id,
        "--format",
        "JobIDRaw,State,Start,ElapsedRaw",
        "--parsable2",
        "--noheader",
    ]

    for _ in range(retries):
        out = _run_command(cmd)
        for raw in out.splitlines():
            parts = raw.split("|")
            if len(parts) < 4:
                continue
            row_job_id, state, start, elapsed_raw = (parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip())
            if row_job_id != job_id:
                continue
            if start in {"", "Unknown", "N/A"}:
                continue
            if elapsed_raw in {"", "Unknown", "N/A"}:
                continue
            return state, start, int(elapsed_raw)
        time.sleep(retry_sleep)

    raise RuntimeError(f"Could not retrieve timing for job {job_id} from sacct")


def _find_case(label: str) -> CaseSpec:
    for case in CASES:
        if case.label == label:
            return case
    raise ValueError(f"Unknown case label: {label}")


def _run_single_case(case: CaseSpec, args: argparse.Namespace) -> dict[str, object]:
    run_name = args.run_name or f"scaling_demo_{case.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    kwargs: dict[str, object] = {
        "mode": case.mode,
        "n_trials": args.n_trials,
        "n_seeds": case.n_seeds,
        "worker_parallelism": case.worker_parallelism,
        "use_processes": case.use_processes,
        "run_root": args.run_root,
        "run_name": run_name,
    }

    if case.mode == "distributed":
        kwargs.update(
            {
                "chunk_size": case.chunk_size,
                "loss_module": args.loss_module,
                "cpus_per_task": args.distributed_cpus_per_task,
                "mem_per_cpu": args.mem_per_cpu,
                "slurm_timeout_minutes": args.slurm_timeout_minutes,
                "slurm_poll_seconds": 5,
            }
        )

    result = optimize_run(scaling_demo_loss, **kwargs)

    payload = {
        "label": case.label,
        "mode": case.mode,
        "n_seeds": case.n_seeds,
        "worker_parallelism": case.worker_parallelism,
        "use_processes": case.use_processes,
        "run_name": run_name,
        "run_dir": result.run_dir,
        "best_value": result.best_value,
        "best_params": result.best_params,
    }

    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def _submit_case_job(
    *,
    case: CaseSpec,
    args: argparse.Namespace,
    stamp: str,
    workspace: Path,
    logs_dir: Path,
    controller_results_dir: Path,
) -> tuple[str, Path]:
    result_json = controller_results_dir / f"{case.label}_{stamp}.json"
    run_name = f"scaling_demo_{case.label}_{stamp}"

    python_exe = workspace / ".venv" / "bin" / "python"
    case_cmd = [
        str(python_exe),
        "benchmark/run_parallel_scaling_demo.py",
        "--run-case",
        case.label,
        "--run-name",
        run_name,
        "--result-json",
        str(result_json),
        "--run-root",
        args.run_root,
        "--loss-module",
        args.loss_module,
        "--n-trials",
        str(args.n_trials),
        "--slurm-timeout-minutes",
        str(args.slurm_timeout_minutes),
        "--distributed-cpus-per-task",
        str(args.distributed_cpus_per_task),
        "--mem-per-cpu",
        args.mem_per_cpu,
        "--workspace",
        str(workspace),
    ]

    wrapped = " && ".join(
        [
            f"cd {shlex.quote(str(workspace))}",
            f"source {shlex.quote(str(workspace / '.venv' / 'bin' / 'activate'))}",
            f"export PYTHONPATH={shlex.quote(str(workspace))}",
            shlex.join(case_cmd),
        ]
    )

    submit_cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=stuna-{case.label}",
        f"--output={logs_dir / (case.label + '_%j.out')}",
        f"--error={logs_dir / (case.label + '_%j.err')}",
        f"--time={args.case_time}",
        f"--cpus-per-task={case.controller_cpus}",
        f"--mem={case.controller_mem}",
        "--wrap",
        wrapped,
    ]

    job_id = _run_command(submit_cmd).split(";")[0].strip()
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch output for {case.label}: {job_id}")
    return job_id, result_json


def _run_controller(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace).resolve()
    logs_dir = workspace / "benchmark" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    controller_results_dir = Path(args.run_root) / f"controller_{stamp}"
    controller_results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 92)
    print("  Parallel scaling demo (fair timing via per-case sbatch job start/end)")
    print("=" * 92)
    print(f"  n_trials: {args.n_trials}")
    print(f"  distributed chunk cpus_per_task: {args.distributed_cpus_per_task}")
    print(f"  distributed chunk mem_per_cpu: {args.mem_per_cpu}")
    print()

    rows: list[dict[str, object]] = []
    for case in CASES:
        job_id, result_json = _submit_case_job(
            case=case,
            args=args,
            stamp=stamp,
            workspace=workspace,
            logs_dir=logs_dir,
            controller_results_dir=controller_results_dir,
        )
        print(f"  submitted {case.label:<28} job_id={job_id}", flush=True)
        print(f"  waiting   {case.label:<28} job_id={job_id}", flush=True)
        _wait_for_completion(job_id, poll_seconds=args.controller_poll_seconds)
        state, start, wall_s = _get_job_timing(job_id)

        if not result_json.exists():
            raise RuntimeError(f"Missing case result file: {result_json}")

        case_payload = json.loads(result_json.read_text(encoding="utf-8"))
        throughput = case.n_seeds / max(1, wall_s)

        row = {
            "label": case.label,
            "mode": case.mode,
            "job_id": job_id,
            "state": state,
            "job_start": start,
            "n_seeds": case.n_seeds,
            "workers": case.worker_parallelism,
            "wall_s": wall_s,
            "seeds_per_s": throughput,
            "run_dir": case_payload.get("run_dir"),
        }
        rows.append(row)
        print(
            f"      done {case.label:<28} state={state:<12} wall_s={wall_s:>6d}  seeds/s={throughput:>10.1f}",
            flush=True,
        )

    baseline_tp = float(rows[0]["seeds_per_s"])

    print("\n" + "=" * 92)
    print("  Results (wall_s measured from case sbatch job start->end)")
    print("=" * 92)
    print(
        f"  {'case':<28} {'mode':<12} {'job':>9} {'state':<12} {'seeds':>10} {'wall_s':>8} {'seeds/s':>12} {'vs_base_tp':>11}"
    )
    print("  " + "-" * 90)
    for row in rows:
        speedup = float(row["seeds_per_s"]) / baseline_tp
        print(
            f"  {str(row['label']):<28} {str(row['mode']):<12} {str(row['job_id']):>9} {str(row['state']):<12} "
            f"{int(row['n_seeds']):>10d} {int(row['wall_s']):>8d} {float(row['seeds_per_s']):>12.1f} {speedup:>11.2f}"
        )

    summary_path = controller_results_dir / "summary.json"
    summary_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\n  wrote summary: {summary_path}")
    print()


def main() -> None:
    args = _parse_args()

    if args.run_case:
        case = _find_case(args.run_case)
        _run_single_case(case, args)
        return

    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run this controller via sbatch so child-job timing is recorded consistently")

    _run_controller(args)


if __name__ == "__main__":
    main()
