#!/usr/bin/env python3
"""Measure the peak GPU and host memory one training task needs.

The harness launches a training command, samples memory once a second, and
reports the peak. It attributes GPU memory to the launched process tree
through ``nvidia-smi``'s per-process view, so a second job on the same GPU
does not inflate the number. It still refuses to start when another compute
process is already resident, because a shared GPU changes allocator
behaviour and the comparison would not be honest.

Example:

    pixi run python scripts/bench/measure_task_memory.py \\
        --label sharpa --num-envs 128 -- \\
        pixi run -e isaaclab python scripts/rlopt/train_newton.py ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Name for this measurement.")
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Append JSON here.")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Measure even when another compute process already holds the GPU.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("Pass the training command after `--`.")
    return args


def _nvidia_smi(query: str, extra: list[str] | None = None) -> list[str]:
    command = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"]
    result = subprocess.run(command + (extra or []), capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def compute_processes() -> dict[int, int]:
    """Return ``{pid: gpu_mib}`` for every process holding GPU memory."""

    processes: dict[int, int] = {}
    for line in _nvidia_smi("compute-apps=pid,used_memory"):
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            processes[int(parts[0])] = int(parts[1])
    return processes


def gpu_total_used() -> int:
    values = _nvidia_smi("gpu=memory.used")
    return int(values[0]) if values else 0


def descendants(root: int) -> set[int]:
    """Return the pid set of ``root`` and everything below it."""

    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name can contain spaces, so parse after the closing paren.
        fields = stat[stat.rfind(")") + 2 :].split()
        parent = int(fields[1])
        children.setdefault(parent, []).append(int(entry.name))
    found = {root}
    queue = [root]
    while queue:
        current = queue.pop()
        for child in children.get(current, ()):
            if child not in found:
                found.add(child)
                queue.append(child)
    return found


def host_rss_mib(pids: set[int]) -> int:
    total = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total // 1024


def main() -> int:
    args = parse_args()
    baseline_processes = compute_processes()
    baseline_gpu = gpu_total_used()
    if baseline_processes and not args.allow_busy_gpu:
        raise SystemExit(
            "Another process already holds GPU memory "
            f"({baseline_processes}); the measurement would not be comparable. "
            "Wait for it, or pass --allow-busy-gpu."
        )

    started = time.time()
    process = subprocess.Popen(
        args.command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    peak_gpu = 0
    peak_rss = 0
    samples = 0
    try:
        while process.poll() is None:
            time.sleep(args.interval)
            tree = descendants(process.pid)
            resident = compute_processes()
            gpu = sum(mib for pid, mib in resident.items() if pid in tree)
            if gpu == 0 and resident:
                # The launcher may exec a process outside the visible tree;
                # fall back to everything that appeared after the baseline.
                gpu = sum(
                    mib
                    for pid, mib in resident.items()
                    if pid not in baseline_processes
                )
            peak_gpu = max(peak_gpu, gpu)
            peak_rss = max(peak_rss, host_rss_mib(tree))
            samples += 1
    finally:
        if process.poll() is None:  # pragma: no cover - defensive
            process.terminate()
    record = {
        "label": args.label,
        "num_envs": args.num_envs,
        "exit_code": process.returncode,
        "peak_gpu_mib": peak_gpu,
        "peak_host_rss_mib": peak_rss,
        "baseline_gpu_mib": baseline_gpu,
        "wall_seconds": round(time.time() - started, 1),
        "samples": samples,
        "command": args.command,
    }
    print(json.dumps(record))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return 0 if process.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
