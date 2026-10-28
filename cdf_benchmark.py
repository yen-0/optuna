"""Noise-controlled benchmark for the TPE normal CDF implementations.

Run from the repository root, passing one or more worktree paths.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def pin_process() -> None:
    if os.name != "nt":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process = kernel32.GetCurrentProcess()
    # Pin to CPU 0. This is deliberately dependency-free (no psutil needed).
    # Some hosted runners expose a pseudo process handle. In that case affinity
    # is unavailable; keep the benchmark usable and report timings normally.
    kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(1))


def child(kind: str, size: int, repeats: int, warmup: int) -> None:
    pin_process()
    import numpy as np
    from optuna.samplers._tpe import _truncnorm

    rng = np.random.default_rng(20261028)
    x = rng.uniform(-40.0, 40.0, size=size)
    fn = _truncnorm._ndtr if kind == "ndtr" else _truncnorm._log_ndtr
    for _ in range(warmup):
        fn(x)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = fn(x)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        # Prevent an over-aggressive optimizer or accidental lazy path.
        if not np.all(np.isfinite(result)):
            raise RuntimeError(f"non-finite result from {kind}")
        samples.append(elapsed)
    samples.sort()
    median = samples[len(samples) // 2]
    deviations = sorted(abs(v - median) for v in samples)
    mad = deviations[len(deviations) // 2]
    print(json.dumps({"kind": kind, "median_ms": median, "mad_ms": mad, "samples_ms": samples}))


def run_one(label: str, repo: Path, kind: str, size: int, repeats: int, warmup: int) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    command = [sys.executable, str(Path(__file__).resolve()), "--child", kind,
               "--size", str(size), "--repeats", str(repeats), "--warmup", str(warmup)]
    completed = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    result["implementation"] = label
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=["ndtr", "log_ndtr"])
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repo", action="append", nargs=2, metavar=("LABEL", "PATH"))
    args = parser.parse_args()
    if args.child:
        child(args.child, args.size, args.repeats, args.warmup)
        return
    if not args.repo:
        parser.error("at least one --repo LABEL PATH is required")
    jobs = [(label, Path(path).resolve()) for label, path in args.repo]
    results = []
    for round_index in range(3):
        order = jobs[:]
        random.Random(20261028 + round_index).shuffle(order)
        for label, repo in order:
            for kind in ("ndtr", "log_ndtr"):
                results.append({"round": round_index, **run_one(label, repo, kind, args.size, args.repeats, args.warmup)})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
