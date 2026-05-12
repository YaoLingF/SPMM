#!/usr/bin/env python3
"""Benchmark src2/bench_mtx on SuiteSparse MatrixMarket files.

The script runs the standalone static-matrix benchmark for every `.mtx` file
under `data/suitesparse_388`, parses cuSPARSE and custom SpMM timings, and
writes a CSV that marks whether the custom kernel is faster.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "suitesparse_388"
DEFAULT_BENCH = ROOT / "src2" / "bench_mtx"
DEFAULT_OUTPUT = ROOT / "results" / "suitesparse_388_static_bench.csv"

GPU_RE = re.compile(r"gpu:\s*(?P<gpu>.+?)\s+sm_(?P<sm>\d+)")
MATRIX_RE = re.compile(
    r"matrix:\s*M=(?P<M>\d+)\s+K=(?P<K>\d+)\s+nnz=(?P<nnz>\d+)\s+N=(?P<N>\d+)\s+iter=(?P<iter>\d+)"
)
ERROR_RE = re.compile(
    r"max_abs=(?P<max_abs>[-+0-9.eE]+)\s+max_rel=(?P<max_rel>[-+0-9.eE]+)"
)
CUSPARSE_RE = re.compile(r"cuSPARSE:\s*(?P<ms>[-+0-9.eE]+)\s*ms")
OUR_RE = re.compile(r"our\s*:\s*(?P<ms>[-+0-9.eE]+)\s*ms")
VALIDATION_RE = re.compile(r"validation:\s*(?P<validation>\w+)")


FIELDNAMES = [
    "matrix_id",
    "group",
    "name",
    "path",
    "N",
    "iter",
    "M",
    "K",
    "nnz",
    "avg_nnz_per_row",
    "gpu",
    "sm",
    "cusparse_ms",
    "our_ms",
    "speedup",
    "our_faster",
    "validation",
    "max_abs",
    "max_rel",
    "status",
    "returncode",
    "wall_s",
    "command",
    "message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run src2/bench_mtx on data/suitesparse_388 and save CSV results."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--bench", default=str(DEFAULT_BENCH))
    parser.add_argument(
        "--features",
        default="256",
        help="comma-separated dense feature sizes N, e.g. 64,128,256,512",
    )
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--summary-output",
        default="",
        help="summary CSV path; default is <output>.summary.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="run only the first N matrices after sorting; 0 means all",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per benchmark command timeout in seconds",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip matrix/N pairs already present in the output CSV",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="do not run make when src2/bench_mtx is missing",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="run make -C src2 before benchmarking",
    )
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="continue after failed benchmarks",
    )
    return parser.parse_args()


def parse_features(value: str) -> list[int]:
    features = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not features:
        raise ValueError("--features must contain at least one positive integer")
    bad = [item for item in features if item <= 0]
    if bad:
        raise ValueError(f"feature sizes must be positive: {bad}")
    return features


def matrix_id(path: Path, data_dir: Path) -> str:
    return path.relative_to(data_dir).with_suffix("").as_posix()


def discover_matrices(data_dir: Path, limit: int) -> list[Path]:
    matrices = sorted(data_dir.rglob("*.mtx"))
    if limit > 0:
        matrices = matrices[:limit]
    return matrices


def ensure_bench(bench: Path, rebuild: bool, no_build: bool) -> None:
    if rebuild or not bench.exists():
        if no_build:
            if not bench.exists():
                raise FileNotFoundError(f"missing benchmark binary: {bench}")
            return
        subprocess.run(["make", "-C", str(ROOT / "src2")], check=True)
    if not bench.exists():
        raise FileNotFoundError(f"missing benchmark binary after build: {bench}")


def existing_keys(output: Path) -> set[tuple[str, str]]:
    if not output.exists():
        return set()
    with output.open(newline="") as f:
        rows = csv.DictReader(f)
        return {(row["path"], row["N"]) for row in rows}


def parse_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for regex in (GPU_RE, MATRIX_RE, ERROR_RE, CUSPARSE_RE, OUR_RE, VALIDATION_RE):
        match = regex.search(stdout)
        if not match:
            continue
        if regex is CUSPARSE_RE:
            parsed["cusparse_ms"] = match.group("ms")
        elif regex is OUR_RE:
            parsed["our_ms"] = match.group("ms")
        else:
            parsed.update(match.groupdict())
    return parsed


def tail_message(stdout: str, stderr: str, max_chars: int = 800) -> str:
    text = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    text = text.replace("\r", "")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_one(bench: Path, mtx: Path, data_dir: Path, feature: int, n_iter: int, timeout: float) -> dict[str, str]:
    cmd = [str(bench), str(mtx), str(feature), str(n_iter)]
    started = time.perf_counter()
    group = mtx.relative_to(data_dir).parts[0] if len(mtx.relative_to(data_dir).parts) > 1 else ""
    name = mtx.stem
    row: dict[str, str] = {
        "matrix_id": matrix_id(mtx, data_dir),
        "group": group,
        "name": name,
        "path": str(mtx.relative_to(ROOT)),
        "N": str(feature),
        "iter": str(n_iter),
        "M": "",
        "K": "",
        "nnz": "",
        "avg_nnz_per_row": "",
        "gpu": "",
        "sm": "",
        "cusparse_ms": "",
        "our_ms": "",
        "speedup": "",
        "our_faster": "",
        "validation": "",
        "max_abs": "",
        "max_rel": "",
        "status": "error",
        "returncode": "",
        "wall_s": "",
        "command": " ".join(cmd),
        "message": "",
    }

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT / "src2"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        row["returncode"] = str(completed.returncode)
        parsed = parse_output(completed.stdout)
        row.update({key: str(value) for key, value in parsed.items()})
        row["message"] = tail_message(completed.stdout, completed.stderr)

        if row["M"] and row["nnz"]:
            m = int(row["M"])
            nnz = int(row["nnz"])
            row["avg_nnz_per_row"] = f"{(nnz / m) if m else 0.0:.6f}"

        validation = row.get("validation", "")
        if row["cusparse_ms"] and row["our_ms"]:
            cusparse_ms = float(row["cusparse_ms"])
            our_ms = float(row["our_ms"])
            speedup = cusparse_ms / our_ms if our_ms > 0 else math.inf
            row["speedup"] = f"{speedup:.6f}"
            row["our_faster"] = "1" if validation == "passed" and speedup > 1.0 else "0"

        if completed.returncode == 0 and validation == "passed":
            row["status"] = "passed"
        elif validation == "failed" or completed.returncode == 2:
            row["status"] = "validation_failed"
        else:
            row["status"] = "error"
    except subprocess.TimeoutExpired as exc:
        row["status"] = "timeout"
        row["returncode"] = ""
        row["message"] = tail_message(exc.stdout or "", exc.stderr or f"timeout after {timeout}s")
    finally:
        row["wall_s"] = f"{time.perf_counter() - started:.6f}"

    return row


def append_row(output: Path, row: dict[str, str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output.exists()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def write_summary(output: Path, summary_output: Path) -> None:
    if not output.exists():
        return

    rows: list[dict[str, str]] = []
    with output.open(newline="") as f:
        rows = list(csv.DictReader(f))

    by_feature: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_feature.setdefault(row["N"], []).append(row)

    summary_rows: list[dict[str, str]] = []
    for feature, part in sorted(by_feature.items(), key=lambda item: int(item[0])):
        passed = [row for row in part if row["status"] == "passed"]
        speedups = [float(row["speedup"]) for row in passed if row["speedup"]]
        wins = [row for row in passed if row["our_faster"] == "1"]
        summary_rows.append(
            {
                "N": feature,
                "total": str(len(part)),
                "passed": str(len(passed)),
                "validation_failed": str(sum(row["status"] == "validation_failed" for row in part)),
                "errors": str(sum(row["status"] == "error" for row in part)),
                "timeouts": str(sum(row["status"] == "timeout" for row in part)),
                "our_faster": str(len(wins)),
                "win_rate_pct": f"{100.0 * len(wins) / len(passed):.2f}" if passed else "",
                "geomean_speedup": f"{math.exp(sum(math.log(x) for x in speedups) / len(speedups)):.6f}" if speedups else "",
                "mean_speedup": f"{statistics.fmean(speedups):.6f}" if speedups else "",
                "median_speedup": f"{statistics.median(speedups):.6f}" if speedups else "",
            }
        )

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    bench = Path(args.bench).resolve()
    output = Path(args.output).resolve()
    summary_output = (
        Path(args.summary_output).resolve()
        if args.summary_output
        else output.with_suffix(output.suffix + ".summary.csv")
    )
    features = parse_features(args.features)

    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    ensure_bench(bench, args.rebuild, args.no_build)
    matrices = discover_matrices(data_dir, args.limit)
    if not matrices:
        raise FileNotFoundError(f"no .mtx files found under {data_dir}")

    done = existing_keys(output) if args.resume else set()
    total_jobs = len(matrices) * len(features)
    job_idx = 0
    failures = 0

    print(f"matrices={len(matrices)} features={features} total_jobs={total_jobs}")
    print(f"output={output}")
    for mtx in matrices:
        for feature in features:
            job_idx += 1
            rel = mtx.relative_to(data_dir)
            key = (str(mtx.relative_to(ROOT)), str(feature))
            if key in done:
                print(f"[{job_idx}/{total_jobs}] skip {rel} N={feature}")
                continue

            print(f"[{job_idx}/{total_jobs}] run {rel} N={feature}", flush=True)
            row = run_one(bench, mtx, data_dir, feature, args.iter, args.timeout)
            append_row(output, row)
            if row["status"] != "passed":
                failures += 1
                print(f"  status={row['status']} returncode={row['returncode']}")
                if not args.keep_going:
                    write_summary(output, summary_output)
                    return 1
            else:
                print(
                    f"  cuSPARSE={row['cusparse_ms']} ms our={row['our_ms']} ms "
                    f"speedup={row['speedup']} faster={row['our_faster']}"
                )

    write_summary(output, summary_output)
    print(f"summary={summary_output}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
