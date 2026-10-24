#!/usr/bin/env python3
"""Fail-fast environment checks for the StreamWeave Makefile workflow."""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_MODULES = ("numpy", "pandas", "scipy", "sklearn", "torch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--search-mode", choices=("pairwise", "full_factorial"), required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    print("StreamWeave preflight")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required by the runner.")
    if not args.runner.is_file():
        errors.append(f"Runner not found: {args.runner}")
    else:
        print(f"Runner: {args.runner}")

    imported: dict[str, object] = {}
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            imported[module_name] = module
            version = getattr(module, "__version__", "version unavailable")
            print(f"Dependency OK: {module_name} {version}")
        except Exception as exc:  # report broken binary/version installs clearly
            errors.append(f"Could not import {module_name}: {exc!r}")

    torch = imported.get("torch")
    cuda_ok = False
    if torch is not None:
        try:
            cuda_ok = bool(torch.cuda.is_available())
            if cuda_ok:
                print(f"CUDA: available — {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA: not available to this PyTorch environment")
        except Exception as exc:
            warnings.append(f"Could not query CUDA cleanly: {exc!r}")
        if args.require_cuda and not cuda_ok:
            errors.append("REQUIRE_CUDA=1, but PyTorch cannot access CUDA. Install/verify CUDA-enabled PyTorch and the NVIDIA driver, or explicitly run with REQUIRE_CUDA=0.")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("NVIDIA driver sees: " + ", ".join(x.strip() for x in result.stdout.splitlines() if x.strip()))
        elif args.require_cuda and not cuda_ok:
            warnings.append("nvidia-smi did not confirm an NVIDIA GPU (command unavailable or driver problem).")
    except (OSError, subprocess.TimeoutExpired):
        if args.require_cuda and not cuda_ok:
            warnings.append("nvidia-smi is unavailable; CUDA could not be verified independently.")

    for directory, label in ((args.data_dir, "data"), (args.results_dir, "results")):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".streamweave_check_", dir=directory, delete=True):
                pass
            print(f"{label.title()} directory writable: {directory}")
        except Exception as exc:
            errors.append(f"{label.title()} directory is not writable ({directory}): {exc!r}")

    try:
        free_bytes = shutil.disk_usage(args.results_dir).free
        free_gib = free_bytes / (1024**3)
        print(f"Free disk at results location: {free_gib:.1f} GiB")
        if free_gib < 10:
            warnings.append("Less than 10 GiB is free. Dataset downloads and experiment outputs may require more space.")
    except OSError as exc:
        warnings.append(f"Could not inspect free disk space: {exc!r}")

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not datasets:
        errors.append("No datasets are configured.")
    if "N-BaIoT" in datasets:
        roots = [args.data_dir / name for name in ("N-BaIoT", "NBaIoT", "nbaiot", "n_baiot")]
        has_csv = any(root.exists() and any(root.rglob("*.csv")) for root in roots)
        if not has_csv:
            warnings.append(f"N-BaIoT was selected but no CSV files were found under {args.data_dir}/N-BaIoT (or supported directory-name variants). The dataset needs to be placed locally before that dataset can run.")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Search mode: {args.search_mode}")
    if args.search_mode == "full_factorial":
        print("Screening size: 1,470 candidate rows per dataset with the current factor lists (210 neural triples × 7 combiners).")
        print("Estimated screening rows across configured datasets: " + f"{1470 * len(datasets):,}")
    else:
        print("Screening mode: pairwise-covering design (faster, but not exhaustive over all 3-way neural combinations).")

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print(f"Preflight failed with {len(errors)} error(s).", file=sys.stderr)
        return 2
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
