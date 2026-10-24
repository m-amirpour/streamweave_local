#!/usr/bin/env python3
"""Summarize resumable StreamWeave outputs and optionally validate completion."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def csv_stats(results_dir: Path) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                rows = sum(1 for _ in reader)
            stats.append({"file": path.name, "rows": rows, "bytes": path.stat().st_size, "columns": len(header)})
        except (OSError, csv.Error) as exc:
            stats.append({"file": path.name, "rows": None, "bytes": path.stat().st_size if path.exists() else None, "error": str(exc)})
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--expected-datasets", default="")
    parser.add_argument("--since-epoch", type=float, default=None)
    parser.add_argument("--strict", action="store_true", help="return non-zero unless a fresh manifest shows every expected dataset completed")
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    manifest_path = results_dir / "FINAL_MANIFEST.json"
    plan_path = results_dir / "EXPERIMENT_PLAN.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else None
    plan = load_json(plan_path) if plan_path.is_file() else None
    expected = [x.strip() for x in args.expected_datasets.split(",") if x.strip()]
    if not expected and plan:
        expected = [str(x) for x in plan.get("datasets", [])]

    completed = manifest.get("completed", []) if manifest else []
    failures = manifest.get("failures", []) if manifest else []
    if not isinstance(completed, list):
        completed = []
    if not isinstance(failures, list):
        failures = [{"error": "Manifest 'failures' field has an invalid shape."}]
    completed_names = {str(item.get("dataset")) for item in completed if isinstance(item, dict) and item.get("dataset") is not None}
    missing = [name for name in expected if name not in completed_names]
    if manifest_path.is_file():
        manifest_mtime = manifest_path.stat().st_mtime
        if args.since_epoch is not None:
            fresh = manifest_mtime >= args.since_epoch
        elif plan_path.is_file():
            # If a new plan was written after the last manifest, the last run does
            # not correspond to the most recently previewed configuration.
            fresh = manifest_mtime >= plan_path.stat().st_mtime
        else:
            fresh = True
    else:
        fresh = False

    per_dataset: list[dict[str, Any]] = []
    run_complete = True
    for name in expected or sorted(completed_names):
        record = next((x for x in completed if isinstance(x, dict) and str(x.get("dataset")) == name), None)
        if not record:
            per_dataset.append({"dataset": name, "status": "MISSING/FAILED"})
            run_complete = False
            continue
        final_runs = record.get("final_test_runs_successful")
        expected_seeds = record.get("final_test_seed_count_expected")
        baselines_ok = record.get("all_prespecified_baseline_seed_runs_successful") is True
        done = final_runs is not None and expected_seeds is not None and int(final_runs) == int(expected_seeds) and baselines_ok
        if not done:
            run_complete = False
        per_dataset.append({
            "dataset": name,
            "status": "COMPLETE" if done else "INCOMPLETE",
            "condition": record.get("condition"),
            "screening_successful": record.get("screening_successful"),
            "screening_planned": record.get("screening_planned"),
            "selected": record.get("final_selected"),
            "test_f1_mean": record.get("final_test_f1_mean"),
            "test_f1_std": record.get("final_test_f1_std"),
            "test_pr_auc_mean": record.get("final_test_pr_auc_mean"),
            "final_test_runs_successful": final_runs,
            "final_test_seed_count_expected": expected_seeds,
            "baselines_complete": baselines_ok,
        })

    csv_summary = csv_stats(results_dir) if results_dir.exists() else []
    status = {
        "results_dir": str(results_dir),
        "generated_at_epoch": time.time(),
        "manifest_exists": manifest_path.is_file(),
        "manifest_fresh_for_current_invocation": fresh,
        "search_mode": (manifest or plan or {}).get("search_mode"),
        "device": (manifest or plan or {}).get("device"),
        "expected_datasets": expected,
        "completed_datasets": sorted(completed_names),
        "missing_datasets": missing,
        "manifest_failures": failures,
        "run_complete": bool(manifest and fresh and run_complete and not failures and not missing),
        "dataset_summaries": per_dataset,
        "csv_files": csv_summary,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "run_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")

    lines = [
        "# StreamWeave Results Overview",
        "",
        f"- Results directory: `{results_dir}`",
        f"- Search mode: `{status.get('search_mode') or 'unknown'}`",
        f"- Device: `{status.get('device') or 'unknown'}`",
        f"- Fresh final manifest: **{'yes' if fresh else 'no'}**",
        f"- Overall complete: **{'yes' if status['run_complete'] else 'no'}**",
        "",
    ]
    if per_dataset:
        lines += [
            "## Dataset status",
            "",
            "| Dataset | Status | Screening | Test F1 mean ± SD | Test PR-AUC mean | Final seeds |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for rec in per_dataset:
            screening = "—"
            if rec.get("screening_successful") is not None and rec.get("screening_planned") is not None:
                screening = f"{rec['screening_successful']}/{rec['screening_planned']}"
            f1, sd, pr = rec.get("test_f1_mean"), rec.get("test_f1_std"), rec.get("test_pr_auc_mean")
            f1_text = "—" if f1 is None else f"{float(f1):.4f} ± {float(sd):.4f}" if sd is not None else f"{float(f1):.4f} ± —"
            pr_text = "—" if pr is None else f"{float(pr):.4f}"
            seeds = "—"
            if rec.get("final_test_runs_successful") is not None and rec.get("final_test_seed_count_expected") is not None:
                seeds = f"{rec['final_test_runs_successful']}/{rec['final_test_seed_count_expected']}"
            lines.append(f"| {rec.get('dataset', '—')} | {rec.get('status', '—')} | {screening} | {f1_text} | {pr_text} | {seeds} |")
        lines.append("")
    if failures:
        lines += ["## Dataset failures", ""]
        for failure in failures:
            if isinstance(failure, dict):
                lines.append(f"- **{failure.get('dataset', 'unknown')}**: {failure.get('error', 'error details unavailable')}")
            else:
                lines.append(f"- {failure}")
        lines.append("")
    if csv_summary:
        lines += ["## Saved CSV outputs", "", "| File | Rows | Size |", "|---|---:|---:|"]
        for item in csv_summary:
            size = item.get("bytes")
            size_text = "—" if size is None else f"{size / (1024**2):.2f} MiB"
            rows = "—" if item.get("rows") is None else f"{item['rows']:,}"
            lines.append(f"| `{item['file']}` | {rows} | {size_text} |")
        lines.append("")
    lines += [
        "## Interpretation note",
        "",
        "This overview reports saved runner outputs; it does not certify scientific validity or publication readiness. Inspect the experiment plan, dataset fingerprints, failure logs, and protocol assumptions before interpreting results.",
        "",
    ]
    (results_dir / "RESULTS_OVERVIEW.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Results directory: {results_dir}")
    print(f"Final manifest: {'found' if manifest else 'not found/invalid'} (fresh={fresh})")
    print(f"Run complete: {'yes' if status['run_complete'] else 'no'}")
    for item in per_dataset:
        print(f"  {item['dataset']}: {item['status']}")
    if failures:
        print(f"Manifest dataset failures: {len(failures)}")
    print("Wrote RESULTS_OVERVIEW.md and run_status.json")

    if args.strict:
        problems: list[str] = []
        if not manifest:
            problems.append("FINAL_MANIFEST.json is missing or invalid")
        if not fresh:
            problems.append("FINAL_MANIFEST.json was not written during this run")
        if missing:
            problems.append("missing/failed datasets: " + ", ".join(missing))
        if failures:
            problems.append(f"manifest records {len(failures)} dataset failure(s)")
        incomplete = [rec["dataset"] for rec in per_dataset if rec.get("status") != "COMPLETE"]
        if incomplete:
            problems.append("incomplete final evaluation or baselines: " + ", ".join(incomplete))
        if problems:
            for problem in problems:
                print("ERROR: " + problem, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
