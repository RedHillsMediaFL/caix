#!/usr/bin/env python3
"""Validate caix quant-quality task fixtures and raw run evidence.

This script is intentionally no-load: it reads local TSV/JSON artifacts only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


TASK_HEADER = ["prompt_id", "slice", "grader", "prompt", "expected_answer", "notes"]
RESULT_HEADER = ["prompt_id", "slice", "expected_answer", "candidate_answer", "pass"]
REQUIRED_SLICES = {"knowledge", "math_reasoning", "instruction_following"}
ALLOWED_GRADERS = {"exact", "numeric_exact", "contains", "json_exact"}
REQUIRED_METADATA_KEYS = {
    "caix_commit",
    "model_repo",
    "model_revision",
    "reference_bundle",
    "candidate_bundle",
    "hardware",
    "os_build",
    "command",
}
REQUIRED_PPL_KEYS = {
    "token_count",
    "reference_perplexity",
    "candidate_perplexity",
    "relative_delta",
    "pass",
}
REQUIRED_SUMMARY_KEYS = {"gate_ids", "aggregate_scores", "thresholds", "final_decision"}
ALLOWED_DECISIONS = {"pass", "needs-test", "blocked", "comparator-only"}


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"{path} is empty")
        rows = list(reader)
        return reader.fieldnames, rows


def load_json(path: Path) -> object:
    try:
        with path.open() as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"{path} is not valid JSON: {exc}")


def validate_tasks(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        fail(f"task fixture not found: {path}")

    header, rows = load_tsv(path)
    if header != TASK_HEADER:
        fail(f"{path} header must be: {' '.join(TASK_HEADER)}")
    if len(rows) != 100:
        fail(f"{path} must contain exactly 100 tasks, found {len(rows)}")

    seen: dict[str, dict[str, str]] = {}
    slice_counts = {name: 0 for name in REQUIRED_SLICES}
    for line_number, row in enumerate(rows, start=2):
        prompt_id = row["prompt_id"]
        if prompt_id in seen:
            fail(f"{path}:{line_number} duplicate prompt_id: {prompt_id}")
        if not prompt_id.startswith("qt-"):
            fail(f"{path}:{line_number} prompt_id must start with qt-")
        if row["slice"] not in REQUIRED_SLICES:
            fail(f"{path}:{line_number} unknown slice: {row['slice']}")
        if row["grader"] not in ALLOWED_GRADERS:
            fail(f"{path}:{line_number} unknown grader: {row['grader']}")
        if not row["prompt"] or not row["expected_answer"]:
            fail(f"{path}:{line_number} prompt and expected_answer are required")
        slice_counts[row["slice"]] += 1
        seen[prompt_id] = row

    if slice_counts["math_reasoning"] < 20:
        fail("task fixture must include at least 20 math_reasoning prompts")
    if slice_counts["knowledge"] < 30:
        fail("task fixture must include at least 30 knowledge prompts")
    if slice_counts["instruction_following"] < 30:
        fail("task fixture must include at least 30 instruction_following prompts")

    return seen


def require_object(value: object, path: Path) -> dict[str, object]:
    if not isinstance(value, dict):
        fail(f"{path} must contain a JSON object")
    return value


def require_keys(value: dict[str, object], keys: set[str], path: Path) -> None:
    missing = sorted(key for key in keys if key not in value)
    if missing:
        fail(f"{path} missing required keys: {', '.join(missing)}")


def require_positive_number(value: object, label: str, path: Path) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        fail(f"{path} field {label} must be a positive number")


def parse_pass(value: str, path: Path, line_number: int) -> bool:
    if value in {"true", "pass"}:
        return True
    if value in {"false", "fail"}:
        return False
    fail(f"{path}:{line_number} pass must be true/false/pass/fail")


def validate_run(run_dir: Path, tasks: dict[str, dict[str, str]]) -> None:
    if not run_dir.is_dir():
        fail(f"run directory not found: {run_dir}")

    metadata_path = run_dir / "metadata.json"
    ppl_path = run_dir / "quant_ppl.json"
    tasks_path = run_dir / "quant_tasks.tsv"
    summary_path = run_dir / "summary.json"
    for path in [metadata_path, ppl_path, tasks_path, summary_path]:
        if not path.is_file():
            fail(f"required artifact missing: {path}")

    metadata = require_object(load_json(metadata_path), metadata_path)
    require_keys(metadata, REQUIRED_METADATA_KEYS, metadata_path)

    ppl = require_object(load_json(ppl_path), ppl_path)
    require_keys(ppl, REQUIRED_PPL_KEYS, ppl_path)
    require_positive_number(ppl["token_count"], "token_count", ppl_path)
    require_positive_number(ppl["reference_perplexity"], "reference_perplexity", ppl_path)
    require_positive_number(ppl["candidate_perplexity"], "candidate_perplexity", ppl_path)
    if not isinstance(ppl["relative_delta"], (int, float)) or isinstance(ppl["relative_delta"], bool):
        fail(f"{ppl_path} field relative_delta must be numeric")
    if ppl["relative_delta"] < 0:
        fail(f"{ppl_path} field relative_delta must be non-negative")
    if not isinstance(ppl["pass"], bool):
        fail(f"{ppl_path} field pass must be boolean")

    header, result_rows = load_tsv(tasks_path)
    if header != RESULT_HEADER:
        fail(f"{tasks_path} header must be: {' '.join(RESULT_HEADER)}")
    if len(result_rows) != len(tasks):
        fail(f"{tasks_path} must contain {len(tasks)} result rows, found {len(result_rows)}")

    seen_results: set[str] = set()
    result_slices: set[str] = set()
    slice_totals = {name: 0 for name in REQUIRED_SLICES}
    slice_passed = {name: 0 for name in REQUIRED_SLICES}
    failed_prompts: list[str] = []
    for line_number, row in enumerate(result_rows, start=2):
        prompt_id = row["prompt_id"]
        if prompt_id not in tasks:
            fail(f"{tasks_path}:{line_number} unknown prompt_id: {prompt_id}")
        if prompt_id in seen_results:
            fail(f"{tasks_path}:{line_number} duplicate prompt_id: {prompt_id}")
        if row["slice"] != tasks[prompt_id]["slice"]:
            fail(f"{tasks_path}:{line_number} slice does not match fixture for {prompt_id}")
        if row["expected_answer"] != tasks[prompt_id]["expected_answer"]:
            fail(f"{tasks_path}:{line_number} expected_answer does not match fixture for {prompt_id}")
        if not row["candidate_answer"]:
            fail(f"{tasks_path}:{line_number} candidate_answer is required")
        passed = parse_pass(row["pass"], tasks_path, line_number)
        seen_results.add(prompt_id)
        result_slices.add(row["slice"])
        slice_totals[row["slice"]] += 1
        if passed:
            slice_passed[row["slice"]] += 1
        else:
            failed_prompts.append(prompt_id)

    missing_results = sorted(set(tasks) - seen_results)
    if missing_results:
        fail(f"{tasks_path} missing result rows for: {', '.join(missing_results[:5])}")
    if result_slices != REQUIRED_SLICES:
        fail(f"{tasks_path} must include all required slices: {', '.join(sorted(REQUIRED_SLICES))}")

    summary = require_object(load_json(summary_path), summary_path)
    require_keys(summary, REQUIRED_SUMMARY_KEYS, summary_path)
    if not isinstance(summary["gate_ids"], list) or not summary["gate_ids"]:
        fail(f"{summary_path} field gate_ids must be a non-empty array")
    gate_ids = set(str(item) for item in summary["gate_ids"])
    if "quant_task_eval" not in gate_ids:
        fail(f"{summary_path} gate_ids must include quant_task_eval")
    if not ({"quant_ppl_default_4bit", "quant_ppl_ladder_variant"} & gate_ids):
        fail(f"{summary_path} gate_ids must include a quant perplexity gate")
    if summary["final_decision"] not in ALLOWED_DECISIONS:
        fail(f"{summary_path} has invalid final_decision: {summary['final_decision']}")
    aggregate_scores = summary.get("aggregate_scores", {})
    if not isinstance(aggregate_scores, dict):
        fail(f"{summary_path} aggregate_scores must be an object")
    for slice_name in REQUIRED_SLICES:
        if slice_name not in aggregate_scores:
            fail(f"{summary_path} aggregate_scores must include {slice_name}")
        score = aggregate_scores[slice_name]
        if not isinstance(score, dict):
            fail(f"{summary_path} aggregate_scores.{slice_name} must be an object")
        for key in ["passed", "total"]:
            if not isinstance(score.get(key), int) or isinstance(score.get(key), bool):
                fail(f"{summary_path} aggregate_scores.{slice_name}.{key} must be an integer")
        if score["total"] != slice_totals[slice_name]:
            fail(
                f"{summary_path} aggregate_scores.{slice_name}.total must match raw rows "
                f"({slice_totals[slice_name]})"
            )
        if score["passed"] != slice_passed[slice_name]:
            fail(
                f"{summary_path} aggregate_scores.{slice_name}.passed must match raw rows "
                f"({slice_passed[slice_name]})"
            )
    if summary["final_decision"] == "pass":
        if not ppl["pass"]:
            fail(f"{summary_path} final_decision pass requires quant_ppl.json pass=true")
        if failed_prompts:
            fail(
                f"{summary_path} final_decision pass requires all quant task rows to pass; "
                f"failed prompt(s): {', '.join(failed_prompts[:5])}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="quality/quant_tasks_v0.tsv", help="task fixture TSV")
    parser.add_argument("--run", help="quality/raw/<run> directory to validate")
    parser.add_argument("--tasks-only", action="store_true", help="validate only the task fixture")
    args = parser.parse_args()

    task_rows = validate_tasks(Path(args.tasks))
    if args.tasks_only:
        print(f"quant task fixture ok: {len(task_rows)} tasks")
        return 0
    if not args.run:
        fail("--run is required unless --tasks-only is set")
    validate_run(Path(args.run), task_rows)
    print(f"quant raw run ok: {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
