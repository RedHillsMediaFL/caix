#!/usr/bin/env python3
"""Validate caix diffusion-quality fixtures and raw run evidence.

This script is intentionally no-load: it reads local TSV/JSON artifacts only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROMPT_HEADER = ["prompt_id", "slice", "rubric", "prompt", "acceptance_criteria", "notes"]
RESULT_HEADER = [
    "prompt_id",
    "slice",
    "reference_output",
    "candidate_output",
    "rubric_result",
    "rater_notes",
]
REQUIRED_SLICES = {"text_completion", "math_reasoning", "instruction_following"}
ALLOWED_RUBRICS = {"semantic", "numeric", "format"}
ALLOWED_RESULTS = {"pass", "fail", "needs_review"}
REQUIRED_METADATA_KEYS = {
    "caix_commit",
    "model_repo",
    "model_revision",
    "reference_path",
    "candidate_bundle",
    "hardware",
    "os_build",
    "command",
}
REQUIRED_API_KEYS = {"selected_api_mode", "streaming_decision", "request_examples", "pass"}
REQUIRED_SUMMARY_KEYS = {"gate_ids", "aggregate_results", "final_decision"}
ALLOWED_DECISIONS = {"pass", "needs-test", "blocked"}


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"{path} is empty")
        return reader.fieldnames, list(reader)


def load_json(path: Path) -> object:
    try:
        with path.open() as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"{path} is not valid JSON: {exc}")


def require_object(value: object, path: Path) -> dict[str, object]:
    if not isinstance(value, dict):
        fail(f"{path} must contain a JSON object")
    return value


def require_keys(value: dict[str, object], keys: set[str], path: Path) -> None:
    missing = sorted(key for key in keys if key not in value)
    if missing:
        fail(f"{path} missing required keys: {', '.join(missing)}")


def validate_prompts(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        fail(f"diffusion prompt fixture not found: {path}")

    header, rows = load_tsv(path)
    if header != PROMPT_HEADER:
        fail(f"{path} header must be: {' '.join(PROMPT_HEADER)}")
    if len(rows) != 30:
        fail(f"{path} must contain exactly 30 prompts, found {len(rows)}")

    seen: dict[str, dict[str, str]] = {}
    slice_counts = {name: 0 for name in REQUIRED_SLICES}
    for line_number, row in enumerate(rows, start=2):
        prompt_id = row["prompt_id"]
        if prompt_id in seen:
            fail(f"{path}:{line_number} duplicate prompt_id: {prompt_id}")
        if not prompt_id.startswith("dq-"):
            fail(f"{path}:{line_number} prompt_id must start with dq-")
        if row["slice"] not in REQUIRED_SLICES:
            fail(f"{path}:{line_number} unknown slice: {row['slice']}")
        if row["rubric"] not in ALLOWED_RUBRICS:
            fail(f"{path}:{line_number} unknown rubric: {row['rubric']}")
        if not row["prompt"] or not row["acceptance_criteria"]:
            fail(f"{path}:{line_number} prompt and acceptance_criteria are required")
        slice_counts[row["slice"]] += 1
        seen[prompt_id] = row

    for slice_name, count in slice_counts.items():
        if count != 10:
            fail(f"diffusion prompt fixture must include 10 {slice_name} prompts, found {count}")

    return seen


def validate_run(run_dir: Path, prompts: dict[str, dict[str, str]]) -> None:
    if not run_dir.is_dir():
        fail(f"run directory not found: {run_dir}")

    metadata_path = run_dir / "metadata.json"
    quality_path = run_dir / "diffusion_quality.tsv"
    api_path = run_dir / "diffusion_api.json"
    summary_path = run_dir / "summary.json"
    for path in [metadata_path, quality_path, api_path, summary_path]:
        if not path.is_file():
            fail(f"required artifact missing: {path}")

    metadata = require_object(load_json(metadata_path), metadata_path)
    require_keys(metadata, REQUIRED_METADATA_KEYS, metadata_path)

    api = require_object(load_json(api_path), api_path)
    require_keys(api, REQUIRED_API_KEYS, api_path)
    if api["selected_api_mode"] != "nonstreaming_only_v1":
        fail(f"{api_path} selected_api_mode must be nonstreaming_only_v1")
    if "token" in str(api["streaming_decision"]).lower():
        fail(f"{api_path} streaming_decision must not imply token streaming")
    if not isinstance(api["request_examples"], list) or not api["request_examples"]:
        fail(f"{api_path} request_examples must be a non-empty array")
    if not isinstance(api["pass"], bool):
        fail(f"{api_path} field pass must be boolean")

    header, result_rows = load_tsv(quality_path)
    if header != RESULT_HEADER:
        fail(f"{quality_path} header must be: {' '.join(RESULT_HEADER)}")
    if len(result_rows) != len(prompts):
        fail(f"{quality_path} must contain {len(prompts)} result rows, found {len(result_rows)}")

    seen_results: set[str] = set()
    result_slices: set[str] = set()
    slice_totals = {name: 0 for name in REQUIRED_SLICES}
    slice_passed = {name: 0 for name in REQUIRED_SLICES}
    failed_prompts: list[str] = []
    for line_number, row in enumerate(result_rows, start=2):
        prompt_id = row["prompt_id"]
        if prompt_id not in prompts:
            fail(f"{quality_path}:{line_number} unknown prompt_id: {prompt_id}")
        if prompt_id in seen_results:
            fail(f"{quality_path}:{line_number} duplicate prompt_id: {prompt_id}")
        if row["slice"] != prompts[prompt_id]["slice"]:
            fail(f"{quality_path}:{line_number} slice does not match fixture for {prompt_id}")
        if not row["reference_output"] or not row["candidate_output"]:
            fail(f"{quality_path}:{line_number} reference_output and candidate_output are required")
        if row["rubric_result"] not in ALLOWED_RESULTS:
            fail(f"{quality_path}:{line_number} rubric_result must be pass, fail, or needs_review")
        seen_results.add(prompt_id)
        result_slices.add(row["slice"])
        slice_totals[row["slice"]] += 1
        if row["rubric_result"] == "pass":
            slice_passed[row["slice"]] += 1
        else:
            failed_prompts.append(prompt_id)

    missing_results = sorted(set(prompts) - seen_results)
    if missing_results:
        fail(f"{quality_path} missing result rows for: {', '.join(missing_results[:5])}")
    if result_slices != REQUIRED_SLICES:
        fail(f"{quality_path} must include all required slices: {', '.join(sorted(REQUIRED_SLICES))}")

    summary = require_object(load_json(summary_path), summary_path)
    require_keys(summary, REQUIRED_SUMMARY_KEYS, summary_path)
    if not isinstance(summary["gate_ids"], list) or not summary["gate_ids"]:
        fail(f"{summary_path} field gate_ids must be a non-empty array")
    gate_ids = set(str(item) for item in summary["gate_ids"])
    for gate_id in ["diffusion_block_quality", "diffusion_api_contract"]:
        if gate_id not in gate_ids:
            fail(f"{summary_path} gate_ids must include {gate_id}")
    if summary["final_decision"] not in ALLOWED_DECISIONS:
        fail(f"{summary_path} has invalid final_decision: {summary['final_decision']}")
    aggregate = summary.get("aggregate_results", {})
    if not isinstance(aggregate, dict):
        fail(f"{summary_path} aggregate_results must be an object")
    for slice_name in REQUIRED_SLICES:
        if slice_name not in aggregate:
            fail(f"{summary_path} aggregate_results must include {slice_name}")
        result = aggregate[slice_name]
        if not isinstance(result, dict):
            fail(f"{summary_path} aggregate_results.{slice_name} must be an object")
        for key in ["passed", "total"]:
            if not isinstance(result.get(key), int) or isinstance(result.get(key), bool):
                fail(f"{summary_path} aggregate_results.{slice_name}.{key} must be an integer")
        if result["total"] != slice_totals[slice_name]:
            fail(
                f"{summary_path} aggregate_results.{slice_name}.total must match raw rows "
                f"({slice_totals[slice_name]})"
            )
        if result["passed"] != slice_passed[slice_name]:
            fail(
                f"{summary_path} aggregate_results.{slice_name}.passed must match raw rows "
                f"({slice_passed[slice_name]})"
            )
    if summary["final_decision"] == "pass":
        if not api["pass"]:
            fail(f"{summary_path} final_decision pass requires diffusion_api.json pass=true")
        if failed_prompts:
            fail(
                f"{summary_path} final_decision pass requires all diffusion quality rows to pass; "
                f"failed prompt(s): {', '.join(failed_prompts[:5])}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="quality/diffusion_prompts_v0.tsv", help="prompt fixture TSV")
    parser.add_argument("--run", help="quality/raw/<run> directory to validate")
    parser.add_argument("--prompts-only", action="store_true", help="validate only the prompt fixture")
    args = parser.parse_args()

    prompts = validate_prompts(Path(args.prompts))
    if args.prompts_only:
        print(f"diffusion prompt fixture ok: {len(prompts)} prompts")
        return 0
    if not args.run:
        fail("--run is required unless --prompts-only is set")
    validate_run(Path(args.run), prompts)
    print(f"diffusion raw run ok: {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
