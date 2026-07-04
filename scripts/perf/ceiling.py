#!/usr/bin/env python3
"""Compute decode throughput ceilings from reviewed active-weight estimates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


REQUIRED_ASSUMPTION_FIELDS = {
    "repo",
    "local_dir",
    "kind",
    "benchmark_mode",
    "active_weight_gib",
    "evidence_status",
    "source",
    "notes",
}
VALID_EVIDENCE_STATUSES = {"estimate", "reviewed", "measured"}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"error: empty TSV: {path}")
        return [{key: value for key, value in row.items()} for row in reader]


def require_fields(path: Path, rows: list[dict[str, str]], fields: set[str]) -> None:
    if not rows:
        return
    missing = fields - set(rows[0])
    if missing:
        names = ", ".join(sorted(missing))
        raise SystemExit(f"error: missing required fields in {path}: {names}")


def require_value(row: dict[str, str], field: str, label: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise SystemExit(f"error: missing {field} for {label}")
    return value


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def first_numeric(row: dict[str, str], names: Iterable[str]) -> float | None:
    for name in names:
        if name in row:
            value = parse_float(row[name])
            if value is not None:
                return value
    return None


def key_variants(row: dict[str, str]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for name in ("repo", "local_dir"):
        value = row.get(name, "").strip()
        if value:
            keys.append((name, value))
    return keys


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        for key in key_variants(row):
            index[key] = row
    return index


def validate_assumptions(rows: list[dict[str, str]]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        repo = require_value(row, "repo", "assumption row")
        local_dir = require_value(row, "local_dir", repo)
        require_value(row, "kind", repo)
        require_value(row, "benchmark_mode", repo)
        require_value(row, "source", repo)
        require_value(row, "notes", repo)

        active_weight = parse_float(row.get("active_weight_gib", ""))
        if active_weight is None or active_weight <= 0:
            raise SystemExit(f"error: active_weight_gib must be positive for {repo}")

        evidence_status = require_value(row, "evidence_status", repo)
        if evidence_status not in VALID_EVIDENCE_STATUSES:
            allowed = ", ".join(sorted(VALID_EVIDENCE_STATUSES))
            raise SystemExit(f"error: invalid evidence_status for {repo}: {evidence_status}; expected one of {allowed}")

        for key in (("repo", repo), ("local_dir", local_dir)):
            if key in seen:
                raise SystemExit(f"error: duplicate ceiling assumption for {key[0]}={key[1]}")
            seen.add(key)


def validate_assumptions_against_manifest(
    assumptions: list[dict[str, str]],
    manifest: list[dict[str, str]],
) -> None:
    manifest_by_repo = {row.get("repo", ""): row for row in manifest if row.get("repo", "")}
    manifest_by_local_dir = {row.get("local_dir", ""): row for row in manifest if row.get("local_dir", "")}

    for assumption in assumptions:
        repo = assumption["repo"]
        local_dir = assumption["local_dir"]
        manifest_row = manifest_by_repo.get(repo)
        if manifest_row is None:
            manifest_row = manifest_by_local_dir.get(local_dir)
        if manifest_row is None:
            raise SystemExit(f"error: ceiling assumption has no manifest row: {repo} ({local_dir})")

        for field in ("repo", "local_dir", "kind", "benchmark_mode"):
            expected = manifest_row.get(field, "").strip()
            actual = assumption.get(field, "").strip()
            if actual != expected:
                raise SystemExit(
                    f"error: ceiling assumption drift for {repo}: {field} is '{actual}', "
                    f"manifest has '{expected}'"
                )


def filter_manifest(rows: list[dict[str, str]], statuses: set[str]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("status", "").strip() in statuses
        and row.get("benchmark_mode", "").strip() not in {"manual", "-"}
    ]


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def format_number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024 ** 3)


def logical_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise SystemExit(f"error: cannot parse JSON metadata {path}: {error}") from error


def asset_paths_from_metadata(bundle_dir: Path) -> list[Path]:
    metadata = read_json(bundle_dir / "metadata.json")
    assets = metadata.get("assets")
    paths: list[Path] = []
    if isinstance(assets, dict):
        for value in assets.values():
            if isinstance(value, str) and value:
                paths.append(bundle_dir / value)

    paths.extend(
        path for path in bundle_dir.rglob("*.aimodel")
        if not any(part.startswith(".") for part in path.relative_to(bundle_dir).parts)
    )
    unique = {path.relative_to(bundle_dir).as_posix(): path for path in paths}
    return [unique[key] for key in sorted(unique)]


def measure_bundle(bundle_root: Path | None, local_dir: str) -> dict[str, str]:
    if bundle_root is None:
        return {
            "bundle_dir_gib": "-",
            "bundle_asset_gib": "-",
            "bundle_assets": "-",
            "bundle_status": "not_requested",
        }

    bundle_dir = bundle_root / local_dir
    if not bundle_dir.exists():
        return {
            "bundle_dir_gib": "-",
            "bundle_asset_gib": "-",
            "bundle_assets": "-",
            "bundle_status": "missing_bundle",
        }
    if not bundle_dir.is_dir():
        return {
            "bundle_dir_gib": "-",
            "bundle_asset_gib": "-",
            "bundle_assets": "-",
            "bundle_status": "not_directory",
        }

    asset_paths = asset_paths_from_metadata(bundle_dir)
    existing_assets = [path for path in asset_paths if path.exists()]
    if not existing_assets:
        return {
            "bundle_dir_gib": format_number(bytes_to_gib(logical_size(bundle_dir)), 4),
            "bundle_asset_gib": "-",
            "bundle_assets": "-",
            "bundle_status": "missing_assets",
        }

    asset_size = sum(logical_size(path) for path in existing_assets)
    return {
        "bundle_dir_gib": format_number(bytes_to_gib(logical_size(bundle_dir)), 4),
        "bundle_asset_gib": format_number(bytes_to_gib(asset_size), 4),
        "bundle_assets": ",".join(path.relative_to(bundle_dir).as_posix() for path in existing_assets),
        "bundle_status": "measured",
    }


def merge_rows(
    assumptions: list[dict[str, str]],
    manifest: list[dict[str, str]],
    report: list[dict[str, str]],
    statuses: set[str],
    bandwidth_gib_s: float,
    bundle_root: Path | None,
) -> list[dict[str, str]]:
    assumption_index = index_rows(assumptions)
    report_index = index_rows(report)

    if manifest:
        base_rows = filter_manifest(manifest, statuses)
    else:
        base_rows = assumptions

    seen: set[tuple[str, str]] = set()
    output: list[dict[str, str]] = []

    for base in base_rows:
        row_key = key_variants(base)[0]
        seen.add(row_key)
        assumption = assumption_index.get(("repo", base.get("repo", ""))) \
            or assumption_index.get(("local_dir", base.get("local_dir", ""))) \
            or {}
        measured = report_index.get(("repo", base.get("repo", ""))) \
            or report_index.get(("local_dir", base.get("local_dir", ""))) \
            or {}
        output.append(build_row(base, assumption, measured, bandwidth_gib_s, bundle_root))

    for assumption in assumptions:
        row_key = key_variants(assumption)[0]
        if row_key in seen:
            continue
        output.append(build_row(assumption, assumption, {}, bandwidth_gib_s, bundle_root))

    return output


def build_row(
    base: dict[str, str],
    assumption: dict[str, str],
    measured: dict[str, str],
    bandwidth_gib_s: float,
    bundle_root: Path | None,
) -> dict[str, str]:
    active_weight = parse_float(assumption.get("active_weight_gib", ""))
    measured_tps = first_numeric(measured, ("median_decode_tps", "decode_tps"))

    ceiling = None
    utilization = None
    gap_to_70 = None
    status = "missing_estimate"

    if active_weight is not None and active_weight > 0:
        ceiling = bandwidth_gib_s / active_weight
        status = "needs_measurement"
        if measured_tps is not None:
            utilization = measured_tps / ceiling * 100
            gap_to_70 = max(0.0, ceiling * 0.70 - measured_tps)
            status = "near_ceiling" if utilization >= 70.0 else "gap"

    row = {
        "repo": base.get("repo", assumption.get("repo", "")),
        "local_dir": base.get("local_dir", assumption.get("local_dir", "")),
        "kind": base.get("kind", assumption.get("kind", "")),
        "benchmark_mode": base.get("benchmark_mode", assumption.get("benchmark_mode", "")),
        "active_weight_gib": format_number(active_weight, 2),
        "bandwidth_gib_s": format_number(bandwidth_gib_s, 0),
        "ceiling_decode_tps": format_number(ceiling, 1),
        "measured_decode_tps": format_number(measured_tps, 1),
        "utilization_pct": format_number(utilization, 1),
        "gap_to_70pct_tps": format_number(gap_to_70, 1),
        "status": status,
        "evidence_status": assumption.get("evidence_status", "missing"),
        "source": assumption.get("source", ""),
        "notes": assumption.get("notes", "active-weight estimate needed"),
    }
    row.update(measure_bundle(bundle_root, row["local_dir"]))
    return row


def require_estimates(rows: list[dict[str, str]], required: list[str]) -> None:
    if not required:
        return

    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        for key in (row["repo"], row["local_dir"]):
            by_key[key] = row

    for key in required:
        row = by_key.get(key)
        if row is None:
            raise SystemExit(f"error: required ceiling estimate row is missing: {key}")
        if row["status"] == "missing_estimate":
            raise SystemExit(f"error: required ceiling estimate is missing active-weight bytes: {key}")


def write_tsv(rows: list[dict[str, str]], out: Path | None) -> None:
    fields = [
        "repo",
        "local_dir",
        "kind",
        "benchmark_mode",
        "active_weight_gib",
        "bandwidth_gib_s",
        "ceiling_decode_tps",
        "measured_decode_tps",
        "utilization_pct",
        "gap_to_70pct_tps",
        "status",
        "evidence_status",
        "source",
        "notes",
        "bundle_dir_gib",
        "bundle_asset_gib",
        "bundle_assets",
        "bundle_status",
    ]
    handle = out.open("w", newline="") if out else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if out:
            handle.close()


def write_markdown(rows: list[dict[str, str]], out: Path | None, bandwidth_gib_s: float) -> None:
    lines = [
        "# Decode Ceiling Estimates",
        "",
        "Derived no-load planning artifact. Not benchmark evidence.",
        "",
        f"- Assumed usable bandwidth: `{bandwidth_gib_s:.0f} GiB/s`.",
        "- Ceiling formula: `decode_tps <= usable_bandwidth_gib_s / active_weight_gib_per_token`.",
        "- Rows with `missing_estimate` need reviewed active-weight bytes before they can be used as a denominator.",
        "- `gap_to_70pct_tps` is blank until a benchmark report supplies measured decode throughput.",
        "",
        "| repo | local dir | mode | active GiB/token | ceiling tok/s | measured tok/s | util % | gap to 70% | status | evidence | source | bundle asset GiB | bundle status | notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---|",
    ]

    for row in rows:
        lines.append(
            "| {repo} | {local_dir} | {benchmark_mode} | {active_weight_gib} | "
            "{ceiling_decode_tps} | {measured_decode_tps} | {utilization_pct} | "
            "{gap_to_70pct_tps} | {status} | {evidence_status} | {source} | "
            "{bundle_asset_gib} | {bundle_status} | {notes} |".format(
                **{key: escape_markdown(value) for key, value in row.items()}
            )
        )

    content = "\n".join(lines) + "\n"
    if out:
        out.write_text(content)
    else:
        sys.stdout.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assumptions", default="benchmarks/CEILING_ASSUMPTIONS.tsv")
    parser.add_argument("--manifest", default="benchmarks/MANIFEST.tsv")
    parser.add_argument("--benchmark-report", default="")
    parser.add_argument(
        "--bundle-root",
        default="",
        help="Optional local bundle root for no-load logical size measurements.",
    )
    parser.add_argument("--bandwidth-gib-s", type=float, default=550.0)
    parser.add_argument("--include-status", default="eligible")
    parser.add_argument(
        "--require-estimate",
        action="append",
        default=[],
        help="Require a repo or local_dir to have a non-missing active-weight estimate. May be repeated.",
    )
    parser.add_argument("--format", choices=("markdown", "tsv"), default="markdown")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    assumptions_path = Path(args.assumptions)
    manifest_path = Path(args.manifest)
    report_path = Path(args.benchmark_report) if args.benchmark_report else None
    bundle_root = Path(args.bundle_root) if args.bundle_root else None
    out_path = Path(args.out) if args.out else None

    assumptions = read_tsv(assumptions_path)
    require_fields(assumptions_path, assumptions, REQUIRED_ASSUMPTION_FIELDS)
    validate_assumptions(assumptions)

    manifest = read_tsv(manifest_path) if manifest_path.exists() else []
    if manifest:
        validate_assumptions_against_manifest(assumptions, manifest)
    report = read_tsv(report_path) if report_path else []
    statuses = {status.strip() for status in args.include_status.split(",") if status.strip()}
    rows = merge_rows(assumptions, manifest, report, statuses, args.bandwidth_gib_s, bundle_root)
    require_estimates(rows, args.require_estimate)

    if args.format == "tsv":
        write_tsv(rows, out_path)
    else:
        write_markdown(rows, out_path, args.bandwidth_gib_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
