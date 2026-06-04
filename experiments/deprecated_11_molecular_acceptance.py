"""Acceptance check for molecular scans (H2/LiH)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _extract_score(entry: dict[str, Any], method_key: str) -> float | None:
    method = entry.get(method_key, {})
    for k in ("fidelity", "proxy_fidelity"):
        v = method.get(k)
        if v is not None:
            return float(v)
    return None


def _evaluate_dataset(path: Path, dataset_name: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("results", [])

    checked = 0
    passed = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        ours = _extract_score(row, "ours")
        base = _extract_score(row, "qiskit_4th")
        if ours is None or base is None:
            skipped += 1
            continue
        checked += 1
        if ours + 1e-12 >= base:
            passed += 1
        else:
            failures.append(
                {
                    "dataset": dataset_name,
                    "index": idx,
                    "bond_length": row.get("bond_length"),
                    "ours": ours,
                    "baseline": base,
                }
            )

    return {
        "dataset": dataset_name,
        "path": str(path),
        "total_rows": len(rows),
        "checked": checked,
        "passed": passed,
        "skipped": skipped,
        "failed": checked - passed,
        "failures": failures,
        "status": (checked > 0 and passed == checked),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h2-path",
        default="experiments/benchmark_results/h2_bond_scan.json",
    )
    parser.add_argument(
        "--lih-path",
        default="experiments/benchmark_results/lih_bond_scan.json",
    )
    parser.add_argument(
        "--output",
        default="experiments/benchmark_results/molecular_acceptance_report.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when acceptance fails.",
    )
    args = parser.parse_args()

    h2_path = Path(args.h2_path)
    lih_path = Path(args.lih_path)
    if not h2_path.exists():
        raise FileNotFoundError(f"H2 result file not found: {h2_path}")
    if not lih_path.exists():
        raise FileNotFoundError(f"LiH result file not found: {lih_path}")

    h2 = _evaluate_dataset(h2_path, "H2")
    lih = _evaluate_dataset(lih_path, "LiH")

    overall_status = bool(h2["status"] and lih["status"])
    report = {
        "overall_status": overall_status,
        "h2": h2,
        "lih": lih,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved acceptance report to {out_path}")
    print(f"overall_status={overall_status} | h2={h2['status']} | lih={lih['status']}")

    if args.strict and not overall_status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
