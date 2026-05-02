#!/usr/bin/env python3
"""
Sentinel-Stream governance gate: fail CI if validation metrics fall below mlops/promotion_policy.yaml.
Reads artifacts/metadata.json (no network; no secrets).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check training metrics against promotion policy.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "artifacts" / "metadata.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "mlops" / "promotion_policy.yaml",
    )
    args = parser.parse_args()

    if not args.metadata.is_file():
        print(f"ERROR: metadata not found: {args.metadata}", file=sys.stderr)
        return 2
    if not args.policy.is_file():
        print(f"ERROR: policy not found: {args.policy}", file=sys.stderr)
        return 2

    with open(args.metadata, encoding="utf-8") as f:
        meta = json.load(f)
    with open(args.policy, encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    pr_auc = float(meta["pr_auc_val"])
    roc_auc = float(meta["roc_auc_val"])
    min_pr = float(policy["min_pr_auc_val"])
    min_roc = float(policy["min_roc_auc_val"])

    failures: list[str] = []
    if pr_auc < min_pr:
        failures.append(f"pr_auc_val {pr_auc:.4f} < required {min_pr:.4f}")
    if roc_auc < min_roc:
        failures.append(f"roc_auc_val {roc_auc:.4f} < required {min_roc:.4f}")

    if failures:
        print("PROMOTION GATE FAILED:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print(
        f"Promotion gate OK: pr_auc_val={pr_auc:.4f} (>={min_pr:.4f}), "
        f"roc_auc_val={roc_auc:.4f} (>={min_roc:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
