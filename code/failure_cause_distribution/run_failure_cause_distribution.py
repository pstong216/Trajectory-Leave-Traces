#!/usr/bin/env python3
"""
Appendix Table 4 -- distribution of annotated failure causes.

Counts the `failureCause` field of every key-step annotation in
keystep_labels/<benchmark>/*.json, grouped into the five AgentErrorTaxonomy
modules (Action / Planning / Reflection / Memory / System) and their
fine-grained sub-causes.

Both annotation roles (<model>_baseline.json and <model>_ours.json) are
pooled, because each is an independent annotation pass over that model's
own rollout -- see docs/DATA.md, "Key-step labels".

Reproduction status (see README, "Known deviations"):
  * WebArena-Lite reproduces the paper's column exactly
    (Action 307 / 29.8%, Planning 382 / 37.1%, Reflection 272 / 26.4%).
  * Online Mind2Web comes out ~10% lower than the published column; the
    paper's Mind2Web counts were taken from an annotation snapshot that is
    not the one released here.

Usage:
  python3 run_failure_cause_distribution.py [--labels ../../keystep_labels] [--out results]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

MODULE_ORDER = ["Action", "Planning", "Reflection", "Memory", "System"]
HERE = Path(__file__).resolve().parent


def collect(label_dir: Path) -> tuple[Counter, dict[str, Counter], int]:
    modules: Counter = Counter()
    sub: dict[str, Counter] = defaultdict(Counter)
    n_files = 0
    for path in sorted(label_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        items = payload.get("items", payload)
        if not isinstance(items, dict):
            continue
        n_files += 1
        for item in items.values():
            cause = item.get("failureCause") if isinstance(item, dict) else None
            if not cause:
                continue
            module, _, detail = cause.partition(":")
            modules[module] += 1
            sub[module][detail or "(unspecified)"] += 1
    return modules, sub, n_files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=HERE.parent.parent / "keystep_labels")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for bench in ["webarena", "mind2web"]:
        modules, sub, n_files = collect(args.labels / bench)
        total = sum(modules.values())
        print(f"\n== {bench}  ({n_files} annotation files, {total} annotated failures)")
        for module in MODULE_ORDER:
            n = modules.get(module, 0)
            if not n:
                continue
            print(f"  {module:<12s} {n:5d}  {100 * n / total:5.1f}%")
            rows.append(dict(benchmark=bench, module=module, sub_cause="",
                             count=n, pct=round(100 * n / total, 1)))
            for detail, m in sub[module].most_common():
                print(f"    {detail:<32s} {m:5d}  {100 * m / total:5.1f}%")
                rows.append(dict(benchmark=bench, module=module, sub_cause=detail,
                                 count=m, pct=round(100 * m / total, 1)))

    out_csv = args.out / "failure_cause_distribution.csv"
    import csv
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["benchmark", "module", "sub_cause", "count", "pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
