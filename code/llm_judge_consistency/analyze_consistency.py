#!/usr/bin/env python3
"""
LLM-judge self-consistency analysis: compares N independent Azure OpenAI
judge runs over the same stratified sample against each other (no
comparison to the original Gemini annotations -- this is purely
self-consistency: does the judge agree with itself across repeated,
independent calls with the same model/prompt/temperature?).

Metrics:
  failureCause (categorical):
    - exact match rate across all N passes
    - majority agreement rate (>=ceil(N/2) agree)
    - Fleiss' kappa across the N passes
    - pairwise exact match + Cohen's kappa for each pass pair
  keyStep (integer):
    - exact match rate across all N passes
    - "tight" rate: max-min <= 1 across the N passes
    - mean pairwise absolute difference (aggregate + per pass pair)

Usage:
  python analyze_consistency.py --passes results/temp0_pass1,results/temp0_pass2,results/temp0_pass3 \
      --manifest results/sample_150.json --out results/temp0_consistency_report.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def load_pass_items(pass_dir: Path) -> dict:
    items = {}
    for f in pass_dir.glob("*_azure.json"):
        d = json.load(open(f))
        items.update(d["items"])
    return items


def fleiss_kappa(rows: list[list[str]]) -> float:
    """rows: one list of N category labels per item (N raters, constant)."""
    n_items = len(rows)
    N = len(rows[0])

    P_i = []
    cat_totals = Counter()
    for row in rows:
        counts = Counter(row)
        for c, n_ij in counts.items():
            cat_totals[c] += n_ij
        p_i = sum(n_ij * (n_ij - 1) for n_ij in counts.values()) / (N * (N - 1))
        P_i.append(p_i)

    P_bar = sum(P_i) / n_items
    p_j = {c: total / (n_items * N) for c, total in cat_totals.items()}
    P_e_bar = sum(v ** 2 for v in p_j.values())

    if P_e_bar == 1.0:
        return 1.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Standard 2-rater Cohen's kappa."""
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = sorted(set(a) | set(b))
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in cats)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", required=True, help="comma-separated pass output dirs (relative to this script or absolute)")
    ap.add_argument("--manifest", default=str(SCRIPT_DIR / "results" / "sample_150.json"))
    ap.add_argument("--out", required=True, help="path to write the machine-readable report json")
    args = ap.parse_args()

    pass_dirs = [Path(p) if Path(p).is_absolute() else SCRIPT_DIR / p for p in args.passes.split(",")]
    pass_names = [p.name for p in pass_dirs]

    manifest = json.load(open(args.manifest))
    task_keys = [it["taskKey"] for it in manifest["items"]]

    pass_items = {name: load_pass_items(d) for name, d in zip(pass_names, pass_dirs)}

    missing = {name: [k for k in task_keys if k not in pass_items[name]] for name in pass_names}
    for name, m in missing.items():
        if m:
            print(f"WARNING: {name} missing {len(m)} items: {m[:5]}{'...' if len(m) > 5 else ''}")

    complete_keys = [k for k in task_keys if all(k in pass_items[name] for name in pass_names)]
    print(f"Items with all {len(pass_names)} passes: {len(complete_keys)} / {len(task_keys)}")
    print()

    fc_rows, ks_rows, disagreements = [], [], []
    for k in complete_keys:
        fcs = [pass_items[name][k]["failureCause"] for name in pass_names]
        kss = [pass_items[name][k]["keyStep"] for name in pass_names]
        fc_rows.append(fcs)
        ks_rows.append(kss)
        if len(set(fcs)) > 1:
            disagreements.append((k, fcs, kss))

    n = len(complete_keys)
    fc_exact = sum(1 for row in fc_rows if len(set(row)) == 1) / n
    fc_majority = sum(1 for row in fc_rows if Counter(row).most_common(1)[0][1] >= (len(pass_names) // 2 + 1)) / n
    kappa = fleiss_kappa(fc_rows)

    ks_exact = sum(1 for row in ks_rows if len(set(row)) == 1) / n
    ks_tight = sum(1 for row in ks_rows if max(row) - min(row) <= 1) / n
    pairwise_diffs = [abs(row[i] - row[j]) for row in ks_rows
                       for i in range(len(row)) for j in range(i + 1, len(row))]
    ks_mean_pairwise_diff = sum(pairwise_diffs) / len(pairwise_diffs)

    print("=" * 60)
    print(f"SELF-CONSISTENCY ({len(pass_names)} independent {'/'.join(pass_names)} judge runs, n={n})")
    print("=" * 60)
    print()
    print("failureCause:")
    print(f"  exact match (all agree)   : {fc_exact:.1%}")
    print(f"  majority agreement        : {fc_majority:.1%}")
    print(f"  Fleiss' kappa             : {kappa:.3f}")
    print()
    print("keyStep:")
    print(f"  exact match (all agree)   : {ks_exact:.1%}")
    print(f"  tight (max-min <= 1)      : {ks_tight:.1%}")
    print(f"  mean pairwise |diff|      : {ks_mean_pairwise_diff:.2f} steps")
    print()

    pairwise_report = {}
    print("=" * 60)
    print("PAIRWISE AGREEMENT")
    print("=" * 60)
    for i in range(len(pass_names)):
        for j in range(i + 1, len(pass_names)):
            pi, pj = pass_names[i], pass_names[j]
            fc_a = [row[i] for row in fc_rows]
            fc_b = [row[j] for row in fc_rows]
            ks_a = [row[i] for row in ks_rows]
            ks_b = [row[j] for row in ks_rows]
            exact = sum(1 for x, y in zip(fc_a, fc_b) if x == y) / n
            kappa_ij = cohen_kappa(fc_a, fc_b)
            diffs = [abs(x - y) for x, y in zip(ks_a, ks_b)]
            mean_diff = sum(diffs) / n
            tight = sum(1 for d in diffs if d <= 1) / n
            print(f"  {pi} vs {pj}: failureCause exact={exact:.1%}  kappa={kappa_ij:.3f}   "
                  f"keyStep mean|diff|={mean_diff:.2f}  tight(<=1)={tight:.1%}")
            pairwise_report[f"{pi}_vs_{pj}"] = {
                "failureCause_exact_match": exact,
                "failureCause_cohen_kappa": kappa_ij,
                "keyStep_mean_abs_diff": mean_diff,
                "keyStep_tight_leq1": tight,
            }
    print()

    print("=" * 60)
    print(f"failureCause DISAGREEMENTS ({len(disagreements)} / {n} items)")
    print("=" * 60)
    for k, fcs, kss in disagreements[:20]:
        print(f"  {k}\n    failureCause: {fcs}\n    keyStep     : {kss}")
    if len(disagreements) > 20:
        print(f"  ... and {len(disagreements) - 20} more")

    report = {
        "n_items": n,
        "failureCause": {"exact_match_rate": fc_exact, "majority_agreement_rate": fc_majority, "fleiss_kappa": kappa},
        "keyStep": {"exact_match_rate": ks_exact, "tight_rate_leq1": ks_tight, "mean_pairwise_abs_diff": ks_mean_pairwise_diff},
        "pairwise": pairwise_report,
        "n_disagreements": len(disagreements),
        "disagreement_task_keys": [d[0] for d in disagreements],
    }
    out_path = Path(args.out) if Path(args.out).is_absolute() else SCRIPT_DIR / args.out
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
