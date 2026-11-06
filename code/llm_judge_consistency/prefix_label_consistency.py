#!/usr/bin/env python3
"""
Prefix-LABEL consistency, not raw keyStep consistency.

The trained risk model never sees `keyStep` or `failureCause` directly -- it
sees a per-step binary label derived from `keyStep` (see
`code/l2_training/train_risk_model_temp_sca.py::apply_keystep_labels`,
lines 182-214):

    label[step_k] = 0   if step_k <  keyStep
    label[step_k] = 1   if step_k >= keyStep     (for a failed task)

Two judge passes whose keyStep differs by a few steps can still produce
IDENTICAL prefix labels for most of the trajectory -- only the steps strictly
between the two keySteps flip. This script measures that directly: per-step
label agreement across N independent judge passes, both pairwise and
majority-vote, macro (mean over tasks) and micro (step-weighted over the
whole corpus).

Closed form used for the pairwise case (sanity-checked against a brute-force
per-step comparison): for a task of length n with two keySteps a, b, the
number of steps whose label DIFFERS is exactly |a - b| (the steps in
[min(a,b), max(a,b))), so per-task pairwise step-agreement = 1 - |a-b|/n.
For 3 passes, the disagreement zone (any step where not all 3 agree) is
[min(a,b,c), max(a,b,c)), i.e. per-task 3-way agreement = 1 - (max-min)/n.
The majority-vote label at each step equals the indicator step_k >= median(a,b,c)
-- i.e. combining 3 judge passes by majority vote is equivalent to just using
the MEDIAN keyStep as the effective keyStep.

Usage:
  python prefix_label_consistency.py --passes results/gpt52_pass1,results/gpt52_pass2,results/gpt52_pass3 \
      --manifest results/sample_150.json --out results/gpt52_prefix_consistency_report.json
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def load_pass_items(pass_dir: Path) -> dict:
    items = {}
    for f in pass_dir.glob("*_azure.json"):
        d = json.load(open(f))
        items.update(d["items"])
    return items


def per_step_labels(key_step: int, n_steps: int) -> list[int]:
    return [0 if k < key_step else 1 for k in range(n_steps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", required=True, help="comma-separated pass output dirs")
    ap.add_argument("--manifest", default=str(SCRIPT_DIR / "results" / "sample_150.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pass_dirs = [Path(p) if Path(p).is_absolute() else SCRIPT_DIR / p for p in args.passes.split(",")]
    pass_names = [p.name for p in pass_dirs]
    n_pass = len(pass_names)

    manifest = json.load(open(args.manifest))
    task_keys = [it["taskKey"] for it in manifest["items"]]
    pass_items = {name: load_pass_items(d) for name, d in zip(pass_names, pass_dirs)}

    complete_keys = [k for k in task_keys if all(k in pass_items[name] for name in pass_names)]
    print(f"Items with all {n_pass} passes: {len(complete_keys)} / {len(task_keys)}")

    # per-task data: n_steps (validated identical across passes), keySteps per pass
    tasks = []
    n_steps_mismatch = []
    for k in complete_keys:
        n_steps_per_pass = [len(pass_items[name][k]["step_analyses"]) for name in pass_names]
        if len(set(n_steps_per_pass)) != 1:
            n_steps_mismatch.append((k, n_steps_per_pass))
            continue
        n_steps = n_steps_per_pass[0]
        key_steps = [pass_items[name][k]["keyStep"] for name in pass_names]
        tasks.append((k, n_steps, key_steps))
    if n_steps_mismatch:
        print(f"WARNING: {len(n_steps_mismatch)} tasks had mismatched n_steps across passes, excluded: {n_steps_mismatch[:3]}")

    n = len(tasks)
    total_steps = sum(t[1] for t in tasks)

    # ── pairwise ──────────────────────────────────────────────────────────
    pairwise_report = {}
    print()
    print("=" * 60)
    print("PAIRWISE PREFIX-LABEL AGREEMENT (per-step)")
    print("=" * 60)
    for i in range(n_pass):
        for j in range(i + 1, n_pass):
            pi, pj = pass_names[i], pass_names[j]
            macro_vals, micro_num = [], 0
            for k, n_steps, key_steps in tasks:
                diff = abs(key_steps[i] - key_steps[j])
                agree_frac = 1 - diff / n_steps
                macro_vals.append(agree_frac)
                micro_num += n_steps - diff
            macro = sum(macro_vals) / n
            micro = micro_num / total_steps
            print(f"  {pi} vs {pj}: macro (mean per-task) = {macro:.1%}   micro (step-weighted) = {micro:.1%}")
            pairwise_report[f"{pi}_vs_{pj}"] = {"macro_agreement": macro, "micro_agreement": micro}

    # ── N-way (all agree) + majority-vote-as-median ──────────────────────
    macro_all_agree, micro_all_agree_num = [], 0
    macro_majority_recall = {name: [] for name in pass_names}  # how often does pass p match the median-derived label?
    for k, n_steps, key_steps in tasks:
        span = max(key_steps) - min(key_steps)
        all_agree_frac = 1 - span / n_steps
        macro_all_agree.append(all_agree_frac)
        micro_all_agree_num += n_steps - span

        median_ks = sorted(key_steps)[n_pass // 2]
        for name, ks in zip(pass_names, key_steps):
            diff = abs(ks - median_ks)
            macro_majority_recall[name].append(1 - diff / n_steps)

    macro_all = sum(macro_all_agree) / n
    micro_all = micro_all_agree_num / total_steps

    print()
    print("=" * 60)
    print(f"{n_pass}-WAY PREFIX-LABEL AGREEMENT (all passes agree at that step)")
    print("=" * 60)
    print(f"  macro (mean per-task) = {macro_all:.1%}")
    print(f"  micro (step-weighted) = {micro_all:.1%}")
    print()
    print("Per-pass agreement with the majority-vote label sequence")
    print("(== agreement with the MEDIAN keyStep -- majority vote of N thresholds is a median threshold):")
    majority_report = {}
    for name in pass_names:
        vals = macro_majority_recall[name]
        m = sum(vals) / n
        print(f"  {name}: {m:.1%}")
        majority_report[name] = m

    report = {
        "n_items": n,
        "n_items_excluded_n_steps_mismatch": len(n_steps_mismatch),
        "total_steps": total_steps,
        "pairwise": pairwise_report,
        "all_agree": {"macro": macro_all, "micro": micro_all},
        "per_pass_agreement_with_median": majority_report,
    }
    out_path = Path(args.out) if Path(args.out).is_absolute() else SCRIPT_DIR / args.out
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
