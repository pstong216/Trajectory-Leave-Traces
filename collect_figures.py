#!/usr/bin/env python3
"""Collect the paper's figures into figures/, named by their number in the paper.

The plotting scripts each write into their own analysis directory under whatever
name that experiment used while the paper was being written (f5_, f7_, f9_ ...,
which do not line up with the final figure numbers). This gathers them into one
place under unambiguous names. Run it after reproduce.sh, which calls it anyway.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"

# paper figure number -> (source path, destination stem)
FIGURES = {
    3:  ("figures/success_false_alarm_curves.png", "fig03_success_false_alarm"),
    4:  ("code/generalization/detection_false_cut_analysis/results/detection_false_cut_curves.png",
         "fig04_detection_false_cut"),
    5:  ("code/generalization/sizematched_control/results/website_generalization_sizematched_control.png",
         "fig05_cross_category_sizematched"),
    6:  ("figures/f10_case_micro.pdf", "fig06_case_study_macro_micro"),
    7:  ("figures/f5_clustering_metrics.pdf", "fig07_clustering_robustness"),
    8:  ("figures/f11_case_internal.pdf", "fig08_case_study_vs_internal"),
    9:  ("code/feature_importance/results/feature_ablation_ranking.png",
         "fig09_feature_group_ablation"),
    10: ("code/generalization/prefix_time_analysis/results/prefix_time_curves_matched8.png",
         "fig10_prefix_time"),
}
# Figures 1 and 2 are hand-drawn schematics with no generating script.

def main() -> int:
    OUT.mkdir(exist_ok=True)
    missing = []
    for num, (src_rel, stem) in sorted(FIGURES.items()):
        src = ROOT / src_rel
        if not src.is_file():
            missing.append((num, src_rel))
            continue
        dst = OUT / f"{stem}{src.suffix}"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        print(f"  Figure {num:<2} -> figures/{dst.name}")

    # drop the pre-collection copies so figures/ holds one canonical name each
    for legacy in ("success_false_alarm_curves.png", "success_false_alarm_curves.pdf",
                   "f10_case_micro.pdf", "f5_clustering_metrics.pdf", "f11_case_internal.pdf"):
        (OUT / legacy).unlink(missing_ok=True)

    if missing:
        print("\nNot yet generated (run reproduce.sh):")
        for num, rel in missing:
            print(f"  Figure {num}: {rel}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
