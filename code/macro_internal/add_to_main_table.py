#!/usr/bin/env python3
"""
add_to_main_table.py -- adds 3 rows to ../../main_results_table.csv from
this folder's results/macro_internal_per_combo.csv: "Macro Only (HTC-
matched pop.)", "HTC Internal Only", "Macro + HTC Internal".

IMPORTANT, not swept under the rug: these 3 rows are NOT on the same
population as every other row in the table. Every other row uses each
combo's FULL step population; these 3 use the Macro/HTC inner-joined
population (94-100% of the verbalized rollout's rows -- see
../NARRATIVE.md), and claude (both benchmarks) is NaN throughout (no
logprobs -> no HTC features -> excluded, same convention as M2-M4/M11/M13).
ece is also NaN for these 3 rows -- not computed in this experiment
(run_macro_internal.py's finalize_metrics() reports auroc/e_aurc/brier
only). Do not read across a row from these 3 to any other row's numbers
for the SAME combo as if they were comparable; they answer a different,
self-contained question (does HTC help Macro on Macro/HTC's own matched
population) and are appended here purely for one-file convenience, not to
imply population parity with the rest of the table.
"""
import numpy as np
import pandas as pd
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MAIN_TABLE = THIS_DIR.parent.parent / "main_results_table.csv"

COMBOS_ORDER = [
    "webarena_claude", "webarena_gpt_5_2", "webarena_gptnano", "webarena_kimi", "webarena_qwen30b",
    "mind2web_claude", "mind2web_gpt_5_2", "mind2web_gptnano", "mind2web_kimi", "mind2web_qwen30b",
]

ROW_LABELS = {
    "macro": "Macro Only (HTC-matched pop.)",
    "htc_only": "HTC Internal Only",
    "macro_internal": "Macro + HTC Internal",
}


def main():
    per_combo = pd.read_csv(THIS_DIR / "results" / "macro_internal_per_combo.csv")
    mt = pd.read_csv(MAIN_TABLE)

    new_rows = []
    for fs_name, label in ROW_LABELS.items():
        sub = per_combo[per_combo.feature_set == fs_name].set_index("combo")
        row = {"method": label}
        for metric_key, table_metric in [("auroc", "auroc"), ("e_aurc", "e_aurc"), ("brier", "brier")]:
            vals = []
            for combo in COMBOS_ORDER:
                v = float(sub.loc[combo, metric_key]) if combo in sub.index else np.nan
                row[f"{table_metric}__{combo}"] = v
                vals.append(v)
            arr = np.array(vals, dtype=float)
            valid = arr[~np.isnan(arr)]
            row[f"{table_metric}__mean"] = round(float(valid.mean()), 4) if len(valid) else np.nan
            row[f"{table_metric}__std"] = round(float(valid.std(ddof=0)), 4) if len(valid) else np.nan
        for combo in COMBOS_ORDER:
            row[f"ece__{combo}"] = np.nan
        row["ece__mean"] = np.nan
        row["ece__std"] = np.nan
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)[mt.columns]
    mt = mt[~mt["method"].isin(ROW_LABELS.values())]
    mt = pd.concat([mt, new_df], ignore_index=True)
    mt.to_csv(MAIN_TABLE, index=False)

    for row in new_rows:
        print(f"{row['method']:32s} auroc={row['auroc__mean']}  e_aurc={row['e_aurc__mean']}  "
              f"brier={row['brier__mean']}  ece=NaN (not computed)")
    print(f"\nWrote 3 rows -> {MAIN_TABLE}")


if __name__ == "__main__":
    main()
