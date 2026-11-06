# Case study (paper Sec 4.5, Figure 6)

Supports the single-trajectory case study: **Online Mind2Web / GPT-5.2 / task 173**,
where the observable (Macro) monitor fires at step 6 and the internal-signal (HTC)
predictor never crosses the threshold.

## Population

The **verbalized/baseline rollout** — the only population where Macro features and
HTC internal-state features are genuinely paired per row (`../macro_internal/` computes
Macro from HTC's own rollout, giving 94–100% join retention). Consequences:

- Labels come from `reproduction_output/baseline/<bench>/<model>/results/per_step_scores.csv`,
  already key-step relabeled for the **BASELINE** role via `<model>_baseline.json`.
  In that file `label = 1 - success`, and `outcome` is a **copy of `label`** (per-step,
  used only for fold stratification) — it is *not* a trajectory-level outcome.
  Trajectory-level facts must be derived from the labels:
  `failed <=> any(label==1)`, `k* = first step_k with label==1`.
- Failure causes must be read from **`keystep_labels/<bench>/<model>_baseline.json`**,
  never `_ours.json` — the two roles annotate different rollouts.
- **Claude is excluded** (no logprobs), leaving 8 of 10 combos.
- **Micro is not present** in this population (it needs the Qwen re-embedding pipeline
  re-run against the verbalized rollout, which has not been done). This is
  Macro-vs-Internal, not Macro&Micro-vs-Internal.

## Scripts

| File | What it does |
|---|---|
| `dump_preds.py` | Re-runs `../macro_internal/run_macro_internal.py`'s CV but **keeps per-row predictions** (that script discards them). Same `build_joined`/`make_folds`/`fit_model`, same seed, so aggregate AUROC reproduces `../macro_internal/results/macro_internal_per_combo.csv` exactly. Writes `preds/preds_<bench>_<model>.csv` with `join_id, task_id, step_k, label, outcome, p_macro, p_htc, p_mi`. |
| `blindspot.py` | Early-cut analysis over the pooled predictions: per-combo threshold calibrated to a false-cut budget, then Det/FCR and the internal-vs-observable detection cross-tab, plus the failure-cause breakdown. Pools `preds/*.csv` itself (caching `preds_pooled.csv`) and writes `traj_decisions_b10.csv`. |
| `plot_f10_case_study_hardcoded.py` | Renders the figure. Self-contained — all plotted values are inlined, so it does not depend on the two scripts above. Writes to the paper's `latex/fig/f10_case_study.pdf` by default; `--out PATH` to override. |
| `case173_risk.json` | The exact risk values drawn in the figure. |

```bash
python3 dump_preds.py --combos all      # or --combos mind2web/gpt_5_2
python3 blindspot.py                    # pools preds/*.csv automatically
python3 plot_f10_case_study_hardcoded.py
```

## Findings (for the record, not all used in the paper)

Pooled over 1,633 trajectories (1,118 failed / 515 successful), 20,456 prefixes:

| FCR budget | internal-only | observable (Macro) | both |
|---|---|---|---|
| 5%  | 0.224 | 0.188 | 0.233 |
| 10% | 0.309 | 0.310 | 0.323 |
| 20% | 0.401 | 0.453 | 0.470 |

At 10%: internal misses 773 failed trajectories, observable catches 119 of them (15.4%);
observable misses 771, internal catches 117 (15.2%). **The relationship is symmetric —
there is no one-sided internal blind spot**, and the failure-cause distribution of the
missed set matches the overall distribution. The paper therefore frames the two families
as complementary, and Sec 4.5 states explicitly that the case is one trajectory chosen
to make the mechanism legible.

## Note on `join_id`

`join_id` is unique only *within* a combo. `preds/*.csv` are therefore pooled with a
`combo` column and `uid = combo + "::" + join_id`; pooling on `join_id` alone silently
merges different trajectories.
