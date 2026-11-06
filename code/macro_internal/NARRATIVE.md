# Macro + HTC internal-state (same trajectories) — results summary

Analysis code: `run_macro_internal.py`. Follow-up to an earlier Macro/Micro+HTC variant (not part of this release)
(Macro/Micro combined with HTC across two DIFFERENT rollouts, 30-65% row
loss from the inner join). This version computes Macro directly from the
**verbalized** rollout -- the same trajectories HTC's logprobs come from --
so Macro and HTC-internal are now genuinely paired per-row, not joined
across two independent samples. Micro is dropped entirely here (it needs
the Qwen re-embedding pipeline re-run against this rollout, which hasn't
been done -- this is Macro+Internal, not Macro+Micro+Internal).

## Join quality: fixed

| | earlier variant (Macro/Micro from Ours' rollout) | This experiment (Macro from verbalized) |
|---|---|---|
| Row retention | 35-87% (typically 60-70%) | **94-100%** |

Confirms the population mismatch in the earlier experiment really was a
rollout-source artifact, not something intrinsic to combining any macro-ish
features with HTC -- once Macro is computed from HTC's own rollout, the
join is nearly lossless.

## Method

- **macro** (31 feat.) vs. **macro_internal** (31+48=79 feat.), same
  `fit_model("Logistic_L2", ...)` / 5-fold protocol as Ours, same folds
  for both feature sets per combo.
- Labels: `per_step_scores.csv`'s own "success" column (already
  keystep-relabeled for the BASELINE role via `<model>_baseline.json` --
  reused as-is, not re-derived, to avoid the baseline-vs-ours keystep-file
  mixup this repo's README warns about).
- claude excluded (no logprobs), 8/10 combos usable.

## Results (mean across 8 combos)

| Feature set | AUROC | e-AURC | Brier |
|---|---|---|---|
| Macro only (31 feat.) | 0.7855 | 0.1407 | 0.1830 |
| Macro + Internal (79 feat.) | 0.7823 | 0.1404 | 0.1802 |

**Paired delta (macro_internal − macro), N=8:**

| Metric | Mean Δ | t-test p | Wilcoxon p | Combos worse |
|---|---|---|---|---|
| AUROC | −0.0033 | 0.65 | 0.84 | 4/8 |
| e-AURC | −0.0003 | 0.97 | 1.00 | 4/8 |
| Brier | −0.0028 (better) | 0.47 | 0.55 | 4/8 |

**A genuine null result, not a degradation.** Unlike an earlier Macro/Micro+HTC variant (not part of this release)
(where adding HTC on top of Macro+Micro consistently hurt, 7/8 combos
worse), here it's an exact coin flip (4/8 worse on every metric) with
mean deltas an order of magnitude smaller than the earlier experiment's
and nowhere near significant on any test. Once the population artifact is
removed, HTC's internal-state features don't reliably help OR hurt Macro
alone.

## Per-combo pattern is heterogeneous, not uniformly null

| Combo | ΔAUROC | Direction |
|---|---|---|
| webarena/qwen30b | **+0.0237** | clear improvement (e-AURC 0.157→0.113, Brier 0.147→0.126, both also better) |
| mind2web/qwen30b | **+0.0124** | clear improvement (e-AURC and Brier also both better) |
| webarena/kimi | +0.0088 | small improvement |
| mind2web/gptnano | +0.0025 | ~flat |
| mind2web/kimi | −0.0046 | ~flat |
| webarena/gptnano | −0.0388 | clear degradation |
| mind2web/gpt_5_2 | −0.0123 | small degradation |
| webarena/gpt_5_2 | **−0.0177** | clear degradation |

The mean being near-zero is a real cancellation, not every combo
individually being null -- qwen30b (both benchmarks) gains meaningfully
from HTC's internal-state features on top of Macro, while gpt_5_2 (both
benchmarks) and webarena/gptnano lose. No confirmed mechanism for the
backbone-specific split; flagged as an observation, not explained. This
is a different, and arguably more informative, story than a flat "no
effect" -- it suggests HTC's internal states may substitute for or
duplicate signal Macro already captures for some backbones (where it's a
net negative, if L2 spreads capacity across now-redundant features) while
adding genuinely new signal for others (qwen30b).

## Reading together with an earlier Macro/Micro+HTC variant (not part of this release)

The two experiments answer related but distinct questions:
- an earlier Macro/Micro+HTC variant (not part of this release): does adding HTC help the model actually used in
  the paper (Macro+Micro)? Answer: no, mildly hurts (7/8 combos worse,
  e-AURC p=0.023) -- but that comparison's population was cut 30-65% by
  the cross-rollout join, so some of that effect could in principle be a
  population artifact rather than a genuine feature-interaction effect.
- This experiment: with the population issue fixed (94-100% retention),
  does adding HTC help Macro ALONE? Answer: no reliable effect either way
  (exact 4/8 split, deltas an order of magnitude smaller).

Taken together: there's no version of "add HTC's internal states" that
shows a reliable positive effect. The earlier variant's negative result
is *not* fully explained away by the population artifact (this experiment
still shows several combos with real, if inconsistent, movement in both
directions) — but it's also not confirmed as a clean, population-independent
degradation, since the matched-population version here nets out at zero.

## File map

```
macro_internal/
  run_macro_internal.py               analysis code (macro extraction from verbalized rollout + HTC join)
  NARRATIVE.md                         this file
  results/
    macro_internal_per_combo.csv       per (combo, feature_set): auroc/e_aurc/brier/n_rows/n_tasks
    macro_internal_summary.csv         mean +/- std across 8 combos
    macro_internal_significance.csv    paired t-test/Wilcoxon
```
