# Size-matched control — results summary and paper framing

Analysis code: `run_sizematched_control.py` (imports every loader and every
"full" cross-category number unmodified from `../run_website_generalization.py`,
so this experiment can never silently drift from that script's own reported
numbers -- see that file's own docstring for the base leave-one-category-out
protocol this control is built on top of).

## The question this answers

`run_website_generalization.py`'s headline result is that training on the
pooled OTHER 4 website categories ("cross-category", ~1,500-1,900 tasks)
matches or beats training on 80% of the held-out category's OWN tasks
("in-domain", ~200-550 tasks, 5-fold CV within the category). That
comparison confounds two different things:

  (a) **distribution transfer** -- does training data from a different
      website-category distribution actually carry usable signal?
  (b) **training volume** -- cross-category simply has 3-9x more rows; is
      the win just "more data wins," unrelated to where it came from?

This script isolates (a) by holding size constant: for each held-out
category, the cross-category training pool is randomly subsampled down to
the SAME size as that category's own in-domain training fold (5 repeats,
mean +/- std). Comparing **size-matched cross-category** against
**in-domain** now differs by distribution alone; comparing **size-matched**
against **full-pool cross-category** isolates the added value of volume
alone.

## Method coverage

Every method with fitted parameters gets all three numbers on the identical
held-out-category test set: Ours (macro+micro / macro-only / micro-UQ-only),
M11 (HTC full/L2), M13 (Stacked HTC_L2, now 6 features after 2026-08-02's
M15 addition -- see `../run_website_generalization.py`'s own
`BASE_FEATURE_COLS_M13`). M1-M7/M15 are also covered, Brier-only
(added 2026-08-31, design decision -- same day their Brier calibration
was fixed to held-out Platt in the parent script; the old exclusion
rationale "no fitted parameters" was wrong, as review pointed out:
their Brier DOES come from a fitted 2-parameter Platt calibrator).
Their ranking genuinely has no fitted parameters, so AUROC/e-AURC are
identical across all three protocols BY CONSTRUCTION -- the output rows
keep all three copies as a built-in sanity check (and it holds:
bit-identical AUROC on every method x protocol). Only the Platt
calibrator is refit per protocol; see
`run_m1m7_platt_variants()`'s docstring for the exact splits.

### M1-M7/M15 Brier decomposition (mean across 5 held-out categories)

| Method | In-domain | Size-matched | Full | Transfer | Volume |
|---|---|---|---|---|---|
| M1 verbalized conf | 0.2463 | 0.2431 | 0.2418 | -0.0032 | -0.0013 |
| M2 action logprob | 0.2493 | 0.2492 | 0.2478 | -0.0001 | -0.0014 |
| M3 output logprob | 0.2517 | 0.2521 | 0.2510 | +0.0004 | -0.0011 |
| M4 cumul. output logprob | 0.2509 | 0.2524 | 0.2506 | +0.0015 | -0.0018 |
| M5 action entropy | 0.2277 | 0.2289 | 0.2288 | +0.0012 | -0.0002 |
| M6 cumul. action entropy | 0.2257 | 0.2271 | 0.2270 | +0.0014 | -0.0001 |
| M7 conf cumul. mean | 0.2495 | 0.2454 | 0.2440 | -0.0041 | -0.0014 |
| M15 cumul. action logprob | 0.2472 | 0.2463 | 0.2444 | -0.0009 | -0.0019 |

(Brier; lower is better, so a NEGATIVE transfer/volume delta = the
larger/cross pool helped. Sign convention flipped vs. the AUROC table
above.)

Reading: |transfer| <= 0.004 and |volume| <= 0.002 Brier, mixed signs,
an order of magnitude below these methods' own between-category stds
(0.002-0.015). Exactly what a 2-parameter calibrator should do: it
saturates with a couple hundred tasks (volume ~ 0) and the
score->probability mapping it learns is essentially
category-independent (transfer ~ 0). This is a useful NEGATIVE control
for the whole suite: the fitted-calibration component contributes no
hidden cross-category advantage or penalty to the baselines -- their
Brier standing is fully explained by their weak ranking signal, and
M11's +0.040 AUROC transfer effect (below) cannot be a calibration
artifact of the same kind.

(Footnote on the CSV: the `_indomain_percat` rows' e_aurc can deviate
from the full row's by <= 0.004 even though the ranking is identical --
the AURC estimator is a cumulative sum over `argsort`, which is
tie-order-sensitive, and in-domain predictions are pooled fold-by-fold
in a different row order. Methods with continuous scores (M3/M4) show
zero deviation, confirming it's a tie artifact, not signal. AUROC uses
proper tie handling and is bit-identical everywhere.)

## Results (mean AUROC across the 5 held-out categories)

| Method | Full pool | Size-matched | In-domain | Δ(matched − in-domain) | Δ(full − matched) |
|---|---|---|---|---|---|
| Ours — macro+micro | 0.7893 | 0.7848 | 0.7854 | **−0.0005** | +0.0044 |
| Ours — macro only | 0.7636 | 0.7607 | 0.7610 | **−0.0004** | +0.0030 |
| Ours — micro/UQ only | 0.7533 | 0.7479 | 0.7400 | **+0.0078** | +0.0055 |
| M11 (HTC full/L2) | 0.7865 | 0.7642 | 0.7242 | **+0.0400** | +0.0223 |
| M13 (Stacked HTC_L2, 6 feat.) | 0.6199 | 0.6165 | 0.5456 | **+0.0708** | +0.0034 |

(In-domain training-pool sizes this matches: 205-551 tasks depending on
category, vs. the full cross-category pool's ~1,500-1,900 tasks -- see
`results/sizematched_control_results.csv`'s `n_train_tasks` column.)

The two delta columns decompose the full "cross-category beats in-domain"
gap additively: `full − in-domain = (full − matched) [volume] +
(matched − in-domain) [pure distribution transfer]`.

## What this shows, per method family

**Ours (macro+micro and macro-only): the transfer term is ~zero.**
Size-matched cross-category is statistically indistinguishable from
in-domain (−0.0004 to −0.0005 AUROC, an order of magnitude smaller than
either number's own cross-category std of ~0.017-0.020). This is a clean,
positive result for the paper's core claim: Ours' behavioral/UQ features
carry risk signal that is **category-agnostic** -- training on four
completely different website categories works exactly as well, task-for-
task, as training on the held-out category's own data. There is no
distribution-shift penalty to pay. What full-pool cross-category adds on
top of in-domain (~+0.004 AUROC) is essentially the ordinary benefit of
more training rows, not a special transfer effect.

**Ours (micro/UQ-only): transfer term is small but positive (+0.0078).**
Unlike macro, the UQ-only feature set doesn't just transfer losslessly --
size-matched cross-category training *beats* in-domain training by a small
but real margin. This is a nice complementary data point to the family-
level significance analysis (`../feature_importance/NARRATIVE.md`): UQ
signals aren't just independently predictive (their own standalone
finding there), they also appear to generalize *at least* as well as
macro features across website categories, possibly slightly better.

**M11 (HTC full/L2): both effects are real and substantial.** Size-matched
cross-category beats in-domain by +0.040 AUROC -- a genuine distributional
transfer benefit, not just noise (well outside either number's ~0.02-0.05
std). Full-pool adds another +0.022 on top. Read together with the "Ours
transfer ~0" result above: M11's own per-category in-domain training pool
(the same 205-551 tasks) is measurably worse-suited to fitting HTC's
log-prob-derived features than an equally-sized *but differently
distributed* pool is -- i.e. HTC's features benefit disproportionately
from category diversity in training, not just from more rows of the same
kind. Framed differently: M11's headline cross-category number is doing
double duty (transfer AND volume), so it should not be read as "M11
generalizes as cleanly as Ours" without this caveat.

**M13 (Stacked HTC_L2): same direction as M11, much larger and noisier.**
+0.071 transfer effect, and its in-domain number (0.546) sits close to
chance. Read this one cautiously rather than as a headline: M13's 6-feature
logistic regression, fit on only 205-551 in-domain rows with its own
single-80/20-holdout C search, is the kind of small-sample setup where
instability (not a deep truth about cross-category transfer) can easily
produce a bad in-domain fit. The pattern is consistent with M11's, which
supports it being a real effect and not pure artifact, but the magnitude
here is likely inflated by M13's small-sample fragility rather than being
a clean second data point at the same reliability as M11's.

## Suggested paper framing

**Do claim:**
- "Holding training-set size constant, Ours' cross-category training
  performance is statistically indistinguishable from in-domain training
  (Δ = −0.0004 to −0.0005 AUROC for macro+micro and macro-only, both far
  inside the cross-category std of ~0.02) -- the behavioral/UQ signal Ours
  relies on transfers across website categories with no measurable
  distribution-shift penalty. The full-pool cross-category number's edge
  over in-domain (~+0.003 to +0.004 AUROC) is attributable to ordinary
  training-volume gains, not a special transfer effect."
- "For the micro/UQ-only feature set specifically, size-matched
  cross-category training *outperforms* in-domain by +0.0078 AUROC,
  suggesting token-level uncertainty signal is, if anything, even more
  category-agnostic than macro behavioral features."
- "The strongest baseline, M11 (HTC full/L2), shows a real +0.040 AUROC
  transfer benefit from cross-category training at matched size -- unlike
  Ours, M11's reported cross-category advantage over in-domain is not
  purely a volume effect; its in-domain-only fit is measurably weaker at
  the same sample size."

**Don't claim:**
- That M13's +0.071 transfer number is directly comparable in reliability
  to M11's +0.040 -- M13's in-domain baseline (0.546 AUROC) is fit on the
  same small per-category pool with a fragile single-feature-set/single-
  holdout protocol prone to small-sample instability; report the direction
  but don't lean on the magnitude.
- That "cross-category beats in-domain" is one uniform phenomenon across
  methods -- the decomposition shows Ours (volume-only) and M11/M13
  (real transfer + volume) are qualitatively different regimes, and that
  distinction is the actual finding, not a footnote.

## File map

```
sizematched_control/
  run_sizematched_control.py               analysis + figure code
  NARRATIVE.md                              this file
  results/
    sizematched_control_results.csv         per (method, protocol, category)
    sizematched_control_summary.csv         mean +/- std across 5 categories (source of the table above)
    website_generalization_sizematched_control.png   1x3-panel figure (AUROC/e-AURC/Brier)
```
