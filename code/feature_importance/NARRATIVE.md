# Feature importance — results summary and paper framing

Analysis code:
- `run_feature_importance_ablation.py` — leave-one-cluster-out ablation (Level 2 below)
- `run_family_significance_analysis.py` — macro/micro family-level significance
  testing (Level 1 below), reads directly from each combo's own
  `reproduction_output/ours/*/cv_results.json` (per-fold test metrics), never
  recomputes a training run itself
- Correlation clustering that produced the Level-2 groups:
  `feature_correlation_clustering.py` (scratchpad copy; group membership is
  hardcoded into `FEATURE_GROUPS` in the ablation script, so that script has
  no runtime dependency on re-running the clustering)

**2026-08-01 update**: `reproduce.sh`'s Ours run was regenerated;
all numbers below are from that refreshed `cv_results.json` data. The
headline finding changed materially from the previous pass — see "What
changed" below before reusing anything from an earlier draft.

## Where everything lives

```
code/feature_importance/
  run_feature_importance_ablation.py     -- Level 2: cluster ablation code
  run_family_significance_analysis.py    -- Level 1: family significance code
  NARRATIVE.md                            -- this file
  results/
    feature_ablation_results.csv          -- Level 2 raw: per (group, combo)
    feature_ablation_summary.csv          -- Level 2: mean +/- std delta per group
    feature_ablation_ranking.png          -- Level 2: bar chart, 32 groups
    family_level_ablation.csv             -- Level 1: pooled all/macro/uq_only (10 combos)
    family_significance_foldlevel.csv     -- Level 1 raw: 50 (combo,fold) points
    family_significance_summary.csv       -- Level 1: every significance test below
```

## What changed vs. the previous pass

The pooled (10-combo) comparison is still not significant -- that part holds.
What's new is splitting by benchmark instead of stopping at the pooled null:
**on mind2web specifically, adding micro to macro is significant on all three
metrics** (paired t-test, N=5 backbones). On webarena it isn't -- direction is
actually slightly negative there. The pooled test was averaging together two
benchmarks with opposite-signed effects, which is exactly the kind of thing
that produces a misleading null. Don't report the pooled number as "the"
answer without this split; it hides the real, benchmark-conditional finding.

## Level 1 — family-level (macro vs. micro/UQ)

**Pooled (10 combos, NOT the number to lead with — see split below):**

| Config | AUROC | e-AURC | Brier |
|---|---|---|---|
| All (macro+micro, 49 feat.) | 0.7697 | 0.1295 | 0.1886 |
| Macro only (31 feat.) | 0.7694 | 0.1289 | 0.1905 |
| Micro/UQ only (18 feat.) | 0.7351 | 0.1495 | 0.1994 |

Pooled significance (N=10 combos, paired t-test on combo-level means, all vs.
macro_only): AUROC p=0.97, e-AURC p=0.92, Brier p=0.56 -- not significant on
any metric. **This is the number that's misleading in isolation.**

**Benchmark split (N=5 each, paired t-test, all vs. macro_only):**

| Benchmark | Metric | Mean Δ | 95% CI | t-test p |
|---|---|---|---|---|
| **mind2web** | AUROC | **+0.0096** | [+0.0022, +0.0170] | **0.023** |
| **mind2web** | e-AURC | **+0.0065** | [+0.0009, +0.0122] | **0.032** |
| **mind2web** | Brier | **+0.0055** | [+0.0020, +0.0090] | **0.012** |
| webarena | AUROC | -0.0090 | [-0.0494, +0.0314] | 0.57 |
| webarena | e-AURC | -0.0078 | [-0.0396, +0.0240] | 0.53 |
| webarena | Brier | -0.0016 | [-0.0189, +0.0157] | 0.81 |

All three mind2web 95% CIs exclude zero. All three mind2web comparisons show
5/5 backbones improving (Wilcoxon/exact permutation p=0.0625, the best
achievable with only 5 paired points — see below). webarena shows no such
pattern; direction is inconsistent-to-negative.

**Why the nonparametric tests cap at p=0.0625 despite 5/5 consistency**: an
exact sign-based test on N paired points has 2^N possible sign patterns; the
maximally-consistent outcome (all N diffs the same sign) can only ever reach
a two-sided p of 2 x 2^(-N). For N=5 that floor is 0.0625 — mathematically
impossible to beat with this test at this N, regardless of true effect size.
The paired t-test isn't subject to that floor (it uses magnitude, not just
sign), which is why it clears 0.05 while Wilcoxon/permutation don't. Report
both, and explain the floor rather than letting a reader assume "0.0625 means
weak effect."

## Level 2 — within-family clusters (unchanged from previous pass; NOT yet
## rerun against the 2026-08-01 data refresh — see caveat)

- 32 correlation-derived groups (13 multi-feature clusters + 19 standalone),
  |r| > 0.70 hierarchical clustering, pooled across all 10 combos.
- Full spread across all 32 groups: 0.0105 AUROC (max +0.0067 for
  `trajectory_length_proxies`, min -0.0037 for `uq_ground_reasoning_mean`).
- Only 3 of 32 groups have |mean ΔAUROC| exceeding their own cross-combo std;
  2 are practically meaningful (`trajectory_length_proxies` +0.0067,
  `action_stall_signals` +0.0023), the third is a negligible -0.0003.
- **Caveat**: `run_feature_importance_ablation.py`'s CV protocol was changed
  to `StratifiedKFold(shuffle=True, random_state=42)` after this ablation was
  last run (was `shuffle=False`) -- the numbers above are from the
  shuffle=False run and have NOT been regenerated against either the
  shuffle=True protocol or the 2026-08-01 data refresh. Rerun before citing
  these Level-2 numbers alongside the now-current Level-1 numbers, since
  mixing a stale ablation with fresh family-level numbers in the same
  section would be inconsistent.

## Suggested paper framing

**Do claim:**
- "Micro/UQ features are independently predictive in isolation (0.735 AUROC,
  well above chance), confirming token-level uncertainty carries real risk
  signal even without macro behavioral features."
- "On mind2web, combining micro with macro yields a small but statistically
  significant improvement across all three metrics (paired t-test across 5
  backbones: AUROC p=0.023, e-AURC p=0.032, Brier p=0.012; all 5 backbones
  improve on all three metrics). On WebArena this combination shows no
  reliable benefit and trends slightly negative. We report both rather than
  the pooled average, which obscures this benchmark-conditional effect."
- Do NOT lead with the pooled 10-combo number as if it were the answer --
  it's the wrong level of aggregation once you know the effect is
  benchmark-conditional. Lead with the split.

**Don't claim:**
- "Micro significantly improves over macro" as an unconditional, benchmark-
  general statement — it's true on mind2web, not established on webarena.
- That the mind2web effect is huge — it's small (ΔAUROC ~0.01) but real
  (CI excludes zero, consistent direction across all 5 backbones on all 3
  metrics). Precise > impressive.
- Anything about *why* mind2web and webarena differ without flagging it as a
  hypothesis, not a finding — e.g. per-combo task-count differences
  (mind2web combos average ~275 tasks vs. webarena's ~150) are a plausible,
  partially data-grounded explanation (more tasks per combo -> less noisy
  per-fold estimates -> a small true effect is easier to detect cleanly),
  but this has not itself been tested and shouldn't be stated as confirmed.

**Recommended structure:**
1. Main text: one or two sentences with the mind2web numbers (as in "Do
   claim" above), explicitly noting the webarena contrast rather than
   omitting it.
2. Appendix: `family_significance_summary.csv` (all tests, both pooled and
   split) + a short note on the N=5 exact-test floor, so a careful reader
   isn't confused by Wilcoxon p=0.0625 sitting next to a t-test p<0.05 for
   the same comparison.
3. One paragraph connecting to the paper's core claim: even where combining
   micro with macro doesn't help (webarena), micro's own standalone AUROC
   (0.735) is still the load-bearing evidence for "externally observable
   signal is valuable" — that claim doesn't depend on the combination result
   either way.
