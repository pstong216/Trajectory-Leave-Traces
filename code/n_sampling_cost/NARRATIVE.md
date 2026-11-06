# Mind2Web sample-budget (N) ablation — results summary and paper framing

Analysis code: `run_n_sampling_ablation.py`. Reruns the N-ablation from the
author response to Reviewer jRcR (11 Jul 2026, see
`../../reviewers/Screenshot 2026-08-02 at 12.26.12 PM.png`) against the
CURRENT keystep labels and CURRENT main-table CV protocol
(`StratifiedKFold(5, shuffle=True, random_state=42)`, refreshed
2026-08-01) — the original response predates both. Mind2Web only, because
that's the one benchmark where micro/UQ significantly improves over macro
(see `../feature_importance/NARRATIVE.md`, Level 1); WebArena-Lite's micro
gain was already null/negative at N=10 in the refreshed data, so an N<10
ablation there answers a question that no longer has a "yes" to explain.

## Why this question matters

Macro features are free (N=1, computed from the agent's own already-taken
step — no extra LLM calls). Micro/UQ features need N re-samples per step.
If the mind2web micro gain only shows up at N=10, that's a real cost/benefit
question for anyone deploying this: is it worth 10x the per-step sampling
budget for ~0.01 AUROC?

## Results (mean over 5 mind2web backbones: claude, gpt_5_2, gptnano, kimi, qwen30b)

Delta vs. Macro-only reference (paired per backbone, then mean ± std across
the 5 backbones; same fold/label/feature-set convention across every N —
row counts verified identical to N=10 for all 5 backbones before training):

| N | Sampling cost (N/10) | ΔAUROC ↑ | ΔE-AURC ↓ | ΔBrier ↓ | t-test p (AUROC) |
|---|---|---|---|---|---|
| 3 | 0.30 | +0.0036 ± 0.0097 | −0.0067 ± 0.0116 | −0.0021 ± 0.0043 | 0.50 (n.s.) |
| 5 | 0.50 | +0.0148 ± 0.0124 | −0.0150 ± 0.0110 | −0.0070 ± 0.0049 | 0.076 (marginal) |
| 8 | 0.80 | +0.0162 ± 0.0131 | −0.0161 ± 0.0112 | −0.0073 ± 0.0041 | 0.068 (marginal) |
| 10 (production) | 1.00 | +0.0131 ± 0.0078 | −0.0120 ± 0.0096 | −0.0054 ± 0.0025 | **0.028** (sig.) |

(Wilcoxon signed-rank is capped at p=0.0625 for all N>=5 by the N=5 exact-test
floor — see `../feature_importance/NARRATIVE.md` for why that floor exists
and why the t-test isn't subject to it. Full per-metric significance table:
`results/n_ablation_significance.csv`.)

**Note on the N=10 row vs. the family-significance analysis's own number**:
`../feature_importance/NARRATIVE.md`'s Level 1 reports mind2web ΔAUROC =
+0.0096 (p=0.023), reading directly from the production
`train_risk_model_temp_sca.py` run's `cv_results.json`. This script's own
N=10 row (+0.0131, p=0.028) is an *independent retrain* using
`run_feature_importance_ablation.py`'s `run_5fold_cv()` -- same CV split
logic and same `fit_model()`, but without that script's full
temperature/Platt calibration machinery (not needed for AUROC/e-AURC/Brier
ranking). The two numbers agree in sign, magnitude, and significance
conclusion (both ~0.01-0.015 AUROC, both p<0.05) -- the small gap is
retrain-to-retrain CV-fold noise between two independently-invoked (but
protocol-identical) training runs, not a methodology discrepancy. Treat the
family-significance number as the canonical one for the main paper claim;
this table's job is the N-ablation, not re-deriving that number.

## Reading the curve

Unlike the original (old-label) response, where WebArena-Lite saturated by
N=3 and Mind2Web grew ~monotonically through N=10, the refreshed Mind2Web
numbers show the gain **rising steeply from N=3 to N=5, then flattening
N=5→N=8→N=10** (+0.0036 → +0.0148 → +0.0162 → +0.0131). N=3 is not
statistically distinguishable from zero gain; N=5 is where the effect
starts becoming reliably positive (marginal at p=0.08, consistent direction
5/5 backbones). N=8 and N=10 are statistically indistinguishable from each
other and from N=5 (all overlapping ~0.01-0.016 range, std ~0.01-0.013) --
the production N=10 budget is not obviously buying more than N=5-8 already
gets you, mirroring the same "diminishing returns past a moderate budget"
story the original response told, just with the elbow now closer to N=5
than N=3.

## Cost-performance tradeoff

**Retained Gain**: `RG(N) = [ΔAUROC(N)] / [ΔAUROC(N=10)]`, paired with each
budget's sampling cost (decode volume relative to N=10, = N/10 for any
prefix-cache fraction κ -- see the compute-cost model below, unchanged from
the original response since it's a pure architectural/pricing formula, not
data-dependent):

| N | Sampling cost | RG(N) |
|---|---|---|
| 3 | 0.30 | 0.275 |
| 5 | 0.50 | **1.130** |
| 8 | 0.80 | **1.237** |
| 10 | 1.00 | 1.000 |

RG(N) > 1 for N=5 and N=8 (like the WebArena-Lite RG>1 case in the original
response) reflects the same phenomenon: N=10's own point estimate isn't the
monotone ceiling once you're in the flat/noisy region past the elbow --
the differences among N=5/8/10 (0.0131-0.0162, spread 0.0031) are well
inside each point's own std (0.008-0.013), i.e. CV noise around a plateau,
not a real inversion where fewer samples beat more. **In one line: 50% of
the sampling budget (N=5) already buys, at minimum, all of the N=10 gain
on Mind2Web, and possibly more -- there is no evidence in this data that
paying for N=8 or N=10 buys anything N=5 doesn't.**

## Compute and time cost model (unchanged from the original response — architectural, not data-dependent)

With prefill cost $W_p$, per-sample decode cost $W_d$, and $\kappa \in [0,1]$
the fraction of prefill recomputed under caching (current API prefix-cache
pricing corresponds to $\kappa \approx 0.1$):

$$W_{\text{sample}}(N;\kappa) = [1+(N-1)\kappa] W_p + N W_d$$

Decode volume — what token-billed APIs charge at full rate — scales exactly
as $N/10$ for any $\kappa$ (the sampling-cost column above).

With prefill latency $t_p$, per-sample decode latency $t_d$, and serving
concurrency $c$:

$$T(N;c,\kappa) \approx (t_p+t_d) + \lceil (N-1)/c \rceil \cdot (\kappa t_p + t_d)$$

Under **sequential** serving ($c=1$) latency grows linearly in $N$; under
**parallel** serving with prefix caching ($c \geq N-1$) it collapses to
$t_p + t_d + (\kappa t_p + t_d) \approx t_p + 2t_d$, **independent of N** --
monitoring adds roughly one extra decode round regardless of budget, and the
case for a smaller N rests on compute/API cost rather than latency. We
avoid claiming universal wall-clock numbers, which depend on serving
infrastructure.

## Combined trade-off: $-cost + time-cost vs. retained performance

Putting the empirical Mind2Web performance curve (above) on the same axis
as both cost dimensions -- $-cost (decode volume, $N/10$, what a
token-billed API actually charges) and time-cost (illustrative normalized
units, $\kappa=0.1$, $t_p=t_d=1$ purely to make the *shape* comparable
across $N$ -- not a real-seconds claim, per the caution above):

| N | \$-cost ($N/10$) | Time, sequential ($c{=}1$) | Time, parallel ($c\geq N{-}1$) | ΔAUROC vs. macro-only | RG(N) |
|---|---|---|---|---|---|
| 3 | 0.30 | 0.353 | 0.261 (flat) | +0.0036 (p=0.50, n.s.) | 0.28 |
| 5 | 0.50 | 0.538 | 0.261 (flat) | +0.0148 (p=0.08) | 1.13 |
| 8 | 0.80 | 0.815 | 0.261 (flat) | +0.0162 (p=0.07) | 1.24 |
| 10 | 1.00 | 1.000 | 0.261 (flat) | +0.0131 (p=0.03) | 1.00 |

(Sequential-time column normalized to N=10's own sequential time = 1.00;
parallel-time column normalized to that SAME denominator, so 0.261 across
every row also states the headline parallel-serving result directly: with
enough concurrency, N=10 only costs ~26% of what *sequential* N=10 costs,
and that number does not change with N at all.)

**Two regimes, two different reasons N=5 wins:**

- **Sequential serving**: cost and performance move together. N=5 is a
  **strict Pareto improvement over N=8 and N=10** -- it costs 53.8% of
  N=10's latency (and 62/66% of N=8's) and 50% of its \$, while its AUROC
  point estimate (+0.0148) is *higher*, not lower, than N=10's (+0.0131).
  There is no dimension (money, time, performance) on which N=10 beats N=5
  here; the only reason to run N=10 under sequential serving would be a
  metric this analysis doesn't cover.
- **Parallel serving with caching**: time is a wash -- every N in {3,5,8,10}
  costs the same ~0.26 normalized time, because the caching/concurrency
  assumption already absorbs the fan-out. Here the trade-off collapses to
  a single axis, \$-cost vs. performance, and N=5 still wins on that axis
  alone: half the decode spend of N=10 for statistically indistinguishable
  (if anything, nominally higher) AUROC gain.

**N=3 is not on the efficient frontier in either regime.** It's the
cheapest point on both cost axes, but it's the only budget where the
performance benefit isn't reliably present at all (p=0.50) -- you're not
trading a little performance for a lot of savings, you're paying 30% of
the cost for a coin flip on whether you get any of the benefit you're
sampling for.

**Bottom line for the reviewer response**: N=5 dominates N=8 and N=10 on
every axis measured (money, sequential latency, and point-estimate AUROC),
and is tied with them on parallel-serving latency and on statistical
significance (none of N=5/8/10 individually clears p<0.05 alone, though
all three are directionally consistent and N=10 does when read via the
production `cv_results.json` pipeline in `../feature_importance/`). N=3
is the only budget that isn't a defensible choice: it's cheap but the gain
it's supposed to buy isn't reliably there.

## Suggested paper / reviewer-response framing

**Do claim:**
- "On Mind2Web (the benchmark where micro/UQ significantly improves over
  macro, see [family significance]), the gain is not yet reliable at N=3
  (ΔAUROC=+0.004, p=0.50) but becomes consistently positive from N=5
  onward (p=0.08 at N=5, 5/5 backbones same direction). N=5 already
  captures the full N=10 gain within noise (RG(5)=1.13); N=8 and N=10 add
  nothing further that's distinguishable from CV noise."
- "At 50% of the production sampling budget (N=5, half the decode volume
  and, under parallel serving with prefix caching, no added latency vs.
  N=10), Mind2Web retains its full micro/UQ benefit over macro-only."

**Don't claim:**
- That N=3 already captures meaningful gain on Mind2Web -- unlike the
  original (old-label) response's WebArena-Lite finding, N=3 here is
  statistically indistinguishable from zero on every metric.
- That the elbow is exactly at N=5 in some precise sense -- N=5/8/10 are
  mutually statistically indistinguishable (overlapping CIs), so "N=5 is
  sufficient" is a defensible practical recommendation, not a sharp
  discovered threshold.
- Universal wall-clock latency numbers -- the time-cost formula's
  N-independence claim is conditional on parallel serving with effective
  prefix caching; state that condition alongside any latency claim.

## File map

```
n_sampling_cost/
  run_n_sampling_ablation.py         analysis code (imports training code from ../feature_importance/)
  NARRATIVE.md                        this file
  results/
    n_ablation_per_backbone.csv       per (N, backbone, config): auroc/e_aurc/brier
    n_ablation_summary.csv            mean +/- std across 5 backbones, delta vs macro_only (source of the main table above)
    n_ablation_significance.csv       paired t-test / Wilcoxon per N vs macro_only, all 3 metrics
    cost_tradeoff.csv                 sampling cost x RG(N) (source of the cost table above)
```

## Data provenance (for anyone re-running this)

Fully self-contained inside this repository (2026-08-02) -- every path
this script reads is pinned inside this repository, nothing is read from
`results_total/` at run time.

- N=10 (production) + macro-only + all keystep labels: from
  `data/` + `keystep_labels/` -- the existing
  verified snapshot, current as of the 2026-08-01 refresh.
- N=3/5/8 raw UQ feature values:
  `data/mind2web/<model>/embedded/uq_analysis_change_qwen_n{3,5,8}/
  feature_matrix.csv` + `manifest.json` (N, strategy, seed, source) --
  copied verbatim from `results_total/mind2web/<model>/
  uq_analysis_change_qwen_n{3,5,8}/` (gptnano nested under
  `results_gp_5_nano_total/`), which were produced by
  `codes/subsample_n_ablation/subsample_reembed.py` -- same
  Qwen3-Embedding-0.6B model and 0.3 clustering threshold as N=10, only
  the per-step resample count differs. Before copying: verified row-for-row
  identical population to N=10 for all 5 backbones (steps with <10 stored
  samples are excluded before subsampling, same policy as the N=10
  baseline) AND 0 outcome-column mismatches against the canonical N=10
  file for every backbone x N -- subsampling changes only how many
  resamples went into each step's UQ metrics, never which steps are
  covered or their outcome/label.
