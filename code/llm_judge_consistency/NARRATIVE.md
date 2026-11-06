# LLM-as-judge self-consistency check

Code: `annotate_keystep_azure.py` / `annotate_keystep_azure_grouped.py` (annotation
runners) + `analyze_consistency.py` (raw keyStep/failureCause metrics) +
`prefix_label_consistency.py` (prefix-LABEL metrics, see below). Depends on
`annotate_keystep_gemini.py` for the shared taxonomy/prompts/trajectory loader
(`SOURCE_CONFIG`, `load_webarena_trajectory`, `INVARIANT_SYSTEM_INSTRUCTION`,
`JUDGE_SYSTEM_INSTRUCTION`) -- copied here for reference; trajectory data itself
(`--base_dir`) is **not** duplicated in this folder and still resolves against
`results_total/annotation/data/` (1.2GB, the live trajectory corpus).

## Motivation

The production `annotation/` pipeline (Gemini-based) auto-labels every failed
trajectory with `keyStep` + `failureCause` (16-code taxonomy) via a two-call
(invariant-analysis -> judge) LLM-as-judge pipeline. Before treating these labels
as ground truth for downstream mechanism analysis, we need to know how stable the
judge's output is under repeated, independent calls with the same model and
prompt -- i.e. does the judge agree with *itself*.

## Setup

- Stratified random sample of 150 tasks (of 1030 total already annotated by
  Gemini), proportional to each of the 10 webarena sources
  (`sample_consistency_150.py`, seed=42) -> `results/sample_150.json`.
- Same two-call pipeline, reusing the exact taxonomy/system-instruction text from
  `annotate_keystep_gemini.py` verbatim, run against Azure OpenAI, 3 independent
  times per configuration below. No shared context between passes.
- `analyze_consistency.py --passes <dir1>,<dir2>,<dir3> --out <report.json>`
  compares the passes only against each other (no comparison to the original
  Gemini labels -- purely self-consistency).

## Results (n=150 unless noted)

| Config | failureCause exact (3/3) | majority (>=2/3) | Fleiss' kappa | keyStep exact (3/3) | keyStep tight (<=1) | keyStep mean |diff| |
|---|---|---|---|---|---|---|
| gpt-5.4-nano, temp=default (bug) | 20.7% | 70.7% | 0.256 | 34.7% | 42.0% | 4.36 |
| gpt-5.4-nano, temp=0.0 | 32.7% | 86.0% | 0.400 | 42.7% | 48.7% | 3.74 |
| gpt-5.6-sol, temp=model-default, **n=25/150 partial** | 44.0% | 92.0% | 0.513 | 48.0% | 60.0% | 2.51 |
| **gpt-5.2, temp=0.0, grouped-by-task, n=150** | **46.7%** | **92.7%** | **0.501** | **58.7%** | **68.7%** | **1.75** |

- `results/tempdefault_pass{1,2,3}/` + `tempdefault_consistency_report.json`
- `results/temp0_pass{1,2,3}/` + `temp0_consistency_report.json`
- `results/gpt56_pass{1,2,3}/` + `gpt56_partial_consistency_report.json`
- `results/gpt52_pass{1,2,3}/` + `gpt52_consistency_report.json`

### Bug caught mid-experiment (gpt-5.4-nano)

The first run of `annotate_keystep_azure.py` never set `temperature`, unlike
`annotate_keystep_gemini.py` (which explicitly uses `temperature=0.0`) -- so it
silently ran at Azure's default (effectively temperature=1). All 3 passes were
rerun with `temperature=0.0` explicitly set. Both runs completed 150/150 with
zero errors; the two rows above compare them directly.

### gpt-5.6-sol run: partial (n=25/150)

Tried as the "larger deployment" follow-up. The run died mid-way on a sustained
DNS resolution outage (`Failed to resolve
'<your-resource>.services.ai.azure.com'`) after completing only 3 of the 10
webarena sources (25 items). Numbers above are **not a clean 150-item
comparison point** -- kept in the table for reference only, superseded by the
gpt-5.2 row below.

### gpt-5.2 run: grouped-by-task execution for cache locality

Re-ran the same check with `gpt-5.2` (the confirmed-working Azure deployment on
this project) via `annotate_keystep_azure_grouped.py`, which changes the
*execution order*: instead of running pass1 over all 150 items, then pass2, then
pass3 (spacing repeats of the identical call1/call2 prompt ~150 API calls
apart), the 3 passes for each task fire back-to-back before moving to the next
task -- so Azure's prompt cache can actually hit on the shared prefixes. Same
manifest, taxonomy, prompts, temperature=0.0. All 150/150 items completed on
all 3 passes with zero errors.

Pairwise (temp=0): pass1x2 exact=60.7%/kappa=0.482, pass1x3
exact=64.0%/kappa=0.526, pass2x3 exact=61.3%/kappa=0.497 -- uniform across
pairs, so the residual disagreement reads as genuine judge noise rather than
one anomalous pass.

## Prefix-label consistency (the metric that actually matters for training)

`keyStep` and `failureCause` are never fed to the risk model directly. Training
derives a per-step binary label from `keyStep` alone
(`l2_training/train_risk_model_temp_sca.py::apply_keystep_labels`, lines
182-214): for a failed task, `label[step_k] = 0` if `step_k < keyStep` else
`1` (success tasks: `label = 0` for every step; unannotated failed tasks:
`label = 1` for every step, i.e. raw outcome). Two judge passes whose
`keyStep` differs by a few steps can still emit **identical prefix labels for
most of the trajectory** -- only the steps strictly between the two `keyStep`s
flip. Raw keyStep exact-match (previous section) understates supervision
stability; this section measures the label sequences directly
(`prefix_label_consistency.py`).

Closed form (verified against brute-force per-step comparison on a random
sample): for a task of length `n` with two keySteps `a, b`, the fraction of
steps whose label agrees is `1 - |a-b|/n`. For all `N` passes, the fraction
where *every* pass agrees is `1 - (max-min)/n`. The **majority-vote label
sequence over N=3 passes is identical to just thresholding at the median
keyStep** (each step's majority label flips exactly at the median of the
three keySteps) -- so "combine judges by majority vote" and "use the median
judge's keyStep" are the same operation at the label level.

| Metric | gpt-5.4-nano (temp=0, n=150) | gpt-5.2 (grouped, n=150) |
|---|---|---|
| **Pairwise agreement, macro (mean of 3 pairs) — report this one** | **~79.8%** | **~90.9%** |
| Pairwise agreement, micro (step-weighted, robustness check only) | ~77.4% | ~89.4% |
| **3-way full agreement, macro** | **69.8%** | **86.4%** |
| 3-way full agreement, micro | 66.0% | 84.1% |
| Per-pass agreement with median-derived label (macro, mean of 3) | ~89.9% | ~95.5% |

**Report macro, not micro.** Every raw keyStep/failureCause statistic in the
section above is task-level (one vote per task); a step-weighted (micro)
number for just this metric would mix units of analysis inside the same
narrative without that being visible from the percentages alone. Steps within
one trajectory also aren't independent draws -- the label sequence is a
monotone 0/1 step function, so steps far from any pass's keyStep trivially
agree regardless of judge stability, and micro lets long trajectories
contribute disproportionately many such steps. Macro treats each annotated
task as one observation, matching the rest of this document. (Micro is not
the more flattering choice here either -- it's 1-2pp lower than macro in
every row above -- so this is a consistency call, not result-shopping.)

For comparison, raw keyStep exact-match (all 3 passes identical) was only
42.7% (gpt-5.4-nano) / 58.7% (gpt-5.2) -- the prefix-label view recovers most
of the apparent instability: even at gpt-5.4-nano's noisier keyStep
agreement, ~78-80% of individual step labels already agree pairwise, and each
pass agrees with the majority/median label ~90% of the time. At gpt-5.2,
per-pass agreement with the median label reaches ~95-96%, meaning any single
pass would already be a very close approximation of the majority-vote signal
used for training. This is the stronger, more direct argument for supervision
stability: the raw `keyStep` metric conflates "judges disagree occasionally
on 1-2 boundary steps" (which barely dents the label sequence) with "judges
disagree" in a way that implies the whole label sequence is unreliable, which
it is not.

Reports: `results/temp0_prefix_consistency_report.json`,
`results/gpt52_prefix_consistency_report.json`.

## Interpretation

- **Fixing `temperature=0.0` explains a real chunk of the gpt-5.4-nano
  instability but not all of it.** kappa moved 0.256 -> 0.400, majority
  agreement 71% -> 86%. But even at temp=0, pairwise failureCause exact match
  topped out at ~50% for that model -- a large chat model with internal
  reasoning tokens is not a simple deterministic function of temperature.
- **gpt-5.2 is meaningfully more self-consistent than gpt-5.4-nano across every
  metric** (kappa 0.40 -> 0.50, keyStep exact match 42.7% -> 58.7%, mean keyStep
  drift more than halved: 3.74 -> 1.75 steps). Directionally consistent with
  "stronger/more expensive judge model -> more stable judge," but **not a clean
  single-variable test**: gpt-5.2 differs from gpt-5.4-nano in more than model
  size, and the execution order changed too (grouped-by-task vs sequential).
  Grouping shouldn't affect *content* (each API call is independent and
  stateless -- only latency/cache-hit rate changes), but it is a confound worth
  flagging rather than silently absorbing into "gpt-5.2 is better."
- **keyStep consistently lags failureCause** in every configuration. It is a
  *derived, sequential* quantity: it depends on Call 1's fresh per-step
  `correct`/`recovered` judgments (rerun every pass) feeding into Call 2's "scan
  from step 0, take the first unrecovered violation" rule. A single step early
  in the trajectory flipping its recovered/unrecovered judgment cascades into a
  large keyStep shift, whereas `failureCause` is one holistic judgment per pass
  with no such compounding dependency.
- **Bottom line for downstream use, two audiences.** For `failureCause`-consuming
  *mechanism* analyses (Exp A, F, etc.): even in the best configuration measured
  (gpt-5.2), failureCause 3/3 exact match is still under 50% and pairwise kappa
  tops out at 0.53 ("moderate") -- treat any single-pass `failureCause`
  annotation as noisy, prefer majority-vote-of->=3 passes or soft labels. For
  the *training* signal (per-step prefix labels derived from `keyStep`): the
  picture is materially better -- gpt-5.2 pairwise per-step label agreement is
  ~89-91% and any single pass already agrees with the majority/median label
  ~95-96% of the time (see "Prefix-label consistency" above), because most of
  each trajectory sits far from the disputed boundary. keyStep noise is real
  but its practical impact on the actual supervision signal is much smaller
  than the raw exact-match numbers suggest.

## Open follow-ups

1. Re-run the gpt-5.6-sol pass to completion (grouped-by-task, like the gpt-5.2
   run) once network reliability to the Azure endpoint is no longer a concern,
   to get a clean 150-item number for the largest deployment.
2. Run the same 3x self-consistency check on Gemini itself -- confirm
   `annotate_keystep_gemini.py`'s existing `temperature=0.0` setting actually
   delivers materially better consistency than what we see here, since that's
   the model the production annotations were built with.
3. Add a strict taxonomy validator (reject/retry on unmapped `failureCause`)
   instead of the current substring-fallback -- one malformed label
   (`"NavigationFailure:RetrievalFailure"`, not in the 16-code taxonomy) was
   observed in the pre-fix gpt-5.4-nano run (1/450 calls).
4. Investigate whether reformulating keyStep detection to be less
   sequentially-dependent on Call 1's noisy per-step judgments (e.g. deriving it
   from a single holistic pass rather than a scan-and-stop rule) would reduce
   its outsized instability relative to failureCause.
