# Monitoring Web Agents Without Internal Signals

Code and data for **"Monitoring Web Agents Without Internal Signals: Observable
Trajectories and Key-Step Supervision."**

Given an evolving web-agent trajectory prefix, we predict whether execution is
still on track or tending toward failure, using only signals an observer can
see — no token logits, no hidden states. Two representations:

* **Macro** — prefix-level behavioural descriptors from the action sequence,
  execution metadata, and the agent's own reasoning text (31 features).
* **Micro** — consistency of intention, action, and anticipated state change
  across `N=10` repeated black-box queries at the same context (18 features).

Supervision is anchored on an annotated **key step** — the first uncorrected
critical error — rather than propagating the final outcome to every prefix, so
valid early prefixes of failed trajectories stay labelled on track.

Evaluated on WebArena-Lite and Online Mind2Web with five backbones
(Claude Sonnet 4.6, GPT-5.2, GPT-5.4-nano, Kimi 2.5, Qwen3-VL-30B):
2,183 trajectories, 28,663 prefixes.

## Quick start

```bash
pip install -r requirements.txt
bash reproduce.sh          # regenerates every table and figure, ~15 min, CPU only
```

No raw trajectories, GPU, or API keys are needed: the repository ships the
feature matrices, key-step labels, and baseline scores the paper's numbers are
computed from. `bash reproduce.sh --retrain` additionally refits the Ours/L2
predictors from the feature matrices before regenerating everything.

Figures land in `figures/`, named by their number in the paper; tables and
intermediate CSVs land in each analysis's own `results/` directory. (The plotting
scripts write under the working names the experiments had while the paper was
being written — `collect_figures.py`, which `reproduce.sh` calls last, gathers
them under `figNN_` names that match the published numbering.)

## Where each result comes from

Numbering follows the published paper.

| Paper | What it shows | Run |
|---|---|---|
| **Table 1** | Main results: Ours vs. HTC-Full, Stacked L2, verbalized confidence | `code/l2_training/` + `code/baseline_calculation/` + `code/htc/`; tidy form in `main_results_table.csv` |
| **Table 2** | Early-cut operating points at fixed false-cut budgets | `code/generalization/detection_false_cut_analysis/run_detection_false_cut_regions.py` |
| **Table 3** | LLM-judge run-to-run stability of the key-step labels | `code/llm_judge_consistency/prefix_label_consistency.py` |
| **Table 4** | Failure-cause distribution | `code/failure_cause_distribution/run_failure_cause_distribution.py` |
| **Table 5** | Prefix-time AUROC by region | `code/generalization/prefix_time_analysis/run_prefix_time_regions.py` |
| **Table 6** | Full scalar-baseline comparison | re-layout of `main_results_table.csv` |
| **Tables 7–8** | Micro sampling budget and cost trade-off | `code/n_sampling_cost/run_n_sampling_ablation.py` |
| **Figure 1–2** | Concept and pipeline schematics | hand-drawn (TikZ / draw.io), no code |
| **Figure 3** | False cuts under key-step vs. final-outcome supervision | `code/keystep_ablation/run_feature_set_false_alarm.py` → `regenerate_paper_success_false_alarm_fig.py` |
| **Figure 4** | Detection / false-cut trade-off | same script as Table 2 |
| **Figure 5** | Cross-category generalization, size-matched | `code/generalization/sizematched_control/run_sizematched_control.py` |
| **Figure 6** | Case study: how Macro and Micro divide the work | `case_study/plot_f10_case_micro_hardcoded.py` |
| **Figure 7** | Clustering robustness of the Micro features | `code/clustering_robustness/plot_f5_clustering_metrics.py` |
| **Figure 8** | Case study: observable vs. internal signals | `case_study/plot_f10_case_study_hardcoded.py` |
| **Figure 9** | Grouped leave-one-cluster-out feature ablation | `code/feature_importance/run_feature_importance_ablation.py` |
| **Figure 10** | Prefix-time discrimination curves | same script as Table 5 |

Supporting analyses that back claims in the text rather than a numbered artifact:
`code/macro_internal/` (Macro recomputed on the same execution HTC's internal
features come from — the pairing behind Figure 8), `code/feature_importance/run_family_significance_analysis.py`
(the paired tests in appendix §F), and
`code/generalization/prefix_time_analysis/htc_shuffle_ablation/`.

## Layout

```
data/                  feature matrices + derived per-task companions   see docs/DATA.md
keystep_labels/        key-step and failure-cause annotations
reproduction_output/   precomputed baseline scores and the canonical Ours/L2 runs
code/                  training, baselines, and one directory per analysis
case_study/            the two per-trajectory case figures and their inputs
pipeline/              upstream rollout → features → annotation (not needed to reproduce)
figures/               output directory, one file per published figure
MANIFEST.sha256        checksums for every released data file (verify_data.py)
```

`MANIFEST.sha256` pins the size and checksum of every released data file;
`python3 verify_data.py` checks them. The `NARRATIVE.md` files inside individual
analysis directories are the original working notes for that experiment — they
record why it is set up the way it is, and occasionally reference intermediate
artifacts that are not part of this release.

The predictor itself is `code/l2_training/train_risk_model_temp_sca.py`
(`Logistic_L2`, task-level 5-fold CV, temperature calibration); feature
construction is `code/l2_training/build_feature_matrix.py` with column
definitions in `code/l2_training/feature_definitions.json`.

## Reproducibility

Seeds alone are not enough — multi-threaded BLAS can move the last digits. Pin
the thread count before launching python (`reproduce.sh` does this):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=42
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the environment the
released numbers were produced with, and for what was checked against the
published artifacts.

## Known deviations

Two places where rerunning this code does not land exactly on the printed page.
Both are documented rather than papered over.

1. **Table 4, Online Mind2Web column.** The WebArena-Lite column reproduces
   exactly, module and sub-cause counts alike (Action 307 / 29.8%, Planning
   382 / 37.1%, Reflection 272 / 26.4%). The Mind2Web column does not: the
   released annotations give 2,089 annotated failures against the ~2,335 behind
   the published column, and module shares shift by up to 1.7 points
   (e.g. Reflection 25.9% here vs. 27.6% printed). The published Mind2Web counts
   came from an annotation snapshot that is not the one released here. The
   qualitative claim the table supports — Planning, Action, and Reflection each
   account for roughly a third of failures — holds under both.
2. **Table 5 caption.** The generated `.tex` reproduces every number in the
   table; only the caption differs, having been rewritten for the camera-ready.

One naming note: appendix Table 3's run-to-run stability numbers come from the
three-pass check in `code/llm_judge_consistency/results/gpt52_*`, i.e. the
protocol re-run through **GPT-5.2**. Production annotation used Gemini-3.5-Flash,
and the `temp0_*` results in the same directory are that judge's own three-pass
run, which is less stable (mean key-step distance 3.74 steps vs. 1.75).

## Paper

arXiv:2609.02057

## License

Code (`code/`, `case_study/`, `pipeline/`, and the top-level scripts) is released
under the **MIT License** — see [`LICENSE`](LICENSE).

Data (`data/`, `keystep_labels/`, `reproduction_output/`) is released under
**CC BY 4.0** — see [`data/LICENSE.md`](data/LICENSE.md). Those files are derived
from rollouts on WebArena-Lite and Online-Mind2Web; reuse of anything derived
from those benchmarks remains subject to the original datasets' own terms.
