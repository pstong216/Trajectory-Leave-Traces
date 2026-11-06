# Monitoring Web Agents Without Internal Signals

Code and data for **"Monitoring Web Agents Without Internal Signals: Observable
Trajectories and Key-Step Supervision"** ([arXiv:2609.02057](https://arxiv.org/abs/2609.02057)).

## About the paper

Web agents act over many steps, and a single early mistake can silently derail an
entire run. A monitor that could tell, partway through, whether execution is still
on track would let a system stop, retry, or hand off to a human before the damage
is done. Most existing uncertainty methods need token logits or hidden states,
which are unavailable for API-served agents.

We study monitoring from **observable signals only**. Given a trajectory prefix,
we predict whether the run is on track or heading toward failure, using what an
outside observer can see:

* **Macro features** (31) — prefix-level behavioural descriptors computed from the
  action sequence, execution metadata, and the agent's own reasoning text.
* **Micro features** (18) — how consistent the agent's intention, action, and
  anticipated state change are across `N=10` repeated black-box queries at the
  same context.

**Key-step supervision.** Instead of copying the final outcome onto every prefix,
we annotate the **key step**, the first uncorrected critical error, and label a
prefix as risky only once the run has reached it. Early prefixes of a failed run
that are still sound stay labelled on track, so the monitor is not trained to
raise alarms on valid behaviour.

We evaluate on WebArena-Lite and Online Mind2Web with five backbones
(Claude Sonnet 4.6, GPT-5.2, GPT-5.4-nano, Kimi 2.5, Qwen3-VL-30B), covering
2,183 trajectories and 28,663 prefixes. The predictor is a logistic regression
with L2 regularisation, trained with task-level cross-validation and temperature
calibration.

## Highlights

* **No internal signals needed.** Averaged over the 10 benchmark-backbone
  settings, our Macro+Micro predictor reaches **0.770 AUROC**, against 0.620 for
  verbalized confidence and 0.47 to 0.58 for token-logprob and entropy scalars.
* **Competitive with methods that read internal signals.** On the 8 settings
  where the backbone exposes logits, HTC-Full (which uses them) scores 0.756 AUROC;
  our Macro features alone, on the same population, score 0.786.
* **Key-step supervision cuts false alarms.** Supervising on the key step instead
  of the final outcome lowers the false-cut rate at every threshold, on both
  benchmarks (figure below, left).
* **Better early detection at a fixed false-cut budget** than the scalar baselines
  and Stacked L2, and on par with or above HTC-Full (figure below, right).

<p align="center">
  <img src="figures/fig03_success_false_alarm.png" width="48%" alt="False cuts under key-step vs. trajectory-level supervision">
  <img src="figures/fig04_detection_false_cut.png" width="48%" alt="Detection rate vs. false-cut rate">
</p>

## What this repository contains

```
code/                  training, baselines, and one directory per analysis
  l2_training/           feature construction and the L2 risk predictor
  baseline_calculation/  scalar baselines (e.g. verbalized confidence)
  htc/                   the HTC-Full baseline
  keystep_ablation/      key-step vs. final-outcome supervision
  generalization/        cross-category, early-cut, and prefix-time analyses
  clustering_robustness/ robustness of the Micro clustering
  feature_importance/    grouped feature ablations and significance tests
  n_sampling_cost/       Micro sampling budget and cost
  llm_judge_consistency/ run-to-run stability of the key-step annotation
  failure_cause_distribution/  distribution of annotated failure causes
  macro_internal/        Macro features vs. internal-signal baselines
  baseline_extension/    cumulative-baseline helper
case_study/            the two per-trajectory case-study figures
data/                  feature matrices and per-task companion files
keystep_labels/        key-step and failure-cause annotations
reproduction_output/   precomputed baseline scores and trained-predictor outputs
pipeline/              upstream rollout, embedding, and annotation code (reference)
figures/               figures from the paper
docs/                  data and environment notes
```

The predictor is `code/l2_training/train_risk_model_temp_sca.py`; feature
construction is `code/l2_training/build_feature_matrix.py`, with column
definitions in `code/l2_training/feature_definitions.json`.

The repository ships the feature matrices and key-step labels the paper's results
are computed from, so no raw trajectories, GPU, or API keys are needed to run the
analysis code. The raw rollouts (about 3 GB) are hosted separately; see
[`docs/DATA.md`](docs/DATA.md) for the data layout and download links.

## Running the code

```bash
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` regenerates the tables and figures from the released data on CPU.
Environment details are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Paper

arXiv:2609.02057

## License

Code (`code/`, `case_study/`, `pipeline/`, and the top-level scripts) is released
under the **MIT License**, see [`LICENSE`](LICENSE).

Data (`data/`, `keystep_labels/`, `reproduction_output/`) is released under
**CC BY 4.0**, see [`data/LICENSE.md`](data/LICENSE.md). Those files are derived
from rollouts on WebArena-Lite and Online-Mind2Web; reuse of anything derived
from those benchmarks remains subject to the original datasets' own terms.
