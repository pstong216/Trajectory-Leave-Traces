# Upstream feature pipeline

This directory documents how the released feature matrices and key-step labels
were produced from raw agent rollouts. **It is not needed to reproduce any
number in the paper** — `reproduce.sh` starts from the derived data in `data/`
and `keystep_labels/`. The scripts are included so the construction of the
representations is inspectable, and so the pipeline can be re-run on a new
benchmark or backbone.

These steps require the raw rollouts (not redistributed here — see
[`../docs/DATA.md`](../docs/DATA.md)), a GPU for the sentence encoder, and API
credentials for the LLM judge. They are provided as-is.

## Stage 0 — Agent rollout (external)

Trajectories were collected with a ReAct-style agent on
[WebArena-Lite](https://github.com/THUDM/VisualAgentBench) (Docker) and
[Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web) (live web),
across five backbones. At every step the agent is additionally queried `N = 10`
times under the structured-output prompt of §3 to produce the Micro samples
(intention, executable action, anticipated state change). Rollout produces, per
task, the action/observation log plus `step_<k>_samples.json`.

## Stage 1 — Micro features: embed, cluster, compute uncertainty

Sampled intention and state-change strings are embedded and agglomeratively
clustered (cosine distance threshold `δ`); cluster ids become the discrete
outcomes of the empirical distribution, from which the 11 HLVM uncertainty
indicators are computed.

| script | role |
|---|---|
| `reembed_hlvm_variants.py` | production path — any of `qwen` / `minilm` / `bge_m3` at any `δ`. The paper uses `--model qwen --threshold 0.3`; the other eight settings are the robustness grid behind appendix Figure 7. |
| `reembed_qwen.py` | Qwen-only variant, kept because it is what the canonical run actually invoked |
| `run_webarena_embedding_grid.sh`, `run_mind2web_embedding_grid.sh` | drive the full encoder × threshold grid |
| `../code/l2_training/compute_hlvm_uq_multivariable.py` | the original MiniLM implementation the two above generalise |

```bash
python3 pipeline/reembed_hlvm_variants.py --source webarena/gpt_5_2 --model qwen --threshold 0.3
```

Writes `uq_analysis_change_<encoder>/thr_<δ>/<task_id>/step_<k>_hlvm_uq.json`.

## Stage 2 — Macro + Micro feature matrix

`../code/l2_training/build_feature_matrix.py` emits one row per `(task_id, step_k)`
prefix: 31 Macro columns computed directly from the observed prefix, plus the
Micro indicators aggregated with `{at_k, cumul_mean, cumul_max}`.

```bash
python3 code/l2_training/build_feature_matrix.py \
    --source webarena/gpt_5_2 --uq_dir uq_analysis_change_qwen
```

Writes the `feature_matrix.csv` files released under `data/`. Column definitions
are in `code/l2_training/feature_definitions.json` and appendix §D of the paper.

## Stage 3 — Key-step annotation

`../code/llm_judge_consistency/annotate_keystep_gemini.py` runs the two-stage
localisation protocol of appendix §C (step-wise violation scan, then
chronological root-cause judging) and assigns an AgentErrorTaxonomy failure
cause. Gemini-3.5-Flash was used for production annotation;
`annotate_keystep_azure.py` / `annotate_keystep_azure_grouped.py` run the same
protocol through Azure OpenAI and produced the three-pass stability check in
appendix Table 3.

Writes the `keystep_labels/<benchmark>/<model>_{baseline,ours}.json` files.

## Stage 4 — Train the prefix risk predictor

`../code/l2_training/train_risk_model_temp_sca.py` — this stage *is* reproducible
from the released data; see the repository README.
