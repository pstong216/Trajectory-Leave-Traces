# Data

Everything needed to reproduce the paper's tables and figures is in this
repository (~110 MB). The raw agent rollouts they were derived from (~3.2 GB)
are not redistributed here; see [Raw rollouts](#raw-rollouts).

The unit of analysis is a **prefix**: one row per `(task_id, step_k)`, using only
information available up to step `k`.

| benchmark | backbones | trajectories | prefixes |
|---|---|---|---|
| WebArena-Lite | claude, gpt_5_2, gptnano, kimi, qwen30b | 774 | 10,491 |
| Online Mind2Web | claude, gpt_5_2, gptnano, kimi, qwen30b | 1,409 | 18,172 |

## `data/<benchmark>/<model>/`

### `embedded/uq_analysis_change_<encoder>/thr_<δ>/feature_matrix.csv`

The Macro + Micro feature matrix — the main input to the risk predictor.
67 columns: identifiers, 31 Macro features, and the Micro indicators aggregated
over the prefix. Column definitions: `code/l2_training/feature_definitions.json`
and appendix §D of the paper.

* `uq_analysis_change_qwen/thr_0p3/` is **canonical** — Qwen-0.6B encoder,
  clustering threshold 0.3. This is what Table 1 and every downstream analysis use.
* `webarena/gpt_5_2` additionally ships all nine `{qwen, minilm, bge_m3} × {0.2, 0.3, 0.4}`
  settings — the robustness grid behind appendix Figure 7.
* `mind2web/*/embedded/uq_analysis_change_qwen_n{3,5,8}/` are Micro features
  rebuilt from the first N of the same ten samples — appendix Tables 7–8.

### Derived companions

These three files are computed from the raw rollouts at packaging time so that
the analyses run without them. Each loader prefers the derived file and falls
back to the raw path when `raw/` is present, so results are unchanged either way.

| file | contents | used by |
|---|---|---|
| `task_outcomes.json` | `task_id → true task failure`, from the rollout's own ground truth | early-cut simulation (Figure 4, Table 2) |
| `category_map.json` | `task_id → website category`, from the WebArena page URL / the Mind2Web task text | cross-category generalization (Figure 5) |
| `htc_step_features.parquet` | the 48 token-logprob features of the HTC-Full baseline, per prefix | Figure 5, Table 6 |

`htc_step_features.parquet` is Parquet rather than CSV specifically so the
float64 values and the string `task_id` / `task_group` columns round-trip
exactly; a CSV round-trip shifted HTC's AUROC in the fifth decimal. There is no
file for `claude`, which exposes no logprobs — this is the `N/A` in the paper's
baseline tables, not a packaging omission.

## `keystep_labels/<benchmark>/`

Key-step annotations, one JSON per (model, role). Each entry carries `keyStep`
(the supervision boundary `k*`), `failureCause` (one of 16 sub-causes under five
AgentErrorTaxonomy modules), and a rationale.

Each model was rolled out twice: a plain run, and a separate *verbalized* run
re-prompted to also report self-reported confidence (which the M1 baseline
needs). Each rollout was annotated independently, so there are two passes per
model, and they disagree on `keyStep` for a majority of the tasks they share.
The two roles here are therefore **not interchangeable**:

| file | annotates | feeds |
|---|---|---|
| `<model>_ours.json` | the plain rollout | the Macro/Micro predictor (Ours) |
| `<model>_baseline.json` | the verbalized rollout | the baselines (M1–M14) |

Matching each label file to the rollout it actually annotates is the one step in
this project that silently produces wrong-but-plausible numbers when it goes
wrong, and the original filenames encode it inconsistently across the two
benchmarks (WebArena marks the plain-rollout pass with an `_m` suffix, Mind2Web
marks the verbalized one with `_v`). The `_ours` / `_baseline` names here have
that resolution already applied — read from these, never re-derive it.
`SOURCES.csv` records the role, original path, and sha256 of every file.

## `reproduction_output/`

Precomputed intermediates, so the analyses run without recomputing upstream:

* `baseline/<benchmark>/<model>/results/` — per-step scores for every scalar and
  HTC baseline (`per_step_scores.csv`, `htc_step_scores.csv`, `task_scores.csv`,
  `metrics.csv`). Regenerating these is the one step that needs the raw rollouts.
* `ours/<benchmark>_<model>/thr_0p3/train_risk_model_temp_sca/<run>/` — the
  canonical Ours/L2 cross-validation run: `summary_table.csv`,
  `cv_predictions_{all,macro_only,uq_only}.csv`, risk–coverage curves.
  Reproduce with `bash reproduce.sh --retrain`.

`main_results_table.csv` at the repository root is the tidy form of Table 1 and
Table 6: one row per method, one column per (metric, benchmark, backbone).

## Raw rollouts

Not included (~3.2 GB): the per-step action/observation logs, the ten Micro
samples per step, and the verbalized-confidence rollouts. They are only needed to

* recompute the baseline per-step scores in `reproduction_output/baseline/`, or
* re-run the upstream pipeline in `pipeline/`.

They are 39,614 JSON files: per-task action histories, the agent's reasoning
text, the ten Micro samples per step, and token logprobs where the backbone
exposes them. No screenshots or page captures are included.

**Availability.** The rollouts are archived separately as ten `.tar.gz` files,
one per (benchmark, backbone), so a single setting can be fetched on its own
(hosted at <https://huggingface.co/datasets/Shayla216/Trajectory-Leave-Traces-Rollouts>):

```
https://huggingface.co/datasets/Shayla216/Trajectory-Leave-Traces-Rollouts/resolve/main/webarena_{claude,gpt_5_2,gptnano,kimi,qwen30b}.tar.gz
https://huggingface.co/datasets/Shayla216/Trajectory-Leave-Traces-Rollouts/resolve/main/mind2web_{claude,gpt_5_2,gptnano,kimi,qwen30b}.tar.gz
https://huggingface.co/datasets/Shayla216/Trajectory-Leave-Traces-Rollouts/resolve/main/SHA256SUMS
```

Each archive unpacks to a single `raw/` directory. Place it inside the matching
model directory so the layout becomes
`data/<benchmark>/<model>/raw/{plain,verbalized}/` — every loader picks the raw
path up automatically once it exists, and falls back to the derived companions
above when it does not.

```bash
# one setting
tar -xzf webarena_qwen30b.tar.gz -C data/webarena/qwen30b/

# all ten, from a directory holding the archives
for f in {webarena,mind2web}_*.tar.gz; do
  b=${f%%_*}; m=${f#*_}; m=${m%.tar.gz}
  tar -xzf "$f" -C "data/$b/$m/"
done

# verify before unpacking
shasum -a 256 -c SHA256SUMS
```
