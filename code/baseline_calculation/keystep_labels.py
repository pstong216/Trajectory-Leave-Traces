from __future__ import annotations

import json
from pathlib import Path


# NOTE ON ANNOTATION VARIANTS (2026-07-29 fix — see README.md):
#
# mind2web has TWO independent annotation passes per model sitting side by
# side in annotate_vis/result/, and they disagree on keyStep for ~75-90% of
# tasks they have in common:
#
#   mind2web_<model>_gemini.json     -- annotations of the PLAIN raw
#                                        trajectories (mind2web/<model>/).
#                                        Used ONLY by the Ours/L2 pipeline
#                                        (codes/train_risk_model/
#                                        train_risk_model_temp_sca.py via
#                                        resolve_annotation_path()).
#   mind2web_<model>_v_gemini.json   -- annotations of the VERBALIZED raw
#                                        trajectories
#                                        (mind2web/results_<model>_verbalized/).
#                                        Confirmed 0-orphan task-id match
#                                        against that directory for all 5
#                                        models. This is the correct source
#                                        for the BASELINE pipeline (this
#                                        module), since M1 (verbalized
#                                        confidence) and friends are computed
#                                        from the verbalized re-run.
#   mind2web_qwen30b_log_gemini.json -- a third, stale annotation pass for
#                                        qwen30b only (only 174/195 entries
#                                        match the verbalized raw dir, vs
#                                        202/202 for _v). Not used anywhere.
#
# webarena has a single annotation pass per model in annotation/result/, plus
# an unrelated "_m" variant that is never read here (kept for some other,
# undocumented purpose). webarena also has STALE, DIFFERING copies of some
# webarena_<model>_gemini.json files sitting in annotate_vis/result/ (the
# mind2web annotation dir) -- e.g. webarena_claude_gemini.json differs
# between the two directories. The old default_annotation_dir() pointed at
# annotate_vis/result unconditionally, which meant any webarena caller that
# didn't pass --annotation_dir explicitly (this covers every run.sh actually
# checked into this repo) silently read either a stale/wrong file (claude) or
# nothing at all (gpt_5_2/gptnano/kimi/qwen30b, which have no stray copy in
# annotate_vis/result/, giving an EMPTY keystep_map -- i.e. no key-step
# relabeling was applied for those runs unless annotation_dir was passed by
# hand).
#
# Fix: resolve a single, explicit, benchmark-correct file per (benchmark,
# model) -- no directory-wide glob/merge, no benchmark-independent default
# directory, and raise loudly instead of silently degrading to an empty map.


def _parse_items(data: dict, prefix: str) -> dict[str, int]:
    key_map: dict[str, int] = {}
    for task_key, payload in data.get("items", {}).items():
        if not task_key.startswith(prefix):
            continue
        task_id = task_key[len(prefix):]
        key_step = payload.get("keyStep")
        if key_step is None:
            continue
        try:
            key_map[str(task_id)] = max(0, int(key_step))
        except (TypeError, ValueError):
            continue
    return key_map


def _load_paper_release_file(ann_dir: Path, bench: str, model: str, role: str) -> dict[str, int] | None:
    """
    keystep_labels/<bench>/<model>_{baseline,ours}.json --
    already pre-resolved to exactly one file per (benchmark, model, role), no
    filename-pattern guessing needed. Returns None if this convention doesn't
    apply here (caller falls back to the original resolution below), or the
    parsed map if it does -- so code/baseline_calculation is
    runnable against keystep_labels alone, with no dependency
    on the original annotation/result or annotate_vis/result directories
    that don't exist inside this repository.

    Task keys inside these files keep their ORIGINAL "<bench>::<model_variant>::
    <task_id>" form, where model_variant carries a role suffix that differs
    per (benchmark, role) -- e.g. "claude_m" (webarena ours), "claude_v"
    (mind2web baseline), plain "claude" (webarena baseline / mind2web ours).
    Confirmed 2026-07-31 (reproduction smoke test): assuming a fixed
    "<bench>::<model>::" prefix silently matched 0 items for the two
    suffixed cases, degrading every failed task to key_step=None. Since each
    file is already exclusively scoped to one (benchmark, model, role) --
    not shared/merged like the original annotation files -- there's no
    cross-contamination risk in just taking everything after the last "::"
    as task_id, with no prefix filtering needed at all.
    """
    p = ann_dir / f"{model}_{role}.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    key_map: dict[str, int] = {}
    for task_key, payload in data.get("items", {}).items():
        if "::" not in task_key:
            continue
        task_id = task_key.rsplit("::", 1)[-1]
        key_step = payload.get("keyStep")
        if key_step is None:
            continue
        try:
            key_map[str(task_id)] = max(0, int(key_step))
        except (TypeError, ValueError):
            continue
    return key_map


def load_keystep_map(source: str, annotation_dir: str | Path | None = None) -> dict[str, int]:
    """
    Load {task_id -> keyStep} for one (benchmark, model) source, for the
    BASELINE pipeline (M1-M14). See the module-level note above for why each
    benchmark resolves to exactly one file.

    Expected file format:
      {
        "items": {
          "<benchmark>::<model>::<task_id>": {"keyStep": 3, ...},
          ...
        }
      }

    Raises FileNotFoundError if the expected annotation file is missing,
    rather than silently returning an empty/partial map.
    """
    bench, model = source.split("/", 1)
    results_root = Path(__file__).resolve().parent.parent

    if annotation_dir is not None:
        pr = _load_paper_release_file(Path(annotation_dir), bench, model, "baseline")
        if pr is not None:
            return pr

    if bench == "webarena":
        ann_dir = Path(annotation_dir) if annotation_dir else results_root / "annotation" / "result"
        filename = f"webarena_{model}_gemini.json"
        prefix = f"webarena::{model}::"
    elif bench == "mind2web":
        ann_dir = Path(annotation_dir) if annotation_dir else results_root / "annotate_vis" / "result"
        filename = f"mind2web_{model}_v_gemini.json"
        prefix = f"mind2web::{model}_v::"
    else:
        raise ValueError(f"Unknown benchmark {bench!r} in source {source!r}")

    p = ann_dir / filename
    if not p.is_file():
        raise FileNotFoundError(
            f"No key-step annotation file for source={source!r}: expected {p}. "
            f"(Pass annotation_dir explicitly to override.)"
        )

    data = json.loads(p.read_text())
    return _parse_items(data, prefix)


def load_keystep_map_entropy_source(source: str, annotation_dir: str | Path | None = None) -> dict[str, int]:
    """
    Load {task_id -> keyStep} matching the raw source that M5/M6 (action
    entropy) actually read from -- entropy_dir, the PLAIN multi-sample
    rollout -- NOT the file load_keystep_map() resolves (that one matches
    the VERBALIZED rollout that M1-M4/M7/M11-M14 read from).

    2026-07-30 fix (flagged): compute_baselines.py etc. previously
    loaded ONE keystep_map (always the verbalized-matching file) and applied
    it to every row, including M5/M6 rows whose score came from entropy_dir.
    That mislabeled M5/M6 against an annotation of a trajectory their score
    was never computed from. The correct label source for entropy_dir is the
    file annotated against IT specifically -- which is the same file the
    Ours/L2 pipeline uses (resolve_annotation_path() in
    train_risk_model_temp_sca.py), since Ours' features are also built from
    entropy_dir/the plain rollout:
      mind2web  -> <model>_gemini.json     (no suffix)
      webarena  -> <model>_m_gemini.json
    """
    bench, model = source.split("/", 1)
    results_root = Path(__file__).resolve().parent.parent

    if annotation_dir is not None:
        pr = _load_paper_release_file(Path(annotation_dir), bench, model, "ours")
        if pr is not None:
            return pr

    if bench == "webarena":
        ann_dir = Path(annotation_dir) if annotation_dir else results_root / "annotation" / "result"
        filename = f"webarena_{model}_m_gemini.json"
        prefix = f"webarena::{model}_m::"
    elif bench == "mind2web":
        ann_dir = Path(annotation_dir) if annotation_dir else results_root / "annotate_vis" / "result"
        filename = f"mind2web_{model}_gemini.json"
        prefix = f"mind2web::{model}::"
    else:
        raise ValueError(f"Unknown benchmark {bench!r} in source {source!r}")

    p = ann_dir / filename
    if not p.is_file():
        raise FileNotFoundError(
            f"No key-step annotation file for entropy source={source!r}: expected {p}. "
            f"(Pass annotation_dir explicitly to override.)"
        )

    data = json.loads(p.read_text())
    return _parse_items(data, prefix)


def label_step_from_keystep(final_success: bool | None, step: int, key_step: int | None) -> int | None:
    """
    Return a binary step label where 1=success and 0=failure.

    For failed tasks with key-step annotation:
      step < key_step  -> success
      step >= key_step -> failure
    """
    if final_success is None:
        return None
    if final_success:
        return 1
    if key_step is None:
        return 0
    return 1 if step < int(key_step) else 0


def label_prefix_failure_from_keystep(final_success: bool | None, prefix_len: int, key_step: int | None) -> int | None:
    """
    Return prefix outcome for HTC Pre@k with 1=failure, 0=success.

    Prefix length k contains steps [0, ..., k-1].
    A failed task remains successful until the prefix includes key_step.
    """
    if final_success is None:
        return None
    if final_success:
        return 0
    if key_step is None:
        return 1
    return 1 if prefix_len > int(key_step) else 0


def as_success_text(label: int | None) -> str | None:
    if label is None:
        return None
    return "success" if int(label) == 1 else "failure"
