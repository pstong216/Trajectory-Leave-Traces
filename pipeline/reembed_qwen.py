#!/usr/bin/env python3
"""
Re-embed intention (S) and state-change (C) texts using Qwen/Qwen3-Embedding-0.6B
on GPU, then recompute HLVM UQ metrics per step.

Replaces all-MiniLM-L6-v2 with Qwen3-Embedding-0.6B (last-token pooling, decoder arch).
Output is written to {source}/uq_analysis_change_qwen/ (parallel to uq_analysis_change/).

Supports two data formats detected automatically:

  WebArena  — trajectory_1/uq_samples/{task_id}/step_{k}_samples.json
              each file: {"task_id":int, "step":k, "samples":[{"raw_prediction":str,
                          "parsed_action":str}, ...]}
              outcomes:  trajectory_1/actions/{task_id}.json  ("score" field)

  Mind2Web  — {task_index}/result.json
              each file: {"task_index":int, "step_samples":[[str,...], ...],
                          "action_history":[str,...]}
              outcomes:  ../qwen_result.json  ("tasks":[{"task_id":str,"status":str}])

Usage:
  python reembed_qwen.py --source webarena/claude
  python reembed_qwen.py --source webarena/gpt_5_2
  python reembed_qwen.py --source webarena/qwen30b
  python reembed_qwen.py --source mind2web/qwen30b
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

# ── Model config ──────────────────────────────────────────────────────────────

QWEN_MODEL_NAME       = "Qwen/Qwen3-Embedding-0.6B"
CLUSTER_DIST_THRESHOLD = 0.3
EMBED_BATCH_SIZE       = 64   # reduce if OOM

# ── Device ────────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"[device] Using: {DEVICE}", flush=True)

# ── Model loading (singleton) ──────────────────────────────────────────────────

_TOKENIZER = None
_MODEL     = None


def get_model():
    global _TOKENIZER, _MODEL
    if _MODEL is None:
        print(f"  [model] Loading {QWEN_MODEL_NAME} ...", flush=True)
        _TOKENIZER = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
        dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
        _MODEL = AutoModel.from_pretrained(
            QWEN_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(DEVICE)
        _MODEL.eval()
        print(f"  [model] Loaded on {DEVICE}.", flush=True)
    return _TOKENIZER, _MODEL


# ── Embedding ─────────────────────────────────────────────────────────────────

def _last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Last non-padding token pooling (Qwen3-Embedding convention)."""
    seq_lens = mask.sum(dim=1) - 1          # index of last real token
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, seq_lens]      # (B, D)


def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts with Qwen3-Embedding-0.6B.
    Returns L2-normalised float32 embeddings of shape (N, D).
    """
    tokenizer, model = get_model()
    all_embs = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            out = model(**enc)

        emb = _last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu().float().numpy())

    return np.vstack(all_embs)


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_texts(texts: List[str]) -> Tuple[List[int], np.ndarray]:
    embeddings = encode_texts(texts)
    if len(texts) == 1:
        return [0], embeddings
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=CLUSTER_DIST_THRESHOLD,
    )
    labels = clustering.fit_predict(embeddings).tolist()
    return labels, embeddings


# ── Regex / parsing (identical to compute_hlvm_uq_multivariable.py) ──────────

IN_ORDER_RE = re.compile(
    r'[Ii]n order to\s+(.*?)(?:,\s*I will|,\s*I\'ll|\.\s*I will|\.\s*I\'ll|,\s*I am|\n)',
    re.DOTALL,
)
WOULD_RE = re.compile(
    r'\bwould\s+(.*?)(?:,\s*which\s+(?:is|will|would)|\s*->)',
    re.DOTALL,
)
BY_ACTION_RE = re.compile(
    r'\bBy\s+\w+ing\b[^,]*,\s*(.*?)(?:,\s*which\s+(?:is|will|would)|\s*->)',
    re.DOTALL,
)
VALUE_MAP = {"very_harmful": -2, "harmful": -1, "neutral": 0, "helpful": 1, "very_helpful": 2}
VALUE_RE       = re.compile(r'->\s*(very_harmful|harmful|neutral|helpful|very_helpful)', re.IGNORECASE)
ACTION_ELEM_RE = re.compile(r'\[(\S+?)\]')


def parse_sample(sample: Dict):
    raw    = sample.get("raw_prediction", "")
    action = sample.get("parsed_action", "")

    m = IN_ORDER_RE.search(raw)
    intention = m.group(1).strip() if m else None

    m = WOULD_RE.search(raw)
    change = m.group(1).strip() if m else None
    if change is None:
        m = BY_ACTION_RE.search(raw)
        change = m.group(1).strip() if m else None

    m = VALUE_RE.search(raw)
    value = VALUE_MAP.get(m.group(1).lower()) if m else None

    parts = action.split()
    if parts:
        atype = parts[0].lower()
        m = ACTION_ELEM_RE.search(action)
        target: Optional[str] = (
            f"{atype} [{m.group(1).strip()}]"
            if m
            else (atype if atype in ("scroll", "stop", "goto", "press", "type") else None)
        )
    else:
        target = None

    return intention, change, value, target


# ── Entropy helper ────────────────────────────────────────────────────────────

def _h(probs) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 1e-12))


# ── Per-step HLVM computation ─────────────────────────────────────────────────

def compute_step_hlvm(step_data: Dict) -> Tuple[Dict, Dict, Dict]:
    samples = step_data["samples"]
    N = len(samples)

    parsed = [parse_sample(s) for s in samples]
    valid = [
        (intent, change, val, tgt)
        for intent, change, val, tgt in parsed
        if intent is not None and change is not None and tgt is not None
    ]
    M = len(valid)

    step_meta = {
        "task_id":      step_data["task_id"],
        "step":         step_data["step"],
        "page_url":     step_data.get("page_url", ""),
        "action_taken": step_data.get("action_taken", ""),
        "num_samples":  N,
        "num_valid":    M,
    }

    empty_ci = {"raw_texts": [], "cluster_ids": [], "cluster_to_labels": {}, "embeddings_shape": None}

    if M == 0:
        return (
            {**step_meta, "degenerate": True,
             "U_sem": 0.0, "U_change": 0.0, "U_ground": 0.0, "U_cval": 0.0,
             "H_T": 0.0, "C_S": 0.0, "C_C": 0.0, "R_T": 0.0,
             "identity_error": 0.0,
             "K_S": 0, "K_C": 0, "num_targets": 0,
             "P_SCT": {}, "P_S": {}, "P_C": {}, "P_T": {}},
            empty_ci, empty_ci,
        )

    # Cluster S (intentions)
    s_texts = [x[0] for x in valid]
    s_ids, s_emb = cluster_texts(s_texts)
    K_S = len(set(s_ids))
    s_c2l: Dict = {}
    for cid, txt in zip(s_ids, s_texts):
        s_c2l.setdefault(str(cid), []).append(txt)
    intent_ci = {"raw_texts": s_texts, "cluster_ids": s_ids,
                 "cluster_to_labels": s_c2l, "embeddings_shape": list(s_emb.shape)}

    # Cluster C (state-changes)
    c_texts = [x[1] for x in valid]
    c_ids, c_emb = cluster_texts(c_texts)
    K_C = len(set(c_ids))
    c_c2l: Dict = {}
    for cid, txt in zip(c_ids, c_texts):
        c_c2l.setdefault(str(cid), []).append(txt)
    change_ci = {"raw_texts": c_texts, "cluster_ids": c_ids,
                 "cluster_to_labels": c_c2l, "embeddings_shape": list(c_emb.shape)}

    # Build joint P(s,c,t|O)
    triples = [
        (s_id, c_id, tgt)
        for (_, _, _, tgt), s_id, c_id in zip(valid, s_ids, c_ids)
    ]
    joint_counts: Dict = defaultdict(int)
    for sct in triples:
        joint_counts[sct] += 1
    P_SCT = {k: v / M for k, v in joint_counts.items()}

    P_S: Dict = defaultdict(float)
    P_C: Dict = defaultdict(float)
    P_T: Dict = defaultdict(float)
    for (s, c, t), p in P_SCT.items():
        P_S[s] += p; P_C[c] += p; P_T[t] += p

    unique_S = sorted(P_S.keys())
    unique_C = sorted(P_C.keys())
    unique_T = sorted(P_T.keys())

    U_sem    = _h(list(P_S.values()))
    U_change = _h(list(P_C.values()))

    P_ST: Dict = defaultdict(float)
    for (s, c, t), p in P_SCT.items():
        P_ST[(s, t)] += p

    U_ground = 0.0
    for s in unique_S:
        p_t_s = {t: P_ST.get((s, t), 0.0) / P_S[s] for t in unique_T}
        U_ground += P_S[s] * _h([v for v in p_t_s.values() if v > 0])

    U_cval = 0.0
    for (s, t), p_st in P_ST.items():
        if p_st < 1e-12:
            continue
        p_c_st = {c: P_SCT.get((s, c, t), 0.0) / p_st for c in unique_C}
        U_cval += p_st * _h([v for v in p_c_st.values() if v > 0])

    H_T = _h(list(P_T.values()))

    P_SC: Dict = defaultdict(float)
    for (s, c, t), p in P_SCT.items():
        P_SC[(s, c)] += p

    H_T_given_SC = 0.0
    for (s, c), p_sc in P_SC.items():
        if p_sc < 1e-12:
            continue
        p_t_sc = {t: P_SCT.get((s, c, t), 0.0) / p_sc for t in unique_T}
        H_T_given_SC += p_sc * _h([v for v in p_t_sc.values() if v > 0])

    C_S = H_T - U_ground
    C_C = U_ground - H_T_given_SC
    R_T = H_T_given_SC
    identity_error = abs((C_S + C_C + R_T) - H_T)

    metrics = {
        **step_meta,
        "degenerate":     False,
        "K_S":            K_S,
        "K_C":            K_C,
        "num_targets":    len(unique_T),
        "U_sem":          U_sem,
        "U_change":       U_change,
        "U_ground":       U_ground,
        "U_cval":         U_cval,
        "H_T":            H_T,
        "C_S":            C_S,
        "C_C":            C_C,
        "R_T":            R_T,
        "identity_error": identity_error,
        "P_SCT": {str(k): v for k, v in P_SCT.items()},
        "P_S":   {str(k): v for k, v in P_S.items()},
        "P_C":   {str(k): v for k, v in P_C.items()},
        "P_T":   dict(P_T),
        "P_SC":  {str(k): v for k, v in P_SC.items()},
        "P_ST":  {str(k): v for k, v in P_ST.items()},
    }
    return metrics, intent_ci, change_ci


# ── Task aggregation ──────────────────────────────────────────────────────────

SCALAR_FEATURES = [
    "U_sem", "U_change", "U_ground", "U_cval",
    "H_T", "C_S", "C_C", "R_T",
    "K_S", "K_C", "num_targets",
]


def aggregate_task(task_id: int, steps: List[Dict]) -> Dict:
    valid = [s for s in steps if not s.get("degenerate", False)]
    n    = len(valid)
    agg  = {"task_id": task_id, "num_steps": len(steps), "num_valid_steps": n}
    for feat in SCALAR_FEATURES:
        vals = [s[feat] for s in valid if feat in s]
        if vals:
            m  = float(np.mean(vals))
            mx = float(np.max(vals))
            agg[f"{feat}_mean"]     = m
            agg[f"{feat}_max"]      = mx
            agg[f"{feat}_mean_max"] = (m + mx) / 2.0
        else:
            agg[f"{feat}_mean"] = agg[f"{feat}_max"] = agg[f"{feat}_mean_max"] = 0.0
    return agg


# ── Format detection ──────────────────────────────────────────────────────────

def is_webarena(base_dir: Path) -> bool:
    """WebArena has trajectory_1/uq_samples/; Mind2Web does not."""
    return (base_dir / "trajectory_1" / "uq_samples").exists()


# ── Outcomes ──────────────────────────────────────────────────────────────────

def load_outcomes_webarena(base_dir: Path) -> Dict[int, float]:
    """WebArena: score from trajectory_1/actions/{task_id}.json."""
    actions_dir = base_dir / "trajectory_1" / "actions"
    outcomes: Dict[int, float] = {}
    if not actions_dir.exists():
        return outcomes
    for p in actions_dir.glob("*.json"):
        try:
            tid = int(p.stem)
            d   = json.loads(p.read_text())
            s   = d.get("score")
            if s is not None:
                outcomes[tid] = 1.0 if float(s) >= 1.0 else 0.0
        except (ValueError, KeyError):
            pass
    return outcomes


def load_outcomes_mind2web(base_dir: Path) -> Dict[int, float]:
    """Mind2Web: status from ../qwen_result.json (one level above source dir)."""
    result_file = base_dir.parent / "qwen_result.json"
    outcomes: Dict[int, float] = {}
    if not result_file.exists():
        print(f"  [WARN] outcomes file not found: {result_file}", flush=True)
        return outcomes
    d = json.loads(result_file.read_text())
    for task in d.get("tasks", []):
        try:
            tid    = int(task["task_id"])
            status = task.get("status", "failure")
            outcomes[tid] = 1.0 if status == "success" else 0.0
        except (ValueError, KeyError):
            pass
    return outcomes


# ── Mind2Web sample conversion ────────────────────────────────────────────────

_M2W_ACTION_RE = re.compile(r'```([^`]+)```')


def _extract_parsed_action(raw: str) -> str:
    """Extract the action from a Mind2Web raw prediction string.
    The format ends with: ...the next action I will perform is ```action [id]```
    """
    m = _M2W_ACTION_RE.search(raw)
    return m.group(1).strip() if m else ""


def mind2web_result_to_step_data(task_index: int, step_idx: int,
                                  raw_samples: List[str],
                                  action_taken: str) -> Dict:
    """Convert Mind2Web result.json step entry into the same format as
    WebArena's step_k_samples.json so compute_step_hlvm() works unchanged."""
    samples = [
        {
            "raw_prediction": raw,
            "parsed_action":  _extract_parsed_action(raw),
        }
        for raw in raw_samples
    ]
    return {
        "task_id":      task_index,
        "step":         step_idx,
        "page_url":     "",
        "action_taken": action_taken,
        "samples":      samples,
    }


# ── AUROC helper ──────────────────────────────────────────────────────────────

SUFFIXES = ["mean", "max", "mean_max"]


def compute_auroc(summaries: List[Dict], outcomes: Dict[int, float]) -> Dict:
    labeled = [t for t in summaries if t["task_id"] in outcomes]
    failure = np.array([1.0 - outcomes[t["task_id"]] for t in labeled])
    results = {}
    for feat in SCALAR_FEATURES:
        for suf in SUFFIXES:
            col    = f"{feat}_{suf}"
            scores = np.array([t.get(col, 0.0) for t in labeled], dtype=float)
            if len(np.unique(scores)) < 2:
                results[col] = {"auroc": None}
                continue
            results[col] = {"auroc": round(float(roc_auc_score(failure, scores)), 4)}
    return results


# ── Shared per-task processing ────────────────────────────────────────────────

def process_task(task_id: int, step_data_list: List[Dict],
                 out_task: Path, outcomes: Dict[int, float],
                 force: bool, all_summaries: list):
    """Embed, cluster, compute HLVM, and write outputs for one task.
    step_data_list: list of dicts in WebArena step_samples.json format."""
    existing = out_task / "task_hlvm_uq.json"
    n_steps  = len(step_data_list)

    if not force and existing.exists():
        n_done = len(list(out_task.glob("step_*_hlvm_uq.json")))
        if n_done == n_steps:
            summary = json.loads(existing.read_text())
            summary["outcome"] = outcomes.get(task_id)
            existing.write_text(json.dumps(summary, indent=2))
            all_summaries.append(summary)
            print(f"=== Task {task_id} [skipped — {n_done} steps already done] ===", flush=True)
            return
        else:
            print(f"=== Task {task_id} [RE-PROCESSING — {n_done}/{n_steps} steps] ===", flush=True)
            for stale in out_task.glob("step_*_hlvm_uq.json"):
                stale.unlink()

    print(f"\n=== Task {task_id} ({n_steps} steps) ===", flush=True)
    out_task.mkdir(parents=True, exist_ok=True)

    step_results   = []
    all_s_clusters = {}
    all_c_clusters = {}

    for step_data in step_data_list:
        step_idx = step_data["step"]
        print(f"  step {step_idx}", end=" ", flush=True)

        metrics, s_ci, c_ci = compute_step_hlvm(step_data)
        step_results.append(metrics)
        all_s_clusters[f"step_{step_idx}"] = s_ci
        all_c_clusters[f"step_{step_idx}"] = c_ci

        (out_task / f"step_{step_idx}_hlvm_uq.json").write_text(json.dumps(metrics, indent=2))
        print(f"K_S={metrics.get('K_S',0)} K_C={metrics.get('K_C',0)} "
              f"U_sem={metrics.get('U_sem',0):.3f}", flush=True)

    (out_task / "intent_clusters.json").write_text(json.dumps(all_s_clusters, indent=2))
    (out_task / "change_clusters.json").write_text(json.dumps(all_c_clusters, indent=2))

    summary = aggregate_task(task_id, step_results)
    summary["outcome"] = outcomes.get(task_id)
    (out_task / "task_hlvm_uq.json").write_text(json.dumps(summary, indent=2))
    all_summaries.append(summary)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", required=True,
        help="Path relative to results_total/, e.g. webarena/claude or mind2web/qwen30b",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-process even if output already exists",
    )
    args = ap.parse_args()

    base_dir = Path(__file__).parent.parent.parent / args.source
    if not base_dir.exists():
        raise FileNotFoundError(f"Source not found: {base_dir}")

    out_root = base_dir / "uq_analysis_change_qwen"
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Detect format and load tasks ──────────────────────────────────────────
    if is_webarena(base_dir):
        fmt            = "WebArena"
        outcomes       = load_outcomes_webarena(base_dir)
        uq_samples_dir = base_dir / "trajectory_1" / "uq_samples"

        # Build list of (task_id, [step_data, ...]) from per-step JSON files
        task_entries: List[Tuple[int, List[Dict]]] = []
        for task_dir in sorted(uq_samples_dir.iterdir(), key=lambda p: int(p.name)):
            task_id    = int(task_dir.name)
            step_files = sorted(
                task_dir.glob("step_*_samples.json"),
                key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)),
            )
            step_data_list = []
            for sf in step_files:
                sd = json.loads(sf.read_text())
                if len(sd.get("samples", [])) < 10:
                    print(f"  [WARN] task {task_id} {sf.name}: "
                          f"{len(sd.get('samples',[]))} samples — skipping step", flush=True)
                    continue
                step_data_list.append(sd)
            if step_data_list:
                task_entries.append((task_id, step_data_list))

    else:
        fmt      = "Mind2Web"
        outcomes = load_outcomes_mind2web(base_dir)

        # Build list of (task_id, [step_data, ...]) from per-task result.json files
        task_entries = []
        for task_dir in sorted(base_dir.iterdir(), key=lambda p: int(p.name)
                                if p.name.isdigit() else -1):
            if not task_dir.is_dir() or not task_dir.name.isdigit():
                continue
            result_file = task_dir / "result.json"
            if not result_file.exists():
                continue
            d          = json.loads(result_file.read_text())
            task_index = int(d["task_index"])
            step_samps = d.get("step_samples", [])   # list of lists of strings
            actions    = d.get("action_history", [])

            step_data_list = []
            for k, raw_list in enumerate(step_samps):
                if len(raw_list) < 10:
                    print(f"  [WARN] task {task_index} step {k}: "
                          f"{len(raw_list)} samples — skipping step", flush=True)
                    continue
                action_taken = actions[k] if k < len(actions) else ""
                step_data_list.append(
                    mind2web_result_to_step_data(task_index, k, raw_list, action_taken)
                )
            if step_data_list:
                task_entries.append((task_index, step_data_list))

    print(f"Format     : {fmt}")
    print(f"Source     : {base_dir}")
    print(f"Output     : {out_root}")
    print(f"Tasks      : {len(task_entries)}")
    print(f"Outcomes   : {len(outcomes)} tasks")
    print(f"Embed model: {QWEN_MODEL_NAME}")
    print(f"Device     : {DEVICE}\n", flush=True)

    all_summaries: list = []
    for task_id, step_data_list in task_entries:
        process_task(
            task_id      = task_id,
            step_data_list = step_data_list,
            out_task     = out_root / str(task_id),
            outcomes     = outcomes,
            force        = args.force,
            all_summaries = all_summaries,
        )

    (out_root / "all_tasks_hlvm_uq.json").write_text(json.dumps(all_summaries, indent=2))

    if outcomes:
        auroc_results = compute_auroc(all_summaries, outcomes)
        auroc_dir = out_root / "auroc"
        auroc_dir.mkdir(exist_ok=True)
        (auroc_dir / "auroc_report.json").write_text(json.dumps(auroc_results, indent=2))
        print("\nAUROC summary (mean aggregation):")
        for feat in SCALAR_FEATURES:
            col = f"{feat}_mean"
            a   = auroc_results.get(col, {}).get("auroc")
            if a is not None:
                print(f"  {col:<28}  {a:.4f}")

    print(f"\nDone. Results in {out_root}")


if __name__ == "__main__":
    main()
