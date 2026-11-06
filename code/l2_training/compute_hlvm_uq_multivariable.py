#!/usr/bin/env python3
"""
HLVM UQ Analysis — Multi-variable variant for results_gpt / results_claude.

Treats state-change (C), element/action-target (T), and intention (S) as
latent variables and computes entropy-based UQ metrics.

Variables
---------
  S  – Intention text     ("In order to ...")       → clustered
  C  – State-change text  ("By …, X would happen")  → clustered
  T  – Target element     (e.g. "[24]", "scroll")   → symbolic

Metrics computed per step
--------------------------
  U_sem      = H(S|O)              semantic / intention entropy
  U_change   = H(C|O)              state-change entropy
  U_ground   = H(T|S,O)            grounding entropy
  U_cval     = H(C|T,S,O)          change uncertainty given action
  H_T        = H(T|O)              total action entropy
  C_S        = H(T|O)  - H(T|S,O)      intention contribution
  C_C        = H(T|S,O)- H(T|S,C,O)   change contribution
  R_T        = H(T|S,C,O)              residual

Usage
-----
  python compute_hlvm_uq_multivariable.py --source results_gpt
  python compute_hlvm_uq_multivariable.py --source results_claude

Outputs (written inside the source directory)
---------------------------------------------
  {source}/uq_analysis_change/{task_id}/step_{k}_hlvm_uq.json
  {source}/uq_analysis_change/{task_id}/intent_clusters.json      (S clusters)
  {source}/uq_analysis_change/{task_id}/change_clusters.json      (C clusters)
  {source}/uq_analysis_change/{task_id}/task_hlvm_uq.json
  {source}/uq_analysis_change/all_tasks_hlvm_uq.json
  {source}/uq_analysis_change/auroc_ece_report.json
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score

# ── Ground-truth outcomes ─────────────────────────────────────────────────────

def load_outcomes(base_dir: Path) -> Dict[int, float]:
    """Load per-task outcomes from actions/{task_id}.json, binarized to {0.0, 1.0}."""
    actions_dir = base_dir / "trajectory_1" / "actions"
    outcomes: Dict[int, float] = {}
    if not actions_dir.exists():
        return outcomes
    for p in actions_dir.glob("*.json"):
        try:
            tid = int(p.stem)
            with open(p) as f:
                d = json.load(f)
            score = d.get("score", None)
            if score is not None:
                outcomes[tid] = 1.0 if float(score) >= 1.0 else 0.0
        except (ValueError, KeyError):
            pass
    return outcomes

VALUE_MAP = {
    "very_harmful": -2, "harmful": -1, "neutral": 0,
    "helpful": 1, "very_helpful": 2,
}
VALUE_LEVELS = [-2, -1, 0, 1, 2]

# ── Regex patterns ─────────────────────────────────────────────────────────────

# Intention (S): text after "In order to" up to ", I will" / ", I'll" etc.
IN_ORDER_RE = re.compile(
    r'[Ii]n order to\s+(.*?)(?:,\s*I will|,\s*I\'ll|\.\s*I will|\.\s*I\'ll|,\s*I am|\n)',
    re.DOTALL
)
# State change (C): text between "By …," and ", which is/will" or " ->"
WOULD_RE = re.compile(
    r'\bwould\s+(.*?)(?:,\s*which\s+(?:is|will|would)|\s*->)',
    re.DOTALL
)
# Also try "By clicking/doing X, Y becomes ..." pattern
BY_ACTION_RE = re.compile(
    r'\bBy\s+\w+ing\b[^,]*,\s*(.*?)(?:,\s*which\s+(?:is|will|would)|\s*->)',
    re.DOTALL
)
# Value keyword label
VALUE_RE = re.compile(
    r'->\s*(very_harmful|harmful|neutral|helpful|very_helpful)',
    re.IGNORECASE
)
# Target element
ACTION_ELEM_RE = re.compile(r'\[(\S+?)\]')

CLUSTER_DISTANCE_THRESHOLD = 0.3

# ── Embedding model ───────────────────────────────────────────────────────────

_MODEL: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print("  [model] Loading all-MiniLM-L6-v2 …", flush=True)
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [model] Loaded.", flush=True)
    return _MODEL


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_sample(sample: Dict) -> Tuple[
    Optional[str],   # S: intention text
    Optional[str],   # C: state-change text
    Optional[int],   # V: value score (kept for cross-comparison)
    Optional[str],   # T: target element
]:
    raw    = sample.get("raw_prediction", "")
    action = sample.get("parsed_action", "")

    # Intention (S)
    m = IN_ORDER_RE.search(raw)
    intention = m.group(1).strip() if m else None

    # State change (C): try "would …" pattern first, then "By …, X …" pattern
    m = WOULD_RE.search(raw)
    if m:
        change = m.group(1).strip()
    else:
        m = BY_ACTION_RE.search(raw)
        change = m.group(1).strip() if m else None

    # Value keyword (V)
    m = VALUE_RE.search(raw)
    value = VALUE_MAP.get(m.group(1).lower()) if m else None

    # Target (T) = action_type + element, e.g. "click [24]", "type [42]", "scroll"
    # For "type [id] <text>", we drop the text content to avoid spurious uniqueness.
    parts = action.split()
    if parts:
        action_type = parts[0].lower()
        m = ACTION_ELEM_RE.search(action)
        if m:
            target: Optional[str] = f"{action_type} [{m.group(1).strip()}]"
        else:
            # no element bracket — use just the action type (e.g. "stop", "goto")
            target = action_type if action_type in ("scroll", "stop", "goto", "press", "type") else None
    else:
        target = None

    return intention, change, value, target


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_texts(texts: List[str]) -> Tuple[List[int], np.ndarray]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    if len(texts) == 1:
        return [0], embeddings
    clustering = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
    )
    return clustering.fit_predict(embeddings).tolist(), embeddings


# ── Entropy helpers ───────────────────────────────────────────────────────────

def _h(probs) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 1e-12))


# ── Per-step computation ──────────────────────────────────────────────────────

def compute_step_hlvm(step_data: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Returns: metrics, intent_cluster_info (S), change_cluster_info (C)
    """
    samples = step_data["samples"]
    N = len(samples)

    parsed = [parse_sample(s) for s in samples]
    # Keep samples where S, C, T are all present (V optional for cross-compare)
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

    empty_ci = {
        "raw_texts": [], "cluster_ids": [],
        "cluster_to_labels": {}, "embeddings_shape": None,
    }

    if M == 0:
        return (
            {**step_meta, "degenerate": True,
             "U_sem": 0.0, "U_change": 0.0, "U_ground": 0.0, "U_cval": 0.0,
             "H_T": 0.0, "C_S": 0.0, "C_C": 0.0, "R_T": 0.0,
             "identity_error": 0.0,
             "K_S": 0, "K_C": 0, "num_targets": 0,
             "P_SCT": {}, "P_S": {}, "P_C": {}, "P_T": {}},
            empty_ci, empty_ci
        )

    # ── Cluster intentions (S) ────────────────────────────────────────────────
    s_texts  = [x[0] for x in valid]
    s_ids, s_emb = cluster_texts(s_texts)
    K_S = len(set(s_ids))

    s_c2l: Dict[str, List[str]] = {}
    for cid, txt in zip(s_ids, s_texts):
        s_c2l.setdefault(str(cid), []).append(txt)

    intent_ci = {
        "raw_texts": s_texts,
        "cluster_ids": s_ids,
        "cluster_to_labels": s_c2l,
        "embeddings_shape": list(s_emb.shape),
    }

    # ── Cluster state changes (C) ─────────────────────────────────────────────
    c_texts  = [x[1] for x in valid]
    c_ids, c_emb = cluster_texts(c_texts)
    K_C = len(set(c_ids))

    c_c2l: Dict[str, List[str]] = {}
    for cid, txt in zip(c_ids, c_texts):
        c_c2l.setdefault(str(cid), []).append(txt)

    change_ci = {
        "raw_texts": c_texts,
        "cluster_ids": c_ids,
        "cluster_to_labels": c_c2l,
        "embeddings_shape": list(c_emb.shape),
    }

    # ── Build empirical joint P̂(s, c, t | O) ─────────────────────────────────
    triples: List[Tuple[int, int, str]] = [
        (s_id, c_id, tgt)
        for (_, _, _, tgt), s_id, c_id in zip(valid, s_ids, c_ids)
    ]

    joint_counts: Dict[Tuple[int, int, str], int] = defaultdict(int)
    for sct in triples:
        joint_counts[sct] += 1

    P_SCT: Dict[Tuple[int, int, str], float] = {k: v / M for k, v in joint_counts.items()}

    # Marginals
    P_S: Dict[int, float] = defaultdict(float)
    P_C: Dict[int, float] = defaultdict(float)
    P_T: Dict[str, float] = defaultdict(float)

    for (s, c, t), p in P_SCT.items():
        P_S[s] += p
        P_C[c] += p
        P_T[t] += p

    unique_S = sorted(P_S.keys())
    unique_C = sorted(P_C.keys())
    unique_T = sorted(P_T.keys())

    # ── U_sem = H(S|O) ────────────────────────────────────────────────────────
    U_sem = _h(list(P_S.values()))

    # ── U_change = H(C|O) ─────────────────────────────────────────────────────
    U_change = _h(list(P_C.values()))

    # ── U_ground = H(T|S,O) ───────────────────────────────────────────────────
    P_ST: Dict[Tuple[int, str], float] = defaultdict(float)
    for (s, c, t), p in P_SCT.items():
        P_ST[(s, t)] += p

    U_ground = 0.0
    for s in unique_S:
        p_t_s = {t: P_ST.get((s, t), 0.0) / P_S[s] for t in unique_T}
        U_ground += P_S[s] * _h([v for v in p_t_s.values() if v > 0])

    # ── U_cval = H(C|T,S,O) ───────────────────────────────────────────────────
    U_cval = 0.0
    for (s, t), p_st in P_ST.items():
        if p_st < 1e-12:
            continue
        p_c_st = {
            c: P_SCT.get((s, c, t), 0.0) / p_st
            for c in unique_C
        }
        U_cval += p_st * _h([v for v in p_c_st.values() if v > 0])

    # ── H_T = H(T|O) ─────────────────────────────────────────────────────────
    H_T = _h(list(P_T.values()))

    # ── H(T|S,C,O) for residual ───────────────────────────────────────────────
    P_SC: Dict[Tuple[int, int], float] = defaultdict(float)
    for (s, c, t), p in P_SCT.items():
        P_SC[(s, c)] += p

    H_T_given_SC = 0.0
    for (s, c), p_sc in P_SC.items():
        if p_sc < 1e-12:
            continue
        p_t_sc = {t: P_SCT.get((s, c, t), 0.0) / p_sc for t in unique_T}
        H_T_given_SC += p_sc * _h([v for v in p_t_sc.values() if v > 0])

    # ── Decomposition: H_T = C_S + C_C + R_T ─────────────────────────────────
    C_S = H_T - U_ground          # intention contribution
    C_C = U_ground - H_T_given_SC  # state-change contribution
    R_T = H_T_given_SC             # residual
    identity_error = abs((C_S + C_C + R_T) - H_T)

    metrics = {
        **step_meta,
        "degenerate":     False,
        "K_S":            K_S,
        "K_C":            K_C,
        "num_targets":    len(unique_T),
        # Uncertainty terms
        "U_sem":          U_sem,
        "U_change":       U_change,
        "U_ground":       U_ground,
        "U_cval":         U_cval,
        # Action uncertainty & decomposition
        "H_T":            H_T,
        "C_S":            C_S,
        "C_C":            C_C,
        "R_T":            R_T,
        "identity_error": identity_error,
        # Distributions
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
    n = len(valid)
    agg = {"task_id": task_id, "num_steps": len(steps), "num_valid_steps": n}
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


# ── AUROC + ECE ───────────────────────────────────────────────────────────────

N_BINS  = 10
NEGATE  = set()   # no indicators need negation after removing value stats
SUFFIXES = ["mean", "max", "mean_max"]


def ece(scores: np.ndarray, failure: np.ndarray) -> float:
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return float("nan")
    norm = (scores - mn) / (mx - mn)
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    out  = 0.0
    N    = len(scores)
    for i in range(N_BINS):
        lo, hi = bins[i], bins[i + 1]
        mask = (norm >= lo) & (norm <= hi) if i == N_BINS - 1 else (norm >= lo) & (norm < hi)
        if mask.sum() == 0:
            continue
        out += (mask.sum() / N) * abs(failure[mask].mean() - norm[mask].mean())
    return float(out)


N_BOOT = 1000


def bootstrap_auroc_var(scores: np.ndarray, failure: np.ndarray, n_boot: int = N_BOOT, seed: int = 42) -> float:
    """Bootstrap variance of AUROC: resample tasks with replacement, compute AUROC each time."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s_b, y_b = scores[idx], failure[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, s_b))
    if len(boot_aucs) < 2:
        return float("nan")
    return float(np.var(boot_aucs))


def compute_eval(summaries: List[Dict], outcomes: Dict[int, float]) -> Dict:
    labeled = [t for t in summaries if t["task_id"] in outcomes]
    failure = np.array([1.0 - outcomes[t["task_id"]] for t in labeled])
    results = {}
    for feat in SCALAR_FEATURES:
        for suf in SUFFIXES:
            col    = f"{feat}_{suf}"
            raw    = np.array([t.get(col, 0.0) for t in labeled], dtype=float)
            scores = -raw if feat in NEGATE else raw
            if len(np.unique(scores)) < 2:
                results[col] = {"auroc": None, "auroc_var": None, "ece": None}
                continue
            a   = float(roc_auc_score(failure, scores))
            var = bootstrap_auroc_var(scores, failure)
            e   = ece(scores, failure)
            results[col] = {
                "auroc":     round(a, 4),
                "auroc_var": None if np.isnan(var) else round(var, 6),
                "ece":       None if np.isnan(e) else round(e, 4),
            }
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HLVM UQ analysis with S (intention), C (state-change), T (element) latent vars"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source folder name or path (e.g. results_gpt or results_claude)",
    )
    args = parser.parse_args()

    base_dir = Path(args.source)
    if not base_dir.is_absolute():
        base_dir = Path(__file__).parent / base_dir

    uq_samples_dir = base_dir / "trajectory_1" / "uq_samples"
    out_root       = base_dir / "uq_analysis_change"

    if not uq_samples_dir.exists():
        raise FileNotFoundError(f"uq_samples not found at: {uq_samples_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    outcomes = load_outcomes(base_dir)
    print(f"Source : {uq_samples_dir}", flush=True)
    print(f"Output : {out_root}", flush=True)
    print(f"Outcomes loaded: {len(outcomes)} tasks", flush=True)

    task_dirs = sorted(uq_samples_dir.iterdir(), key=lambda p: int(p.name))
    all_summaries = []

    for task_dir in task_dirs:
        task_id = int(task_dir.name)
        out_task = out_root / str(task_id)

        # Collect valid step files (must have exactly 10 samples)
        all_step_files = sorted(
            task_dir.glob("step_*_samples.json"),
            key=lambda p: int(re.search(r'step_(\d+)', p.name).group(1))
        )
        step_files = []
        for sf in all_step_files:
            with open(sf) as f:
                sd = json.load(f)
            n = len(sd.get("samples", []))
            if n < 10:
                print(f"\n  [WARN] task {task_id} {sf.name}: only {n} samples — skipping step", flush=True)
            else:
                step_files.append(sf)

        # Skip already-processed tasks if step count matches
        existing_summary = out_task / "task_hlvm_uq.json"
        if existing_summary.exists():
            n_hlvm = len(list(out_task.glob("step_*_hlvm_uq.json")))
            if n_hlvm == len(step_files):
                with open(existing_summary) as f:
                    summary = json.load(f)
                # Refresh outcome in case it changed, and persist to file
                summary["outcome"] = outcomes.get(task_id)
                with open(existing_summary, "w") as f:
                    json.dump(summary, f, indent=2)
                all_summaries.append(summary)
                print(f"\n=== Task {task_id} [skipped — already processed, {n_hlvm} steps] ===", flush=True)
                continue
            else:
                print(f"\n=== Task {task_id} [RE-PROCESSING — hlvm has {n_hlvm} steps, samples have {len(step_files)}] ===", flush=True)
                # Remove stale per-step hlvm files so they don't cause repeated mismatches
                for stale in out_task.glob("step_*_hlvm_uq.json"):
                    stale.unlink()

        print(f"\n=== Task {task_id} ===", flush=True)
        out_task.mkdir(parents=True, exist_ok=True)

        step_results   = []
        all_s_clusters = {}   # intention (S)
        all_c_clusters = {}   # state-change (C)

        for sf in step_files:
            with open(sf) as f:
                step_data = json.load(f)
            step_idx = step_data["step"]
            print(f"  step {step_idx}", flush=True)

            metrics, s_ci, c_ci = compute_step_hlvm(step_data)
            step_results.append(metrics)
            all_s_clusters[f"step_{step_idx}"] = s_ci
            all_c_clusters[f"step_{step_idx}"] = c_ci

            with open(out_task / f"step_{step_idx}_hlvm_uq.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # Save all clusters for intention (S) and state-change (C)
        with open(out_task / "intent_clusters.json", "w") as f:
            json.dump(all_s_clusters, f, indent=2)
        with open(out_task / "change_clusters.json", "w") as f:
            json.dump(all_c_clusters, f, indent=2)

        summary = aggregate_task(task_id, step_results)
        summary["outcome"] = outcomes.get(task_id)
        with open(out_task / "task_hlvm_uq.json", "w") as f:
            json.dump(summary, f, indent=2)

        all_summaries.append(summary)
        print(
            f"  K_S={summary.get('K_S_mean', 0):.2f}  "
            f"K_C={summary.get('K_C_mean', 0):.2f}  "
            f"U_sem={summary['U_sem_mean']:.3f}  "
            f"U_change={summary['U_change_mean']:.3f}  "
            f"U_ground={summary['U_ground_mean']:.3f}  "
            f"H_T={summary['H_T_mean']:.3f}  "
            f"C_S={summary['C_S_mean']:.3f}  "
            f"C_C={summary['C_C_mean']:.3f}  "
            f"R_T={summary['R_T_mean']:.3f}",
            flush=True,
        )

    with open(out_root / "all_tasks_hlvm_uq.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    eval_results = compute_eval(all_summaries, outcomes)
    auroc_dir = out_root / "auroc"
    auroc_dir.mkdir(exist_ok=True)
    with open(auroc_dir / "auroc_ece_report.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'Indicator':<28}  {'AUROC':>7}  {'AUROC_Var':>10}  {'ECE':>7}")
    print("-" * 60)
    for suf in SUFFIXES:
        print(f"\n  [aggregation: {suf}]")
        rows = [
            (feat, eval_results[f"{feat}_{suf}"])
            for feat in SCALAR_FEATURES
            if f"{feat}_{suf}" in eval_results
        ]
        for feat, d in sorted(rows, key=lambda x: -(x[1]["auroc"] or 0)):
            a   = d["auroc"]; v = d["auroc_var"]; e = d["ece"]
            a_s = f"{a:.4f}" if a is not None else "   N/A "
            v_s = f"{v:.6f}" if v is not None else "      N/A "
            e_s = f"{e:.4f}" if e is not None else "   N/A "
            print(f"  {feat+'_'+suf:<28}  {a_s:>7}  {v_s:>10}  {e_s:>7}")

    print(f"\nDone. Results in {out_root}")


if __name__ == "__main__":
    main()
