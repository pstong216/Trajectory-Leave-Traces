#!/usr/bin/env python3
"""
Generate Mind2Web/WebArena HLVM-UQ outputs for multiple embedding backends.

This is a generalized variant of:
  - compute_hlvm_uq_multivariable.py   (MiniLM)
  - reembed_qwen/reembed_qwen.py       (Qwen)

It keeps the HLVM parsing / clustering / metric computation fixed while
swapping only the embedding backend and clustering threshold.

Output layout defaults to:
  results_total/mind2web/embeded_hlvm/<source_name>/uq_analysis_change_<variant>/thr_<tag>/

Examples:
  python codes/reembed_hlvm_variants.py --source mind2web/claude --model minilm --threshold 0.3
  python codes/reembed_hlvm_variants.py --source mind2web/claude --model qwen --threshold 0.4
  python codes/reembed_hlvm_variants.py --source mind2web/claude --model bge_m3 --threshold 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score


RESULTS_ROOT = Path(__file__).resolve().parent.parent
MIN_SAMPLES_PER_STEP = 10
SCALAR_FEATURES = [
    "U_sem", "U_change", "U_ground", "U_cval",
    "H_T", "C_S", "C_C", "R_T",
    "K_S", "K_C", "num_targets",
]
SUFFIXES = ["mean", "max", "mean_max"]

IN_ORDER_RE = re.compile(
    r"[Ii]n order to\s+(.*?)(?:,\s*I will|,\s*I'll|\.\s*I will|\.\s*I'll|,\s*I am|\n)",
    re.DOTALL,
)
WOULD_RE = re.compile(
    r"\bwould\s+(.*?)(?:,\s*which\s+(?:is|will|would)|\s*->)",
    re.DOTALL,
)
BY_ACTION_RE = re.compile(
    r"\bBy\s+\w+ing\b[^,]*,\s*(.*?)(?:,\s*which\s+(?:is|will|would)|\s*->)",
    re.DOTALL,
)
VALUE_MAP = {"very_harmful": -2, "harmful": -1, "neutral": 0, "helpful": 1, "very_helpful": 2}
VALUE_RE = re.compile(r"->\s*(very_harmful|harmful|neutral|helpful|very_helpful)", re.IGNORECASE)
ACTION_ELEM_RE = re.compile(r"\[(\S+?)\]")
M2W_ACTION_RE = re.compile(r"```([^`]+)```")

DEVICE = "cpu"
try:
    import torch
    import torch.nn.functional as F

    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
except Exception:
    torch = None
    F = None


def threshold_tag(value: float) -> str:
    return str(value).replace(".", "p")


def variant_name(model_key: str) -> str:
        return {
            "minilm": "uq_analysis_change_minilm",
            "qwen": "uq_analysis_change_qwen",
            "bge_m3": "uq_analysis_change_bge_m3",
        }[model_key]


class Embedder:
    def __init__(self, model_key: str):
        self.model_key = model_key
        self._model = None
        self._tokenizer = None

    @property
    def model_name(self) -> str:
        return {
            "minilm": "all-MiniLM-L6-v2",
            "qwen": "Qwen/Qwen3-Embedding-0.6B",
            "bge_m3": "BAAI/bge-m3",
        }[self.model_key]

    def _ensure_minilm(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print("  [model] Loading all-MiniLM-L6-v2 ...", flush=True)
            self._model = SentenceTransformer(self.model_name)
            print("  [model] Loaded.", flush=True)

    def _ensure_qwen(self):
        if self._model is None:
            if torch is None or F is None:
                raise RuntimeError("torch is required for the Qwen embedding backend.")
            from transformers import AutoModel, AutoTokenizer

            print(f"  [model] Loading {self.model_name} on {DEVICE} ...", flush=True)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(DEVICE)
            self._model.eval()
            print("  [model] Loaded.", flush=True)

    def _ensure_bge(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"  [model] Loading {self.model_name} ...", flush=True)
            self._model = SentenceTransformer(self.model_name)
            print("  [model] Loaded.", flush=True)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.model_key == "minilm":
            self._ensure_minilm()
            return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if self.model_key == "qwen":
            return self._encode_qwen(texts)
        if self.model_key == "bge_m3":
            self._ensure_bge()
            return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        raise ValueError(f"Unsupported model_key={self.model_key}")

    def _encode_qwen(self, texts: List[str]) -> np.ndarray:
        self._ensure_qwen()
        all_embs = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)
            with torch.no_grad():
                out = self._model(**enc)
            seq_lens = enc["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(out.last_hidden_state.size(0), device=out.last_hidden_state.device)
            emb = out.last_hidden_state[batch_idx, seq_lens]
            emb = F.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
        return np.vstack(all_embs)

def _h(probs) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 1e-12))


def cluster_texts(texts: List[str], embedder: Embedder, threshold: float) -> Tuple[List[int], np.ndarray]:
    embeddings = embedder.encode_texts(texts)
    if len(texts) == 1:
        return [0], embeddings
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )
    labels = clustering.fit_predict(embeddings).tolist()
    return labels, embeddings


def parse_sample(sample: Dict):
    raw = sample.get("raw_prediction", "")
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
            if m else (atype if atype in ("scroll", "stop", "goto", "press", "type") else None)
        )
    else:
        target = None

    return intention, change, value, target


def compute_step_hlvm(step_data: Dict, embedder: Embedder, threshold: float) -> Tuple[Dict, Dict, Dict]:
    samples = step_data["samples"]
    parsed = [parse_sample(s) for s in samples]
    valid = [
        (intent, change, val, tgt)
        for intent, change, val, tgt in parsed
        if intent is not None and change is not None and tgt is not None
    ]
    M = len(valid)
    step_meta = {
        "task_id": step_data["task_id"],
        "step": step_data["step"],
        "page_url": step_data.get("page_url", ""),
        "action_taken": step_data.get("action_taken", ""),
        "num_samples": len(samples),
        "num_valid": M,
    }
    empty_ci = {"raw_texts": [], "cluster_ids": [], "cluster_to_labels": {}, "embeddings_shape": None}
    if M == 0:
        return (
            {**step_meta, "degenerate": True,
             "U_sem": 0.0, "U_change": 0.0, "U_ground": 0.0, "U_cval": 0.0,
             "H_T": 0.0, "C_S": 0.0, "C_C": 0.0, "R_T": 0.0,
             "identity_error": 0.0, "K_S": 0, "K_C": 0, "num_targets": 0,
             "P_SCT": {}, "P_S": {}, "P_C": {}, "P_T": {}},
            empty_ci,
            empty_ci,
        )

    s_texts = [x[0] for x in valid]
    s_ids, s_emb = cluster_texts(s_texts, embedder, threshold)
    c_texts = [x[1] for x in valid]
    c_ids, c_emb = cluster_texts(c_texts, embedder, threshold)

    intent_ci = {
        "raw_texts": s_texts,
        "cluster_ids": s_ids,
        "cluster_to_labels": _clusters_to_labels(s_ids, s_texts),
        "embeddings_shape": list(s_emb.shape),
    }
    change_ci = {
        "raw_texts": c_texts,
        "cluster_ids": c_ids,
        "cluster_to_labels": _clusters_to_labels(c_ids, c_texts),
        "embeddings_shape": list(c_emb.shape),
    }

    triples = [(s_id, c_id, tgt) for (_, _, _, tgt), s_id, c_id in zip(valid, s_ids, c_ids)]
    joint_counts: Dict[Tuple[int, int, str], int] = defaultdict(int)
    for sct in triples:
        joint_counts[sct] += 1
    P_SCT = {k: v / M for k, v in joint_counts.items()}

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
    U_sem = _h(list(P_S.values()))
    U_change = _h(list(P_C.values()))

    P_ST: Dict[Tuple[int, str], float] = defaultdict(float)
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

    P_SC: Dict[Tuple[int, int], float] = defaultdict(float)
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
    metrics = {
        **step_meta,
        "degenerate": False,
        "K_S": len(set(s_ids)),
        "K_C": len(set(c_ids)),
        "num_targets": len(unique_T),
        "U_sem": U_sem,
        "U_change": U_change,
        "U_ground": U_ground,
        "U_cval": U_cval,
        "H_T": H_T,
        "C_S": C_S,
        "C_C": C_C,
        "R_T": R_T,
        "identity_error": abs((C_S + C_C + R_T) - H_T),
        "P_SCT": {str(k): v for k, v in P_SCT.items()},
        "P_S": {str(k): v for k, v in P_S.items()},
        "P_C": {str(k): v for k, v in P_C.items()},
        "P_T": dict(P_T),
        "P_SC": {str(k): v for k, v in P_SC.items()},
        "P_ST": {str(k): v for k, v in P_ST.items()},
    }
    return metrics, intent_ci, change_ci


def _clusters_to_labels(ids: List[int], texts: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cid, txt in zip(ids, texts):
        out.setdefault(str(cid), []).append(txt)
    return out


def aggregate_task(task_id: int, steps: List[Dict]) -> Dict:
    valid = [s for s in steps if not s.get("degenerate", False)]
    agg = {"task_id": task_id, "num_steps": len(steps), "num_valid_steps": len(valid)}
    for feat in SCALAR_FEATURES:
        vals = [s[feat] for s in valid if feat in s]
        if vals:
            m = float(np.mean(vals))
            mx = float(np.max(vals))
            agg[f"{feat}_mean"] = m
            agg[f"{feat}_max"] = mx
            agg[f"{feat}_mean_max"] = (m + mx) / 2.0
        else:
            agg[f"{feat}_mean"] = agg[f"{feat}_max"] = agg[f"{feat}_mean_max"] = 0.0
    return agg


def compute_auroc(summaries: List[Dict], outcomes: Dict[int, float]) -> Dict:
    labeled = [t for t in summaries if t["task_id"] in outcomes]
    failure = np.array([1.0 - outcomes[t["task_id"]] for t in labeled])
    results = {}
    for feat in SCALAR_FEATURES:
        for suf in SUFFIXES:
            col = f"{feat}_{suf}"
            scores = np.array([t.get(col, 0.0) for t in labeled], dtype=float)
            if len(np.unique(scores)) < 2:
                results[col] = {"auroc": None}
                continue
            results[col] = {"auroc": round(float(roc_auc_score(failure, scores)), 4)}
    return results


def is_webarena(base_dir: Path) -> bool:
    return (base_dir / "trajectory_1" / "uq_samples").exists()


def load_outcomes_webarena(base_dir: Path) -> Dict[int, float]:
    actions_dir = base_dir / "trajectory_1" / "actions"
    outcomes: Dict[int, float] = {}
    if not actions_dir.exists():
        return outcomes
    for p in actions_dir.glob("*.json"):
        try:
            tid = int(p.stem)
            d = json.loads(p.read_text())
            s = d.get("score")
            if s is not None:
                outcomes[tid] = 1.0 if float(s) >= 1.0 else 0.0
        except (ValueError, KeyError):
            pass
    return outcomes


def load_outcomes_mind2web(base_dir: Path) -> Dict[int, float]:
    outcomes: Dict[int, float] = {}
    for result_file in sorted(base_dir.parent.glob("*_result.json")):
        d = json.loads(result_file.read_text())
        for task in d.get("tasks", []):
            try:
                tid = int(task["task_id"])
                status = task.get("status", "failure")
                outcomes[tid] = 1.0 if status == "success" else 0.0
            except (ValueError, KeyError):
                pass
    return outcomes


def _extract_parsed_action(raw: str) -> str:
    m = M2W_ACTION_RE.search(raw)
    return m.group(1).strip() if m else ""


def mind2web_result_to_step_data(task_index: int, step_idx: int, raw_samples: List[str], action_taken: str) -> Dict:
    samples = [{"raw_prediction": raw, "parsed_action": _extract_parsed_action(raw)} for raw in raw_samples]
    return {
        "task_id": task_index,
        "step": step_idx,
        "page_url": "",
        "action_taken": action_taken,
        "samples": samples,
    }


def build_task_entries(base_dir: Path) -> Tuple[str, Dict[int, float], List[Tuple[int, List[Dict]]]]:
    if is_webarena(base_dir):
        outcomes = load_outcomes_webarena(base_dir)
        uq_samples_dir = base_dir / "trajectory_1" / "uq_samples"
        task_entries = []
        for task_dir in sorted(uq_samples_dir.iterdir(), key=lambda p: int(p.name)):
            task_id = int(task_dir.name)
            step_files = sorted(
                task_dir.glob("step_*_samples.json"),
                key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)),
            )
            step_data_list = []
            for sf in step_files:
                sd = json.loads(sf.read_text())
                if len(sd.get("samples", [])) >= MIN_SAMPLES_PER_STEP:
                    step_data_list.append(sd)
            if step_data_list:
                task_entries.append((task_id, step_data_list))
        return "WebArena", outcomes, task_entries

    outcomes = load_outcomes_mind2web(base_dir)
    task_entries = []
    for task_dir in sorted(base_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not task_dir.is_dir() or not task_dir.name.isdigit():
            continue
        result_file = task_dir / "result.json"
        if not result_file.exists():
            continue
        d = json.loads(result_file.read_text())
        task_index = int(d["task_index"])
        step_samps = d.get("step_samples", [])
        actions = d.get("action_history", [])
        step_data_list = []
        for k, raw_list in enumerate(step_samps):
            if len(raw_list) >= MIN_SAMPLES_PER_STEP:
                action_taken = actions[k] if k < len(actions) else ""
                step_data_list.append(mind2web_result_to_step_data(task_index, k, raw_list, action_taken))
        if step_data_list:
            task_entries.append((task_index, step_data_list))
    return "Mind2Web", outcomes, task_entries


def process_task(task_id: int, step_data_list: List[Dict], out_task: Path, outcomes: Dict[int, float],
                 embedder: Embedder, threshold: float, force: bool, all_summaries: List[Dict]):
    existing = out_task / "task_hlvm_uq.json"
    n_steps = len(step_data_list)
    if not force and existing.exists():
        n_done = len(list(out_task.glob("step_*_hlvm_uq.json")))
        if n_done == n_steps:
            summary = json.loads(existing.read_text())
            summary["outcome"] = outcomes.get(task_id)
            existing.write_text(json.dumps(summary, indent=2))
            all_summaries.append(summary)
            print(f"=== Task {task_id} [skipped — {n_done} steps already done] ===", flush=True)
            return

    print(f"\n=== Task {task_id} ({n_steps} steps) ===", flush=True)
    out_task.mkdir(parents=True, exist_ok=True)
    step_results = []
    all_s_clusters = {}
    all_c_clusters = {}
    for step_data in step_data_list:
        step_idx = step_data["step"]
        print(f"  step {step_idx}", end=" ", flush=True)
        metrics, s_ci, c_ci = compute_step_hlvm(step_data, embedder, threshold)
        step_results.append(metrics)
        all_s_clusters[f"step_{step_idx}"] = s_ci
        all_c_clusters[f"step_{step_idx}"] = c_ci
        (out_task / f"step_{step_idx}_hlvm_uq.json").write_text(json.dumps(metrics, indent=2))
        print(f"K_S={metrics.get('K_S', 0)} K_C={metrics.get('K_C', 0)} U_sem={metrics.get('U_sem', 0):.3f}", flush=True)

    (out_task / "intent_clusters.json").write_text(json.dumps(all_s_clusters, indent=2))
    (out_task / "change_clusters.json").write_text(json.dumps(all_c_clusters, indent=2))
    summary = aggregate_task(task_id, step_results)
    summary["outcome"] = outcomes.get(task_id)
    (out_task / "task_hlvm_uq.json").write_text(json.dumps(summary, indent=2))
    all_summaries.append(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path relative to results_total/, e.g. mind2web/claude")
    ap.add_argument("--model", required=True, choices=["minilm", "qwen", "bge_m3"])
    ap.add_argument("--threshold", required=True, type=float)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--source_name", default=None, help="Override the output folder name (default: last component of --source)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base_dir = RESULTS_ROOT / args.source
    if not base_dir.exists():
        raise FileNotFoundError(f"Source not found: {base_dir}")

    benchmark = Path(args.source).parts[0]  # e.g. "mind2web" or "webarena"
    out_root_base = args.out_root if args.out_root is not None else str(RESULTS_ROOT / benchmark / "embeded_hlvm")
    source_name = args.source_name if args.source_name is not None else base_dir.name
    out_root = Path(out_root_base) / source_name / variant_name(args.model) / f"thr_{threshold_tag(args.threshold)}"
    out_root.mkdir(parents=True, exist_ok=True)

    fmt, outcomes, task_entries = build_task_entries(base_dir)
    embedder = Embedder(args.model)
    metadata = {
        "source": args.source,
        "format": fmt,
        "model_key": args.model,
        "model_name": embedder.model_name,
        "threshold": args.threshold,
        "output_dir": str(out_root),
        "device": DEVICE,
        "tasks": len(task_entries),
    }
    (out_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Format     : {fmt}")
    print(f"Source     : {base_dir}")
    print(f"Output     : {out_root}")
    print(f"Tasks      : {len(task_entries)}")
    print(f"Outcomes   : {len(outcomes)} tasks")
    print(f"Embed model: {embedder.model_name}")
    print(f"Threshold  : {args.threshold}")
    print(f"Device     : {DEVICE}\n", flush=True)

    all_summaries: List[Dict] = []
    for task_id, step_data_list in task_entries:
        process_task(task_id, step_data_list, out_root / str(task_id), outcomes, embedder, args.threshold, args.force, all_summaries)

    (out_root / "all_tasks_hlvm_uq.json").write_text(json.dumps(all_summaries, indent=2))
    if outcomes:
        auroc_results = compute_auroc(all_summaries, outcomes)
        auroc_dir = out_root / "auroc"
        auroc_dir.mkdir(exist_ok=True)
        (auroc_dir / "auroc_report.json").write_text(json.dumps(auroc_results, indent=2))

    print(f"\nDone. Results in {out_root}")


if __name__ == "__main__":
    main()
