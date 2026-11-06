#!/usr/bin/env python3
"""
Build the MACRO + MICRO feature matrix (one row per (task, step)) for
step-level risk scoring.

  MACRO: loop counts, action repetition, error counts, action-type breakdown,
        element diversity, thought-based error cues, grounding, stagnation,
        reasoning verbosity.
  MICRO (33 features):    11 HLVM UQ indicators × {at_k, cumul_mean, cumul_max}.

Label: outcome = 1 (failure) if task score < 1.0 (WebArena) or status="failure" (Mind2Web).

Detects WebArena vs Mind2Web automatically from directory structure.
Reads UQ metrics from --uq_dir inside --source by default, or from --uq_path
if an external organized UQ directory is provided.

Usage:
  python build_feature_matrix.py --source webarena/claude
  python build_feature_matrix.py --source mind2web/qwen30b
  python build_feature_matrix.py --source webarena/claude --uq_dir uq_analysis_change
  python build_feature_matrix.py --source mind2web/claude \
      --uq_path mind2web/embeded_hlvm/claude/uq_analysis_change_qwen/thr_0p3
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── UQ indicators ──────────────────────────────────────────────────────────────

UQ_INDICATORS = [
    "U_sem", "U_change", "U_ground", "U_cval",
    "H_T", "C_S", "C_C", "R_T",
    "K_S", "K_C", "num_targets",
]

UQ_FEATS = (
    [f"{ind}_at_k"       for ind in UQ_INDICATORS] +
    [f"{ind}_cumul_mean" for ind in UQ_INDICATORS] +
    [f"{ind}_cumul_max"  for ind in UQ_INDICATORS]
)

# ── Thought-text error cues ──────────────────────────────────────────────────

# Thought-based error features are intentionally simple rule-based detectors.
# They operate on the model's own free-form reasoning text and are shared by
# both WebArena and Mind2Web.
#
# - thought_action_ineffective: the agent explicitly says its action/strategy
#   had no effect or the page state did not change as expected.
# - thought_wrong_context: the agent realizes it is on the wrong page/site/flow.
# - thought_environment_error: the page/app/system itself is failing or blocked.
THOUGHT_CUE_PATTERNS = {
    "thought_action_ineffective": [
        r"\bineffective\b",
        r"not working",
        r"isn.t working",
        r"didn.t work",
        r"did not work",
        r"didn.t navigate",
        r"did not navigate",
        r"didn.t update",
        r"did not update",
        r"didn.t change",
        r"did not change",
        r"doesn.t navigate",
        r"does not navigate",
        r"doesn.t work",
        r"does not work",
        r"no change",
        r"no progress",
        r"unchanged",
        r"same state",
        r"page appears unchanged",
        r"page remained unchanged",
        r"without success",
        r"have not yielded any changes",
        r"has not yielded any changes",
        r"keeps concatenating",
        r"continues to concatenate",
        r"concatenated with",
        r"appended to the previous",
        r"appended to the previous destination",
        r"instead of replacing",
        r"didn.t register",
        r"did not register",
        r"didn.t select",
        r"did not select",
        r"didn.t apply",
        r"did not apply",
        r"current approach is ineffective",
        r"current method is ineffective",
    ],
    "thought_wrong_context": [
        r"wrong page",
        r"went to the wrong page",
        r"navigated to the wrong page",
        r"redirected me to the wrong page",
        r"redirected to the wrong page",
        r"not on .* page",
        r"not on .* section",
        r"not on .* subreddit",
        r"not on the correct page",
        r"instead of the store finder",
        r"instead of the store search",
        r"product search instead of",
        r"searched the product catalog instead of",
        r"used the product search instead of",
        r"global product search instead of",
        r"wrong website",
        r"wrong site",
        r"current page shows .* but i need",
        r"i'?m currently on .* but i need",
        r"this page (?:shows|is) .* but i need",
        r"the click took me to .* instead of",
        r"opened .* instead of",
        r"details page instead of",
        r"wiki documentation page instead of",
        r"took me to a different website",
        r"taken to a different website",
    ],
    "thought_environment_error": [
        r"application error",
        r"client-side exception",
        r"unexpected error",
        r"\b404\b",
        r"\b403\b",
        r"\b500\b",
        r"404 error",
        r"403 error",
        r"500 error",
        r"returned a 404 error",
        r"page (?:is|keeps) returning a 404 error",
        r"page shows an error",
        r"the page shows an error",
        r"site shows an error",
        r"access denied",
        r"permission denied",
        r"not available at this time",
        r"browser or app may not be secure",
        r"site is currently unavailable",
        r"site is erroring",
        r"failing to load",
        r"failed to load",
        r"didn.t load",
        r"did not load",
        r"isn.t loading",
        r"is not loading",
        r"not loading properly",
        r"page won.t load",
        r"page will not load",
        r"javascript.*not working",
        r"js.*not working",
        r"captcha",
        r"rate limit",
        r"too many requests",
        r"timed out",
        r"time(?:d)? out",
        r"blocked by",
        r"security warning",
        r"network error",
    ],
}

THOUGHT_CUE_REGEX = {
    name: [re.compile(p, re.IGNORECASE) for p in pats]
    for name, pats in THOUGHT_CUE_PATTERNS.items()
}

GROUNDING_PROXY_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"didn.t navigate",
        r"did not navigate",
        r"didn.t register",
        r"did not register",
        r"didn.t apply",
        r"did not apply",
        r"didn.t select",
        r"did not select",
        r"didn.t update",
        r"did not update",
        r"click may not have registered",
        r"selection didn.t register",
        r"selection did not register",
        r"location still shows",
        r"url still shows",
        r"page still shows",
        r"results still show",
        r"hasn.t changed",
        r"has not changed",
        r"hasn.t updated",
        r"has not updated",
    ]
]

# ── Format detection ───────────────────────────────────────────────────────────

def is_webarena(base_dir: Path) -> bool:
    return (base_dir / "trajectory_1" / "uq_samples").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# WEBARENA helpers
# ═══════════════════════════════════════════════════════════════════════════════

GROUNDING_RE = re.compile(
    r'Attempt to perfom .+? but no matching element found', re.IGNORECASE
)
PARSE_ERR_RE = re.compile(r'^The previous prediction you issued was', re.IGNORECASE)
STOP_RE      = re.compile(r'^stop\s*\[', re.IGNORECASE)
_WA_ATYPE_RE    = re.compile(r'^(\w+)', re.IGNORECASE)
_WA_ELEM_RE     = re.compile(r'\[(\d+)\]')
_WA_TYPE_TXT_RE = re.compile(r'\[\d+\]\s*\[([^\]]*)\]')


def _wa_is_grounding(a: str) -> bool:
    return bool(GROUNDING_RE.search(a))

def _wa_is_parse_err(a: str) -> bool:
    return bool(PARSE_ERR_RE.match(a))

def _wa_is_error(a: str) -> bool:
    return _wa_is_grounding(a) or _wa_is_parse_err(a)

def _wa_canonical(a: str) -> str:
    if _wa_is_error(a):
        return a[:60].lower()
    m_t = _WA_ATYPE_RE.match(a.strip())
    if not m_t:
        return a[:40].lower()
    atype = m_t.group(1).lower()
    m_e = _WA_ELEM_RE.search(a)
    if not m_e:
        return atype
    if atype == "type":
        # keep typed text so "type [66] [abc]" != "type [66] [xyz]"
        m_txt = _WA_TYPE_TXT_RE.search(a)
        if m_txt:
            return f"type [{m_e.group(1)}] [{m_txt.group(1)}]"
    return f"{atype} [{m_e.group(1)}]"

def _wa_action_type(a: str) -> Optional[str]:
    if _wa_is_error(a):
        return None
    m = _WA_ATYPE_RE.match(a.strip())
    return m.group(1).lower() if m else None

def _wa_type_text_len(a: str) -> int:
    m = _WA_TYPE_TXT_RE.search(a)
    return len(m.group(1).strip()) if m else 0


def extract_macro_webarena(actions: List[str], k: int) -> Dict:
    seq = actions[: k + 1]
    n   = len(seq)

    n_grounding = sum(_wa_is_grounding(a) for a in seq)
    n_parse     = sum(_wa_is_parse_err(a) for a in seq)
    streak = 0
    for a in reversed(seq):
        if _wa_is_error(a): streak += 1
        else: break

    canonical = [_wa_canonical(a) for a in seq]
    single_loops = double_loops = ground_loops = 0
    in_single = in_double = in_ground = False
    for i in range(1, n):
        if canonical[i] == canonical[i - 1]:
            if not in_single:
                single_loops += 1; in_single = True
            if _wa_is_grounding(seq[i]) and not in_ground:
                ground_loops += 1; in_ground = True
        else:
            in_single = in_ground = False
        if i >= 2 and canonical[i] == canonical[i-2] and canonical[i] != canonical[i-1]:
            if not in_double:
                double_loops += 1; in_double = True
        else:
            in_double = False

    loop_len = 0
    if n >= 2 and canonical[-1] == canonical[-2]:
        loop_len = 1
    elif (n >= 4 and canonical[-1] == canonical[-3]
          and canonical[-2] == canonical[-4]
          and canonical[-1] != canonical[-2]):
        loop_len = 2

    real_seq = [_wa_canonical(a) for a in seq if not _wa_is_error(a)]
    counts: Dict[str, int] = defaultdict(int)
    for a in real_seq: counts[a] += 1
    n_distinct  = max(len(counts), 1)
    cur_count   = counts.get(real_seq[-1], 1) if real_seq else 1
    max_count   = max(counts.values()) if counts else 1
    repeat_rate = (k + 1) / n_distinct

    type_counts: Dict[str, int] = defaultdict(int)
    elem_ids:    set             = set()
    total_typed  = 0
    last_new_step = 0
    seen_actions: set = set()
    for i, a in enumerate(seq):
        atype = _wa_action_type(a)
        if atype is not None:
            type_counts[atype] += 1
        m_e = _WA_ELEM_RE.search(a)
        if m_e and not _wa_is_error(a):
            elem_ids.add(m_e.group(1))
        if atype == "type":
            total_typed += _wa_type_text_len(a)
        c = _wa_canonical(a)
        if c not in seen_actions:
            seen_actions.add(c); last_new_step = i

    total_real = sum(type_counts.values())
    atype_entropy = -sum(
        (c / total_real) * math.log(c / total_real)
        for c in type_counts.values() if c > 0
    ) if total_real > 0 else 0.0

    return {
        "step_k":                 k,
        "n_single_loops":         single_loops,
        "n_double_loops":         double_loops,
        "loop_length_current":    loop_len,
        "n_distinct_actions":     n_distinct,
        "action_count_current":   cur_count,
        "max_action_count":       max_count,
        "action_repeat_rate":     round(repeat_rate, 4),
        "n_grounding_errors":     n_grounding,
        "n_parse_errors":         n_parse,
        "consecutive_errors_now": streak,
        "grounding_error_at_k":   int(_wa_is_grounding(seq[-1])),
        "parse_error_at_k":       int(_wa_is_parse_err(seq[-1])),
        "grounding_loop_count":   ground_loops,
        "n_click":                type_counts.get("click",   0),
        "n_type":                 type_counts.get("type",    0),
        "n_scroll":               type_counts.get("scroll",  0),
        "n_go_back":              type_counts.get("go_back", 0),
        "n_distinct_elements":    len(elem_ids),
        "action_type_entropy":    round(atype_entropy, 4),
        "steps_since_new_action": k - last_new_step,
        "type_text_length":       total_typed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MIND2WEB helpers
# ═══════════════════════════════════════════════════════════════════════════════
# Action format: "[click [2]] -> CLICK" or "[type [3] [text]] -> TYPE 'text'"
# Some trajectories also use plain-text forms like "scroll [down] -> SCROLL",
# "goto [url] -> NAVIGATE", or "go_back -> NAVIGATE".
# Mind2Web has no explicit action-level grounding error string like WebArena,
# but grounding failures can still be inferred from thought-text proxy cues.

_M2W_ATYPE_RE    = re.compile(r'^\[?(\w+)', re.IGNORECASE)
_M2W_ELEM_RE     = re.compile(r'\[(\d+)\]')
_M2W_TYPE_TXT_RE = re.compile(r'\[\d+\]\s*\[([^\]]*)\]')
_M2W_STOP_RE     = re.compile(r'^\[?stop', re.IGNORECASE)
_M2W_PARSE_ERR_RE = re.compile(r'unknown|failed to parse actions', re.IGNORECASE)


def _m2w_canonical(a: str) -> str:
    if _m2w_is_parse_err(a):
        return a[:60].lower()
    m_t = _M2W_ATYPE_RE.match(a.strip())
    if not m_t:
        return a[:40].lower()
    atype = m_t.group(1).lower()
    m_e = _M2W_ELEM_RE.search(a)
    if not m_e:
        return atype
    if atype == "type":
        # keep typed text so "[type [3] [abc]]" != "[type [3] [xyz]]"
        m_txt = _M2W_TYPE_TXT_RE.search(a)
        if m_txt:
            return f"type [{m_e.group(1)}] [{m_txt.group(1)}]"
    return f"{atype} [{m_e.group(1)}]"

def _m2w_action_type(a: str) -> Optional[str]:
    if _m2w_is_parse_err(a):
        return None
    m = _M2W_ATYPE_RE.match(a.strip())
    return m.group(1).lower() if m else None

def _m2w_is_parse_err(a: str) -> bool:
    return bool(_M2W_PARSE_ERR_RE.search(a))

def _m2w_type_text_len(a: str) -> int:
    m = _M2W_TYPE_TXT_RE.search(a)
    return len(m.group(1).strip()) if m else 0


def extract_macro_mind2web(actions: List[str], k: int) -> Dict:
    seq = actions[: k + 1]
    n   = len(seq)
    n_parse = sum(_m2w_is_parse_err(a) for a in seq)

    streak = 0
    for a in reversed(seq):
        if _m2w_is_parse_err(a): streak += 1
        else: break

    canonical = [_m2w_canonical(a) for a in seq]
    single_loops = double_loops = 0
    in_single = in_double = False
    for i in range(1, n):
        if canonical[i] == canonical[i - 1]:
            if not in_single:
                single_loops += 1; in_single = True
        else:
            in_single = False
        if i >= 2 and canonical[i] == canonical[i-2] and canonical[i] != canonical[i-1]:
            if not in_double:
                double_loops += 1; in_double = True
        else:
            in_double = False

    loop_len = 0
    if n >= 2 and canonical[-1] == canonical[-2]:
        loop_len = 1
    elif (n >= 4 and canonical[-1] == canonical[-3]
          and canonical[-2] == canonical[-4]
          and canonical[-1] != canonical[-2]):
        loop_len = 2

    counts: Dict[str, int] = defaultdict(int)
    for a in canonical: counts[a] += 1
    n_distinct  = max(len(counts), 1)
    cur_count   = counts.get(canonical[-1], 1)
    max_count   = max(counts.values()) if counts else 1
    repeat_rate = (k + 1) / n_distinct

    type_counts: Dict[str, int] = defaultdict(int)
    elem_ids:    set             = set()
    total_typed  = 0
    last_new_step = 0
    seen_actions: set = set()
    for i, a in enumerate(seq):
        atype = _m2w_action_type(a)
        if atype:
            type_counts[atype] += 1
        m_e = _M2W_ELEM_RE.search(a)
        if m_e:
            elem_ids.add(m_e.group(1))
        if atype == "type":
            total_typed += _m2w_type_text_len(a)
        c = _m2w_canonical(a)
        if c not in seen_actions:
            seen_actions.add(c); last_new_step = i

    total_real = sum(type_counts.values())
    atype_entropy = -sum(
        (c / total_real) * math.log(c / total_real)
        for c in type_counts.values() if c > 0
    ) if total_real > 0 else 0.0

    return {
        "step_k":                 k,
        "n_single_loops":         single_loops,
        "n_double_loops":         double_loops,
        "loop_length_current":    loop_len,
        "n_distinct_actions":     n_distinct,
        "action_count_current":   cur_count,
        "max_action_count":       max_count,
        "action_repeat_rate":     round(repeat_rate, 4),
        # Mind2Web parse-error features from UNKNOWN / parse-stop traces
        "n_grounding_errors":     0,
        "n_parse_errors":         n_parse,
        "consecutive_errors_now": streak,
        "grounding_error_at_k":   0,
        "parse_error_at_k":       int(_m2w_is_parse_err(seq[-1])),
        "grounding_loop_count":   0,
        # action-type breakdown
        "n_click":                type_counts.get("click", 0),
        "n_type":                 type_counts.get("type",  0),
        "n_scroll":               type_counts.get("scroll", 0),
        "n_go_back":              type_counts.get("go_back", 0),
        # diversity & stagnation
        "n_distinct_elements":    len(elem_ids),
        "action_type_entropy":    round(atype_entropy, 4),
        "steps_since_new_action": k - last_new_step,
        "type_text_length":       total_typed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MICRO (UQ) features — same for both formats
# ═══════════════════════════════════════════════════════════════════════════════

def load_step_uq(task_uq_dir: Path) -> Dict[int, Dict]:
    result = {}
    for f in task_uq_dir.glob("step_*_hlvm_uq.json"):
        m = re.search(r"step_(\d+)_hlvm_uq", f.name)
        if m:
            result[int(m.group(1))] = json.loads(f.read_text())
    return result


def extract_uq_at_k(step_uq: Dict[int, Dict], k: int) -> Dict:
    vals = defaultdict(list)
    for s in range(k + 1):
        d = step_uq.get(s)
        if d is None or d.get("degenerate", True):
            continue
        for ind in UQ_INDICATORS:
            v = d.get(ind)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals[ind].append(v)

    feats = {}
    for ind in UQ_INDICATORS:
        vs = vals[ind]
        feats[f"{ind}_cumul_mean"] = float(np.mean(vs))  if vs else float("nan")
        feats[f"{ind}_cumul_max"]  = float(np.max(vs))   if vs else float("nan")
        at_k_val = step_uq.get(k, {}).get(ind)
        feats[f"{ind}_at_k"] = (
            float(at_k_val)
            if at_k_val is not None and not (isinstance(at_k_val, float) and math.isnan(at_k_val))
            else float("nan")
        )
    return feats


def _text_has_any_pattern(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def load_thought_texts_webarena(uq_samples_dir: Path, task_id: int,
                                n_steps: int) -> List[List[str]]:
    task_dir = uq_samples_dir / str(task_id)
    result: List[List[str]] = []
    for k in range(n_steps):
        f = task_dir / f"step_{k}_samples.json"
        if not f.exists():
            result.append([])
            continue
        try:
            data = json.loads(f.read_text())
            samps = data.get("samples", [])
            texts = [s.get("raw_prediction", "") for s in samps if s.get("raw_prediction")]
            result.append(texts)
        except Exception:
            result.append([])
    return result


def load_thought_texts_mind2web(base_dir: Path, task_index: int,
                                n_steps: int) -> List[List[str]]:
    result_file = base_dir / str(task_index) / "result.json"
    if not result_file.exists():
        return [[] for _ in range(n_steps)]
    try:
        d = json.loads(result_file.read_text())
        step_samps = d.get("step_samples", [])
        thoughts = d.get("thoughts", [])
        result: List[List[str]] = []
        for k in range(n_steps):
            texts: List[str] = []
            if k < len(step_samps) and step_samps[k]:
                texts.extend([s for s in step_samps[k] if isinstance(s, str) and s])
            elif k < len(thoughts) and isinstance(thoughts[k], str) and thoughts[k]:
                texts.append(thoughts[k])
            result.append(texts)
        return result
    except Exception:
        return [[] for _ in range(n_steps)]


def thought_feats_at_k(step_texts: List[List[str]], k: int) -> Dict:
    feats = {}
    for cue_name, patterns in THOUGHT_CUE_REGEX.items():
        at_k = 0
        cumul = 0
        for s in range(k + 1):
            texts = step_texts[s] if s < len(step_texts) else []
            # A step counts once for a cue if any sampled thought at that step
            # explicitly describes that failure mode.
            hit = int(any(_text_has_any_pattern(t, patterns) for t in texts))
            if s == k:
                at_k = hit
            cumul += hit
        feats[f"{cue_name}_at_k"] = at_k
        feats[f"{cue_name}_cumul_count"] = cumul
    return feats


def grounding_proxy_feats_at_k(step_texts: List[List[str]], k: int) -> Dict:
    at_k = 0
    cumul = 0
    for s in range(k + 1):
        texts = step_texts[s] if s < len(step_texts) else []
        hit = int(any(_text_has_any_pattern(t, GROUNDING_PROXY_PATTERNS) for t in texts))
        if s == k:
            at_k = hit
        cumul += hit
    return {
        "grounding_proxy_at_k": at_k,
        "grounding_proxy_cumul_count": cumul,
    }


def grounding_feats_at_k(explicit_flags: List[bool], step_texts: List[List[str]], k: int) -> Dict:
    combined: List[bool] = []
    for s in range(k + 1):
        texts = step_texts[s] if s < len(step_texts) else []
        proxy_hit = any(_text_has_any_pattern(t, GROUNDING_PROXY_PATTERNS) for t in texts)
        explicit_hit = explicit_flags[s] if s < len(explicit_flags) else False
        combined.append(bool(explicit_hit or proxy_hit))

    loops = 0
    in_loop = False
    for i in range(1, len(combined)):
        if combined[i] and combined[i - 1]:
            if not in_loop:
                loops += 1
                in_loop = True
        else:
            in_loop = False

    return {
        "n_grounding_errors": sum(combined),
        "grounding_error_at_k": int(combined[-1]) if combined else 0,
        "grounding_loop_count": loops,
    }


def combined_error_feats_at_k(parse_flags: List[bool], explicit_grounding_flags: List[bool],
                              step_texts: List[List[str]], k: int) -> Dict:
    # Final benchmark-shared error definition:
    # - grounding error: explicit grounding signal OR thought-derived grounding proxy
    # - consecutive error streak: trailing run of (grounding error OR parse error)
    grounding_feats = grounding_feats_at_k(explicit_grounding_flags, step_texts, k)

    trailing = 0
    for s in range(k, -1, -1):
        texts = step_texts[s] if s < len(step_texts) else []
        proxy_hit = any(_text_has_any_pattern(t, GROUNDING_PROXY_PATTERNS) for t in texts)
        explicit_grounding = explicit_grounding_flags[s] if s < len(explicit_grounding_flags) else False
        parse_hit = parse_flags[s] if s < len(parse_flags) else False
        if explicit_grounding or proxy_hit or parse_hit:
            trailing += 1
        else:
            break

    grounding_feats["consecutive_errors_now"] = trailing
    return grounding_feats


# ═══════════════════════════════════════════════════════════════════════════════
# Reasoning-length features — format-specific loading, same extraction
# ═══════════════════════════════════════════════════════════════════════════════

def load_reasoning_lengths_webarena(uq_samples_dir: Path, task_id: int,
                                     n_steps: int) -> List:
    task_dir = uq_samples_dir / str(task_id)
    result = []
    for k in range(n_steps):
        f = task_dir / f"step_{k}_samples.json"
        if not f.exists():
            result.append(None); continue
        try:
            data  = json.loads(f.read_text())
            samps = data.get("samples", [])
            result.append(float(np.mean([len(s["raw_prediction"]) for s in samps])) if samps else None)
        except Exception:
            result.append(None)
    return result


def load_reasoning_lengths_mind2web(base_dir: Path, task_index: int,
                                     n_steps: int) -> List:
    result_file = base_dir / str(task_index) / "result.json"
    if not result_file.exists():
        return [None] * n_steps
    try:
        d          = json.loads(result_file.read_text())
        step_samps = d.get("step_samples", [])
        result     = []
        for k in range(n_steps):
            if k < len(step_samps) and step_samps[k]:
                result.append(float(np.mean([len(s) for s in step_samps[k]])))
            else:
                result.append(None)
        return result
    except Exception:
        return [None] * n_steps


def reasoning_len_feats_at_k(lengths: List, k: int) -> Dict:
    valid = [v for v in lengths[: k + 1] if v is not None]
    return {
        "reasoning_len_at_k":       lengths[k] if lengths[k] is not None else float("nan"),
        "reasoning_len_cumul_mean": float(np.mean(valid)) if valid else float("nan"),
        "reasoning_len_cumul_max":  float(np.max(valid))  if valid else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Outcomes
# ═══════════════════════════════════════════════════════════════════════════════

def load_outcomes_webarena(base_dir: Path) -> Dict[int, int]:
    actions_dir = base_dir / "trajectory_1" / "actions"
    outcomes: Dict[int, int] = {}
    if not actions_dir.exists():
        return outcomes
    for p in actions_dir.glob("*.json"):
        try:
            tid = int(p.stem)
            d   = json.loads(p.read_text())
            s   = d.get("score")
            if s is not None:
                # WebArena tasks count as success only at the exact full score.
                # Any other score is treated as failure for step-level risk labels.
                outcomes[tid] = 0 if float(s) == 1.0 else 1  # 1 = failure
        except (ValueError, KeyError):
            pass
    return outcomes


def load_outcomes_mind2web(base_dir: Path) -> Dict[int, int]:
    # Backbone-aware: prefer the result file matching this source directory's
    # name. The old glob("*_result.json")[0] picked the same file for every
    # backbone (all matrices silently got gptnano's labels — bug found 2026-07-10).
    named_candidates = [
        base_dir.parent / f"{base_dir.name}_result.json",         # mind2web/claude
        base_dir.parent / f"{base_dir.parent.name}_result.json",  # gptnano nested layout
        base_dir.parent / "qwen_result.json",                     # mind2web/qwen30b
    ]
    result_file = next((c for c in named_candidates if c.exists()), named_candidates[-1])
    outcomes: Dict[int, int] = {}
    if not result_file.exists():
        print(f"  [WARN] outcomes file not found: {result_file}", flush=True)
        return outcomes
    d = json.loads(result_file.read_text())
    for task in d.get("tasks", []):
        try:
            tid    = int(task["task_id"])
            status = task.get("status", "failure")
            outcomes[tid] = 0 if status == "success" else 1  # 1 = failure
        except (ValueError, KeyError):
            pass
    return outcomes


# ═══════════════════════════════════════════════════════════════════════════════
# Feature matrix builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(base_dir: Path, uq_base: Path) -> pd.DataFrame:
    wa       = is_webarena(base_dir)
    fmt      = "WebArena" if wa else "Mind2Web"
    outcomes = load_outcomes_webarena(base_dir) if wa else load_outcomes_mind2web(base_dir)

    print(f"  Format   : {fmt}")
    print(f"  UQ dir   : {uq_base}")
    print(f"  Outcomes : {len(outcomes)} tasks", flush=True)

    rows = []

    if wa:
        # ── WebArena ──────────────────────────────────────────────────────────
        actions_dir    = base_dir / "trajectory_1" / "actions"
        uq_samples_dir = base_dir / "trajectory_1" / "uq_samples"

        for act_file in sorted(actions_dir.glob("*.json"), key=lambda p: int(p.stem)):
            task_id = int(act_file.stem)
            data    = json.loads(act_file.read_text())
            score   = data.get("score")
            if score is None:
                continue
            outcome = outcomes.get(task_id, 1)
            actions = [a for a in data.get("actions", []) if not STOP_RE.match(a)]
            if not actions:
                continue

            step_uq = load_step_uq(uq_base / str(task_id)) if (uq_base / str(task_id)).exists() else {}
            rlens   = load_reasoning_lengths_webarena(uq_samples_dir, task_id, len(actions))
            ttexts  = load_thought_texts_webarena(uq_samples_dir, task_id, len(actions))
            explicit_grounding = [_wa_is_grounding(a) for a in actions]
            parse_flags = [_wa_is_parse_err(a) for a in actions]

            for k in range(len(actions)):
                row = {"task_id": task_id, "source": fmt, "outcome": outcome}
                row.update(extract_macro_webarena(actions, k))
                row.update(extract_uq_at_k(step_uq, k))
                row.update(thought_feats_at_k(ttexts, k))
                row.update(combined_error_feats_at_k(parse_flags, explicit_grounding, ttexts, k))
                row.update(reasoning_len_feats_at_k(rlens, k))
                rows.append(row)

    else:
        # ── Mind2Web ──────────────────────────────────────────────────────────
        for task_dir in sorted(
            (p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        ):
            task_index  = int(task_dir.name)
            result_file = task_dir / "result.json"
            if not result_file.exists():
                continue
            d       = json.loads(result_file.read_text())
            outcome = outcomes.get(task_index, 1)
            actions = [
                a for a in d.get("action_history", [])
                if not _M2W_STOP_RE.match(a)
            ]
            if not actions:
                continue

            step_uq = load_step_uq(uq_base / str(task_index)) if (uq_base / str(task_index)).exists() else {}
            rlens   = load_reasoning_lengths_mind2web(base_dir, task_index, len(actions))
            ttexts  = load_thought_texts_mind2web(base_dir, task_index, len(actions))
            explicit_grounding = [False] * len(actions)
            parse_flags = [_m2w_is_parse_err(a) for a in actions]

            for k in range(len(actions)):
                row = {"task_id": task_index, "source": fmt, "outcome": outcome}
                row.update(extract_macro_mind2web(actions, k))
                row.update(extract_uq_at_k(step_uq, k))
                row.update(thought_feats_at_k(ttexts, k))
                row.update(combined_error_feats_at_k(parse_flags, explicit_grounding, ttexts, k))
                row.update(reasoning_len_feats_at_k(rlens, k))
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("  Rows     : 0  (0 tasks, 0 failure steps)", flush=True)
    else:
        print(f"  Rows     : {len(df)}  ({df['task_id'].nunique()} tasks, "
              f"{df['outcome'].sum()} failure steps)", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", required=True,
        help="Path relative to results_total/, e.g. webarena/claude or mind2web/qwen30b",
    )
    ap.add_argument(
        "--uq_dir", default="uq_analysis_change_qwen",
        help="Name of UQ output directory inside --source "
             "(ignored if --uq_path is set; default: uq_analysis_change_qwen)",
    )
    ap.add_argument(
        "--uq_path", default=None,
        help="Path to the UQ directory to read. Use this when UQ outputs live "
             "outside --source, e.g. in embeded_hlvm.",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output CSV path. Default: <uq_path>/feature_matrix.csv "
             "or {source}/{uq_dir}/feature_matrix.csv",
    )
    args = ap.parse_args()

    base_dir = Path(__file__).parent.parent.parent / args.source
    if not base_dir.exists():
        raise FileNotFoundError(f"Source not found: {base_dir}")

    print(f"Source : {base_dir}")
    uq_base = Path(args.uq_path) if args.uq_path else (base_dir / args.uq_dir)
    if not uq_base.is_absolute():
        uq_base = Path(__file__).parent.parent.parent / uq_base
    if not uq_base.exists():
        raise FileNotFoundError(f"UQ path not found: {uq_base}")

    print(f"UQ dir : {uq_base}")
    df = build_feature_matrix(base_dir, uq_base)

    out_path = Path(args.out) if args.out else uq_base / "feature_matrix.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved  : {out_path}")
    print(f"Shape  : {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
