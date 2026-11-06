#!/usr/bin/env python3
"""Blind-spot analysis: which failed trajectories does the INTERNAL-signal
predictor miss at a fixed false-cut budget, and does the OBSERVABLE one catch
them?  Plus the failure-cause distribution of that missed set.

Population: the verbalized/baseline rollout (the only one where Macro and HTC
internal features are genuinely paired per row).  8 non-Claude combos.

Trajectory-level facts are derived from the keystep prefix labels:
  failed   <=> any(label == 1)
  k*        = first step_k with label == 1
  succeeded <=> all(label == 0)

Early-cut policy (same as the paper's Sec 4.3):
  cut at first k < T with p > tau
  Det  = frac of failed traj. cut with k_cut in [k*, T)
  FCR  = frac of successful traj. cut at all (k_cut < T)
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

SP = Path(__file__).resolve().parent
KS = SP.parent / "keystep_labels"
SCORERS = ["p_htc", "p_macro", "p_mi"]
NICE = {"p_htc": "internal-only", "p_macro": "observable(Macro)", "p_mi": "observable+internal"}
BUDGETS = [0.05, 0.10, 0.20]


def traj_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per trajectory with k*, T, and per-scorer first-cut step."""
    rows = []
    for uid, g in d.groupby("uid", sort=False):
        g = g.sort_values("step_k")
        lab = g["label"].values
        T = len(g)
        failed = bool(lab.max() == 1)
        kstar = int(np.argmax(lab)) if failed else -1
        r = {"uid": uid, "combo": g["combo"].iloc[0], "task_id": g["task_id"].iloc[0],
             "failed": failed, "kstar": kstar, "T": T}
        for s in SCORERS:
            r[s] = g[s].values
        rows.append(r)
    return pd.DataFrame(rows)


def first_cut(probs: np.ndarray, tau: float) -> int:
    """First index k < T-1 (strictly before the final step) with p > tau, else -1."""
    idx = np.where(probs[:-1] > tau)[0]
    return int(idx[0]) if len(idx) else -1


def det_fcr(tt: pd.DataFrame, scorer: str, tau: float):
    fail = tt[tt.failed]
    succ = tt[~tt.failed]
    det = np.mean([1.0 if (lambda k: k >= 0 and k >= r.kstar)(first_cut(getattr(r, scorer), tau))
                   else 0.0 for r in fail.itertuples()]) if len(fail) else np.nan
    fcr = np.mean([1.0 if first_cut(getattr(r, scorer), tau) >= 0 else 0.0
                   for r in succ.itertuples()]) if len(succ) else np.nan
    return det, fcr


def pick_tau(tt: pd.DataFrame, scorer: str, budget: float):
    """Smallest tau meeting FCR <= budget (maximises Det subject to the budget)."""
    grid = np.unique(np.round(np.concatenate([np.linspace(0, 1, 501)]), 4))
    best = None
    for tau in grid:
        det, fcr = det_fcr(tt, scorer, tau)
        if fcr <= budget:
            best = (tau, det, fcr)
            break
    return best if best else (1.0, 0.0, 0.0)


def load_causes() -> dict:
    """uid -> failureCause, from the BASELINE keystep role (this rollout)."""
    out = {}
    for bench in ["webarena", "mind2web"]:
        for f in sorted((KS / bench).glob("*_baseline.json")):
            model = f.name.replace("_baseline.json", "")
            combo = f"{bench}_{model}"
            items = json.load(open(f)).get("items", {})
            for _, v in items.items():
                tid = str(v.get("taskId"))
                cause = v.get("failureCause") or "Unannotated"
                out[f"{combo}::{tid}"] = cause
    return out


def load_pooled() -> pd.DataFrame:
    """Concatenate preds/*.csv, adding `combo` and a collision-free `uid`.

    join_id is only unique within a combo, so pooling without the prefix
    silently merges different trajectories.
    """
    import glob, os, re
    cached = SP / "preds_pooled.csv"
    if cached.exists():
        return pd.read_csv(cached)
    frames = []
    for f in sorted(glob.glob(str(SP / "preds" / "preds_*.csv"))):
        combo = re.sub(r"^preds_|\.csv$", "", os.path.basename(f))
        df = pd.read_csv(f)
        df["combo"] = combo
        df["uid"] = combo + "::" + df.join_id.astype(str)
        frames.append(df)
    if not frames:
        raise SystemExit("no preds/*.csv found -- run dump_preds.py --combos all first")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(cached, index=False)
    return out


def main():
    d = load_pooled()
    tt = traj_table(d)
    print(f"trajectories: {len(tt)}  (failed={tt.failed.sum()}, successful={(~tt.failed).sum()})")
    print(f"prefixes: {len(d)}\n")

    # ---- per-combo tau calibration, then pool the per-trajectory decisions ----
    for budget in BUDGETS:
        print("=" * 78)
        print(f"FALSE-CUT BUDGET = {budget:.0%}   (tau calibrated per combo, per scorer)")
        print("=" * 78)
        marks = {}
        for s in SCORERS:
            flags = {}
            for combo, sub in tt.groupby("combo"):
                tau, _, _ = pick_tau(sub, s, budget)
                for r in sub.itertuples():
                    k = first_cut(getattr(r, s), tau)
                    flags[r.uid] = (k >= 0 and r.failed and k >= r.kstar, k)
            marks[s] = flags
        tt[f"det_htc"] = tt.uid.map(lambda u: marks["p_htc"][u][0])
        tt[f"det_macro"] = tt.uid.map(lambda u: marks["p_macro"][u][0])
        tt[f"det_mi"] = tt.uid.map(lambda u: marks["p_mi"][u][0])
        tt[f"cut_macro"] = tt.uid.map(lambda u: marks["p_macro"][u][1])
        tt[f"cut_htc"] = tt.uid.map(lambda u: marks["p_htc"][u][1])

        fail = tt[tt.failed]
        for s, col in [("p_htc", "det_htc"), ("p_macro", "det_macro"), ("p_mi", "det_mi")]:
            print(f"  pooled Det  {NICE[s]:<22} = {fail[col].mean():.3f}  ({fail[col].sum()}/{len(fail)})")
        ct = pd.crosstab(fail.det_htc, fail.det_macro,
                         rownames=["internal detects"], colnames=["observable detects"])
        print("\n  cross-tab over failed trajectories:")
        print(ct.to_string().replace("\n", "\n  "))
        miss_h = fail[~fail.det_htc]
        rescued = miss_h[miss_h.det_macro]
        print(f"\n  internal-only MISSES {len(miss_h)} failed traj.; observable catches "
              f"{len(rescued)} of them = {len(rescued)/max(len(miss_h),1):.1%}")
        miss_m = fail[~fail.det_macro]
        rescued_h = miss_m[miss_m.det_htc]
        print(f"  observable  MISSES {len(miss_m)} failed traj.; internal   catches "
              f"{len(rescued_h)} of them = {len(rescued_h)/max(len(miss_m),1):.1%}")

        # lead time on jointly-detected
        both = fail[fail.det_htc & fail.det_macro]
        if len(both):
            lm = (both.cut_macro - both.kstar)
            lh = (both.cut_htc - both.kstar)
            print(f"  among jointly detected (n={len(both)}): median cut-minus-k*  "
                  f"observable={lm.median():.1f}  internal={lh.median():.1f}")
        print()

        if budget == 0.10:
            tt.to_csv(SP / "traj_decisions_b10.csv", index=False)
            causes = load_causes()
            fail = fail.copy()
            fail["cause"] = fail.uid.map(lambda u: causes.get(u, "Unannotated"))
            fail["module"] = fail.cause.str.split(":").str[0]
            print("  --- failure-cause MODULE distribution ---")
            allc = fail.module.value_counts(normalize=True)
            missc = fail[~fail.det_htc].module.value_counts(normalize=True)
            resc = rescued.copy()
            resc["module"] = resc.uid.map(lambda u: causes.get(u, "Unannotated").split(":")[0])
            rescc = resc.module.value_counts(normalize=True)
            tab = pd.DataFrame({"all failed": allc, "internal misses": missc,
                                "obs. rescues": rescc}).fillna(0)
            tab["n_all"] = fail.module.value_counts()
            print(tab.round(3).to_string().replace("\n", "\n  "))
            print()


if __name__ == "__main__":
    main()
