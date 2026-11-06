#!/usr/bin/env python3
"""
run_website_generalization.py — Leave-one-website-category-out generalization,
fully self-contained inside this repository (2026-07-31 rebuild; consolidated
2026-07-31: category-assignment logic, data loading, model fitting, and
figure generation all live in this ONE file -- no other script in this repo
needs to be edited to reproduce either the numbers or the figure).

Protocol (per protocol): hold out ONE website-category regime for testing,
train on the pooled remaining regimes, report AUROC / e-AURC / Brier.
Every number in this script comes from data + 
keystep_labels + a reproduce.sh run (
reproduction_output/), never from results_total/baseline or results_total/
codes directly, so nothing here can silently depend on a working-tree file
that isn't actually part of the release.

Methods reported, all evaluated on the SAME 5-way leave-one-category-out
split (pooled across all 10 backbone x benchmark combos):

  Cross-category (retrained per held-out category):
    - Ours Logistic_L2, macro+micro ("all")
    - Ours Logistic_L2, micro only  ("uq_only")
    - Ours Logistic_L2, macro only  ("macro_only")
    - M11  HTC full/L2  (logprob-derived features, step-weighted per the
      established 2026-07-31 default)
    - M13  Stacked HTC_L2 over M1-M4/M7 scalar scores
  (M12 dropped 2026-07-31, design decision -- same call as the main table:
   only M11 is reported for the HTC full/L2 family.)

  Cross-domain, ranking fixed but calibration fitted (M1-M7/M15's RAW
  scores have no fitted parameters, and AUROC/e-AURC are rank-invariant
  under any monotone rescaling, so for those two metrics
  "leave-one-category-out" just means: evaluate on the held-out
  category's rows. Brier is different: it needs an actual probability,
  and the score->probability mapping IS fitted on training data. Fixed
  2026-08-31 (caught in review): each score is now held-out Platt-scaled -- a
  1-D LogisticRegression fit on the pooled other-4-categories rows,
  predicted on the held-out category -- matching the main table's
  brier_platt convention (baseline_calculation/
  update_baselines_with_features.py::calibrate_platt). The previous
  MinMaxScaler was NOT a proper calibrator (pure range rescale, optimises
  no proper scoring rule) and inflated M2/M3/M4/M15's Brier to 0.44-0.51,
  worse than a constant base-rate predictor (~0.21). AUROC/e-AURC still
  rank on the raw oriented score, never on the Platt output, because
  Platt's fitted slope can flip the orientation of train-anti-correlated
  scores like M3/M4 -- also exactly the main table's convention: auroc
  from raw conf, brier from brier_platt.):
    - M1 verbalized_conf, M2 action_logprob_mean, M3 output_logprob_mean,
      M4 cumulative_output_logprob_mean, M5 action_entropy,
      M6 cumulative_action_entropy, M7 conf_cumul_mean,
      M15 cumulative_action_logprob_mean (2026-08-02, added: cumulative
      (0->k) version of M2, the same recipe as M3->M4 and M1->M7 -- M2 was
      the only one of the three at-k scalar scores with no cumulative
      counterpart; see baseline_extension/compute_m2_cumulative.py)

  In-domain reference, TWO variants:
    - whole-combo (NOT held out by category -- the paper's own reported
      numbers, pulled directly from the reproduce.sh Ours run's
      summary_table.csv, mean across the 5 within-combo folds x 10 combos)
    - matched-per-category (added 2026-07-31, in response to "is
      cross-category beating in-domain even plausible?" -- SAME test
      population as the cross-category run for that category, standard
      5-fold CV WITHIN the category instead of holding it out, so the only
      thing that changes vs. the cross-category number is whether that
      category's own tasks were excluded from training. See
      run_indomain_percat()'s docstring.)

M8/M9/M10 excluded by design directive (2026-07-30: not genuine held-out
baselines, see update_baselines_with_features.py's docstring).

All methods share ONE label/metric convention throughout this script:
  y_fail = 1 iff the step is at/after the task's keyStep (task ultimately
  fails) -- the SAME "label" convention already used by Ours' training
  pipeline (train_risk_model_temp_sca.py) and by aurc()/aurc_oracle() there
  (y_true=1=failure, y_prob=P(failure), lower e-AURC = better, i.e. exactly
  the oracle-referenced Excess-AURC used everywhere else in this project).
  M1-M13's own "success"-oriented raw scores are converted to this
  convention before scoring.

Outputs (code/generalization/results/):
  cross_category_results.csv       -- per (method, feature_set, held_out_category)
  cross_category_summary.csv       -- mean +/- std across the 5 held-out categories
  in_domain_reference.csv          -- Ours L2, whole-combo standard CV numbers
  category_stats.csv               -- task counts per category per combo
  website_generalization_figure.png -- 3-panel AUROC/e-AURC/Brier chart,
                                        style: red "Cross-Category Ours" hero
                                        line (value-annotated) + blue dashed
                                        "In-Domain (matched)" hero line, over
                                        a muted cloud of every other baseline
                                        (see results_total/generalization/
                                        exp6_cross_website/results/
                                        rf_generalization_figure.png for the
                                        reference style this reproduces)
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PR = Path(__file__).resolve().parent.parent.parent          # 
REPRO = PR / "reproduction_output"

sys.path.insert(0, str(PR / "code" / "baseline_extension"))
from compute_m2_cumulative import add_m2_cumulative  # noqa: E402 -- M15 feature
sys.path.insert(0, str(PR / "code" / "l2_training"))
sys.path.insert(0, str(PR / "code" / "baseline_calculation"))
sys.path.insert(0, str(PR / "code" / "htc"))

from train_risk_model_temp_sca import fit_model, _cv3, safe_medians, aurc, aurc_oracle  # noqa: E402
from keystep_labels import load_keystep_map_entropy_source  # noqa: E402
from htc_step_baseline import build_htc_step_dataset, best_C_for_fold  # noqa: E402

COMBOS = [
    ("webarena", "claude"), ("webarena", "gpt_5_2"), ("webarena", "gptnano"),
    ("webarena", "kimi"), ("webarena", "qwen30b"),
    ("mind2web", "claude"), ("mind2web", "gpt_5_2"), ("mind2web", "gptnano"),
    ("mind2web", "kimi"), ("mind2web", "qwen30b"),
]

# ═════════════════════════════════════════════════════════════════════════
# Website-category assignment (inlined 2026-07-31 from the former
# website_categories.py, by design to consolidate everything into
# one file). Reused VERBATIM from generalization/exp6_cross_website/
# run_cross_website.py's WA_PORT_MAP + MW_CATEGORY_RULES -- pure
# website-content classification, untouched by anything else this session
# changed. Only get_webarena_category()/get_mind2web_category()'s file
# paths are adapted to the  layout:
#   webarena page_url:  data/webarena/<model>/embedded/uq_analysis_change_qwen/thr_0p3/<task_id>/step_0_hlvm_uq.json
#   mind2web task text: data/mind2web/<model>/raw/plain/<task_id>/result.json
# ═════════════════════════════════════════════════════════════════════════

WEBSITE_CATEGORIES = [
    "ecommerce", "devtools", "social_entertain", "health_services", "travel_navigate",
]

WA_PORT_MAP = {
    "7770": "ecommerce", "7780": "ecommerce",
    "8023": "devtools",
    "9999": "social_entertain",
    "3000": "travel_navigate",
}

MW_CATEGORY_RULES = [
    ("devtools", [
        "github", "sourceforge", "hugging face", "arxiv", "open-source",
        "repository", "cvpr", "iclr ", "sam2", "nielsrogge",
        "y combinator", "ycombinator", "hgx h100", "proton emission",
        "quantum physics", "multiplication quiz", "coolmath",
        "claude api", "anthropic api", "aeroapi", "flightaware",
        "storage plan", "computer science course", "chemistry course",
        "class schedule", "graduate-level", "ted talk",
        "press release", "publication at", "best paper award",
        "nvidia learning", "profile page of the leader",
        "pricing plan for", "input tokens",
    ]),
    ("ecommerce", [
        "amazon", " target ", "best buy", "micro center", "uniqlo",
        " zara ", "birkenstock", "dillard", "carmax", "kbb",
        " cadillac", " acura", " bmw", " tesla ", " toyota",
        " honda ", " nissan", " audi ", " porsche", "trader joe",
        "macbook", "ipad", "iphone", "nordictrack", "samsung ssd",
        "samsung tv", "samsung galaxy", "samsung s25", "qled",
        "gaming monitor", "used car", "used hatchback",
        "certified pre-owned", "car listing", "car loan",
        "ownership cost", "vinyl record", "hiking shoe",
        " sofa", "handbag", "van for sale", "open-box",
        "gift card", "app for ios", "refurbished ipad",
        "taco bell", "swimsuit", " pajamas", " shoes ",
        "computer monitor", "headphones", " laptop",
        "refrigerator", "smart tv", "screen size",
        " wine ", "groceries", "mac studio",
        "jersey", " t-shirt", "men's t-shirt", "shirt in",
        "couch", "couches for sale", "coffee maker",
        "doorbell", "speakers that are", "bluetooth",
        "nest hello", "apple watch", " on apple",
        "apple services", "store near zip",
        "anti-reflective tv", "tv with screen",
        "drip coffee", "zara home",
        "refridgerator", "box combo", "kids combo",
        "lilo", "stitch toys", "xiaomi",
        "smartphones manufactured", " wine,", "wine from",
        "shipping available for", "fastest shipping",
    ]),
    ("social_entertain", [
        "imdb", "rotten tomatoes", "9gag", "discogs",
        "fox sports", "nfl ", " nba ", "nhl ", "mls ",
        "formula 1", "superbike", "ign ", "ticketmaster",
        "steam ", "board game", "boardgame",
        "league of legends", " ahri", "doraemon",
        "most popular tv", "playoff", "rock chart",
        "soundcloud", "chess puzzle", "chess ",
        "meme ", "youtube kids", "top 250 movies",
        "amc+", "lady gaga", "concert in new york",
        " sports ", "standings for", "race time",
        "tumblr", "matrix wiki", "theatre event",
        "theater event", "walkthrough for the game",
        "blocked shots", "assists", "devin booker",
        "online competition", "prize", "competition code",
        "obedience trial", "latest negative reviews from players",
        "miami for 2 adults",
    ]),
    ("health_services", [
        "drug interaction", "interaction between",
        "dosage for", "side effects of",
        "diagnosis", "nursing home", "medicare",
        "pregnancy", "fertility", "ovulation",
        "pill with", "pediatrician", "cardiologist",
        "audiologist", "neurosurgeon", "dentist",
        "dermatolog", "optometrist", " vet ",
        "personal trainer service", "weight management",
        "teen driver safety", "vehicle registration",
        "filing fee", "child benefit",
        "inspector general", "unemployment",
        "income tax", "civil division forms",
        "dmv center", "athletic trainer",
        "nursing home", "pharmacy ",
        "symptom", "cold and flu", "doctor",
        "medicare", "medicaid",
        "birth certificate", "government benefit",
        "federal form", "irs form", "tax form",
        "social security", "public assistance",
        "humira", "vivitrol", "novolin", "montelukast", "elevate at",
        "melatonin", "eulexin", "folate",
        "drug ", "dosage ", "interaction for",
        "stock price", "bitcoin", "crypto", "xrp",
        "401k", "retirement", "federal income",
        "google finance", "cboe", " etf ",
        "eur/usd", "price chart", "monthly price",
        "loan payment", "digital trends report",
        "nyse rule", "etp odd lot",
        "job in ", "jobs in ", "jobs within", "jobs near",
        "job within", "marketing jobs", "internship ",
        "faculty position", "electrician",
        "software developer", "medical health",
        "pharmacy position", "legal occupation",
        "cleaning service", "photographer",
        "solar energy quote", "solar panel",
        "phone number scam", "phone scam", "is a scam",
        "solar energy equipment", "solar energy quote",
        "form 8843", "tax year 20",
        "roofing contractor", "contractor near",
        "dog groomer", "groomer for",
        "divorce lawyer", "work visa", "uk visa",
        "accredited charity", "charity near",
        "get accredited", "how to get accredited",
        "self-storage", "storage unit near",
        "impact statement", "environmental impact",
        "climate news", "occupational fatalit",
        "adoptable dog", "adoptable cat",
        "animal shelter", "rescue group",
        "kitten", "puppy", " cat near", "cats available for",
        " dog near", "hairless dog", "english bulldog",
        "rabbit", "breed comparison", "breeds",
        "boxer near", "husky near", "spayed",
        "neutered dog", "senior dog", "young cat",
        "shorthaired dog", "dog cared for",
        "child's height", "baby boy was born",
        "girl child's height", "height at seven",
        "benefits and financial", "living in england",
        "recipe", "pancake", "eggplant",
        "gluten-free", "vegetarian", "pescatarian",
        "baking", "dish with",
        "apartment", "house for sale", "house for rent",
        "condo ", "student housing", "land for",
        " rental", "zillow", "real estate",
        "paddling permit", "park permit",
        "shared rooms in", "rooms in university",
        "houses for sale", "farms land",
        "hunting land", "rent for a",
        "alpine ridge", "newsletter subscription",
        "gift to um-", "give a gift to",
        "school activity events", "courses related to",
        "beginner", "course in computer science",
        "uk for longer", "work in the uk",
        "bbb accredited", "natural products database",
        "bedroom condos", "bedroom condo",
    ]),
    ("travel_navigate", [
        "weather", "forecast", "temperature",
        "air quality", "fedex", " ups ", "usps",
        "megabus", "shipping cost", "shipping rate",
        "flat-rate shipping", "flat-rate box",
        "transit", "mta ", "amtrak", "ryanair",
        "flight from", "flight leaving", "ua or aa flights",
        " hotel", "marriott", "airbnb",
        "cruise ", "airport", "parking near",
        "road trip", "train ticket", "package weighing",
        "gas station", "tourist attraction",
        "travel deal", "bus s92", "bus route",
        "neighborhood maps", "alley spring",
        "alley creek", "national park", "albion basin",
        "horseback riding", "paddleboard",
        "nearby attraction", "tourism",
        "qatar airways", "roundtrip leaving",
        "po box", "p.o. box", "order tracking",
        "order number", "track the details",
        "tracking information for shipment",
        "onboard activities", "cheapest cruise",
        "wind speed", "wind flow", "wind map",
    ]),
]


def get_webarena_category(task_id: int, embedded_thr_base: Path) -> str:
    step_file = embedded_thr_base / str(task_id) / "step_0_hlvm_uq.json"
    if not step_file.exists():
        return "unknown"
    try:
        url = json.loads(step_file.read_text()).get("page_url", "")
        m = re.search(r":(\d{4})", url)
        if m:
            return WA_PORT_MAP.get(m.group(1), "unknown")
    except Exception:
        pass
    return "unknown"


def get_mind2web_category(task_id: int, raw_plain_base: Path) -> str:
    result_file = raw_plain_base / str(task_id) / "result.json"
    if not result_file.exists():
        return "unknown"
    try:
        task_text = json.loads(result_file.read_text()).get("task", "").lower()
        for cat, keywords in MW_CATEGORY_RULES:
            if any(kw in task_text for kw in keywords):
                return cat
    except Exception:
        pass
    return "misc"


# ═════════════════════════════════════════════════════════════════════════
# Feature sets
# ═════════════════════════════════════════════════════════════════════════

MACRO_FEATS = [
    "step_k", "n_single_loops", "n_double_loops", "loop_length_current",
    "n_distinct_actions", "action_count_current", "max_action_count",
    "action_repeat_rate", "n_grounding_errors", "n_parse_errors",
    "consecutive_errors_now", "grounding_error_at_k", "parse_error_at_k",
    "grounding_loop_count", "n_click", "n_type", "n_scroll", "n_go_back",
    "n_distinct_elements", "action_type_entropy", "steps_since_new_action",
    "type_text_length",
    "thought_action_ineffective_at_k", "thought_action_ineffective_cumul_count",
    "thought_wrong_context_at_k", "thought_wrong_context_cumul_count",
    "thought_environment_error_at_k", "thought_environment_error_cumul_count",
    "reasoning_len_at_k", "reasoning_len_cumul_mean", "reasoning_len_cumul_max",
]  # 31 features
UQ_FEATS = [
    f"{ind}_{suf}"
    for ind in ["U_sem", "U_change", "U_ground", "U_cval", "C_S", "R_T"]
    for suf in ["at_k", "cumul_mean", "cumul_max"]
]  # 18 features (6 indicators x 3 aggregations)
FEATURE_SETS = {"macro_only": MACRO_FEATS, "uq_only": UQ_FEATS, "all": MACRO_FEATS + UQ_FEATS}  # 49 total

# M1-M7 raw-score orientation, exactly matching baseline_calculation/
# update_baselines_with_features.py's HIGHER_IS_CONFIDENT / from_m16 map:
# True => higher raw value = more confident of SUCCESS.
HIGHER_IS_CONFIDENT = {
    "m1_verbalized_conf": True,
    "m2_action_logprob_mean": False,
    "m3_output_logprob_mean": False,
    "m4_cumulative_output_logprob_mean": False,
    "m5_action_entropy": False,
    "m6_cumulative_action_entropy": False,
    "m7_conf_cumul_mean": True,
    "m15_cumulative_action_logprob_mean": False,  # same orientation as m2 (mean neg-log-prob)
}
M1_M7_LABEL_COL = {
    "m1_verbalized_conf": "success", "m2_action_logprob_mean": "success",
    "m3_output_logprob_mean": "success", "m4_cumulative_output_logprob_mean": "success",
    "m5_action_entropy": "success_entropy", "m6_cumulative_action_entropy": "success_entropy",
    "m7_conf_cumul_mean": "success",
    "m15_cumulative_action_logprob_mean": "success",
}
BASE_FEATURE_COLS_M13 = [
    "m1_verbalized_conf", "m2_action_logprob_mean",
    "m3_output_logprob_mean", "m4_cumulative_output_logprob_mean",
    "m7_conf_cumul_mean",
    "m15_cumulative_action_logprob_mean",  # added 2026-08-02, matches
    # baseline_calculation/stacked_logistic_baseline.py's BASE_FEATURE_COLS
]  # 6 features -- far fewer than Ours (49) or M11 (48); see README for why
   # this makes M13 a feature-count-confounded control, not a clean
   # "supervision alone" ablation


def embedded_thr_base(bench: str, model: str) -> Path:
    return PR / "data" / bench / model / "embedded" / "uq_analysis_change_qwen" / "thr_0p3"


def build_category_map(bench: str, model: str) -> dict[str, str]:
    fm = pd.read_csv(embedded_thr_base(bench, model) / "feature_matrix.csv")
    ids = fm["task_id"].unique()
    cat_map: dict[str, str] = {}
    # The public release ships the category map as
    # data/<bench>/<model>/category_map.json, derived at packaging time from
    # the per-task step_0_hlvm_uq.json (webarena page_url) / raw result.json
    # (mind2web task text). Those per-task sources are not redistributed, so
    # without this branch every task would silently resolve to "unknown".
    derived = PR / "data" / bench / model / "category_map.json"
    if derived.is_file():
        saved = json.loads(derived.read_text())
        return {str(t): saved.get(str(t), "unknown") for t in ids}
    if bench == "webarena":
        base = embedded_thr_base(bench, model)
        for tid in ids:
            cat_map[str(tid)] = get_webarena_category(int(tid), base)
    else:
        raw_plain = PR / "data" / "mind2web" / model / "raw" / "plain"
        for tid in ids:
            cat_map[str(tid)] = get_mind2web_category(int(tid), raw_plain)
    return cat_map


# ── Ours (L2): load feature_matrix + keystep-relabel (ours role) ───────────

def load_ours_source(bench: str, model: str, cat_map: dict[str, str]) -> pd.DataFrame:
    fm_path = embedded_thr_base(bench, model) / "feature_matrix.csv"
    df = pd.read_csv(fm_path)
    df["source"] = f"{bench}_{model}"
    df["global_task_id"] = df["source"] + "::" + df["task_id"].astype(str)
    df["category"] = df["task_id"].astype(str).map(cat_map).fillna("unknown")

    annot = load_keystep_map_entropy_source(f"{bench}/{model}", str(PR / "keystep_labels" / bench))
    df["label"] = df["outcome"].copy()
    for tid, ks in annot.items():
        if ks is None:
            continue
        mask = df["task_id"].astype(str) == str(tid)
        if not mask.any():
            continue
        if int(df.loc[mask, "outcome"].iloc[0]) == 0:
            continue
        df.loc[mask & (df["step_k"] < ks), "label"] = 0
        df.loc[mask & (df["step_k"] >= ks), "label"] = 1
    return df


def load_all_ours() -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    frames = []
    cat_maps = {}
    for bench, model in COMBOS:
        cat_map = build_category_map(bench, model)
        cat_maps[f"{bench}_{model}"] = cat_map
        df = load_ours_source(bench, model, cat_map)
        n_tasks = df["task_id"].nunique()
        print(f"  ours  {bench}/{model:10s}  rows={len(df):5d}  tasks={n_tasks:4d}  "
              f"relabeled_from_annot={len(cat_map)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True), cat_maps


def run_holdout_ours(df: pd.DataFrame, category: str, feat_cols: list[str]) -> dict:
    avail = [c for c in feat_cols if c in df.columns]
    train_df = df[df["category"] != category].copy()
    test_df = df[df["category"] == category].copy()
    if train_df.empty or test_df.empty or train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
        return {}

    medians = safe_medians(train_df, avail)
    X_train = train_df[avail].fillna(medians).fillna(0.0).values
    y_train = train_df["label"].values.astype(int)
    X_test = test_df[avail].fillna(medians).fillna(0.0).values
    y_test = test_df["label"].values.astype(int)

    step_counts = train_df.groupby("global_task_id")["step_k"].transform("count")
    sample_weight = (1.0 / step_counts).values
    train_task_ids = train_df["global_task_id"].values

    clf = fit_model("Logistic_L2", X_train, y_train, sample_weight, task_ids=train_task_ids)
    y_prob = clf.predict_proba(X_test)[:, 1]   # P(label=1) = P(failure), native

    return finalize_metrics(y_test, y_prob, n_train_tasks=train_df["global_task_id"].nunique(),
                             n_test_tasks=test_df["global_task_id"].nunique())


def run_indomain_percat(df: pd.DataFrame, category: str, feat_cols: list[str]) -> dict:
    """Matched-population control for the cross-category number above: same
    test population (this category's own pooled tasks, across all 10
    combos), same model/protocol (Logistic_L2, task-length weighting,
    fit_model's inner C search), but a standard 5-fold CV WITHIN the
    category instead of holding it out. Answers "how much does the
    category shift itself cost", isolated from the cross-category run's
    much larger (~pools 4 other categories) training set -- a plain
    whole-combo in-domain number is NOT the right comparison here, since
    it differs in both training-set size AND test population from the
    cross-category run.

    shuffle=True, random_state=42 (2026-08-01, was shuffle=False): this is
    Ours' own in-domain control, so it follows the same outer-CV protocol
    change as train_risk_model_temp_sca.py::run_cv() -- see that function's
    docstring for the full rationale (matching M11/M13's own already-shuffled
    protocol, verified empirically to not be a "pick whichever favors Ours"
    change). M11/M13's OWN in-domain-percat controls
    (sizematched_control/run_sizematched_control.py) are untouched here --
    out of scope for this change, tracked separately."""
    avail = [c for c in feat_cols if c in df.columns]
    sub = df[df["category"] == category].copy()
    if sub.empty:
        return {}
    task_ids = sub["global_task_id"].unique()
    task_outcome = sub.groupby("global_task_id")["outcome"].first().reindex(task_ids).values
    if len(np.unique(task_outcome)) < 2 or len(task_ids) < 10:
        return {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y, all_p, n_train_tasks = [], [], []
    for tr_idx, te_idx in skf.split(task_ids, task_outcome):
        tr_tasks, te_tasks = set(task_ids[tr_idx]), set(task_ids[te_idx])
        train_df = sub[sub["global_task_id"].isin(tr_tasks)]
        test_df = sub[sub["global_task_id"].isin(te_tasks)]
        if train_df.empty or test_df.empty or train_df["label"].nunique() < 2:
            continue
        medians = safe_medians(train_df, avail)
        X_train = train_df[avail].fillna(medians).fillna(0.0).values
        y_train = train_df["label"].values.astype(int)
        X_test = test_df[avail].fillna(medians).fillna(0.0).values
        y_test = test_df["label"].values.astype(int)
        step_counts = train_df.groupby("global_task_id")["step_k"].transform("count")
        sample_weight = (1.0 / step_counts).values
        clf = fit_model("Logistic_L2", X_train, y_train, sample_weight,
                         task_ids=train_df["global_task_id"].values)
        all_y.append(y_test)
        all_p.append(clf.predict_proba(X_test)[:, 1])
        n_train_tasks.append(train_df["global_task_id"].nunique())
    if not all_y:
        return {}
    m = finalize_metrics(np.concatenate(all_y), np.concatenate(all_p))
    if m:
        m["n_train_tasks"] = round(float(np.mean(n_train_tasks)))
        m["n_test_tasks"] = len(task_ids)
    return m


# ── M1-M7: no retraining, direct per-category evaluation ───────────────────

def load_all_per_step_scores(cat_maps: dict[str, dict[str, str]]) -> pd.DataFrame:
    frames = []
    for bench, model in COMBOS:
        p = REPRO / "baseline" / bench / model / "results" / "per_step_scores.csv"
        if not p.is_file():
            print(f"  [SKIP] missing per_step_scores.csv: {p}")
            continue
        df = pd.read_csv(p)
        df = add_m2_cumulative(df).rename(
            columns={"m2c_cumulative_action_logprob_mean": "m15_cumulative_action_logprob_mean"})
        source = f"{bench}_{model}"
        df["source"] = source
        cat_map = cat_maps[source]
        id_col = "task_index" if (bench == "mind2web" and "task_index" in df.columns) else "task_id"
        df["category"] = df[id_col].astype(str).map(cat_map).fillna("unknown")
        df["global_task_group"] = source + "::" + df[id_col].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def eval_m1_m7_holdout(pss: pd.DataFrame, category: str) -> list[dict]:
    records = []
    for col, label_col in M1_M7_LABEL_COL.items():
        if col not in pss.columns:
            continue
        sub = pss[[label_col, col, "category"]].dropna()
        train = sub[sub["category"] != category]
        test = sub[sub["category"] == category]
        if len(test) < 4 or test[label_col].nunique() < 2:
            continue

        def _oriented(s):
            raw = s[col].values.astype(float)
            return raw if HIGHER_IS_CONFIDENT[col] else -raw

        train_conf = _oriented(train)
        test_conf = _oriented(test)
        if len(np.unique(train_conf)) < 2:
            continue
        # Held-out Platt scaling (2026-08-31, replacing MinMaxScaler -- see
        # module docstring): 1-D logistic fit on the pooled
        # other-4-categories rows, predicted on the held-out category. Same
        # fitted-calibrator family as the main table's calibrate_platt().
        y_train_success = train[label_col].values.astype(int)
        if len(np.unique(y_train_success)) < 2:
            continue
        platt = LogisticRegression(solver="lbfgs", max_iter=1000)
        platt.fit(train_conf.reshape(-1, 1), y_train_success)
        p_success_platt = np.clip(
            platt.predict_proba(test_conf.reshape(-1, 1))[:, 1], 1e-7, 1 - 1e-7)
        # Ranking (AUROC/e-AURC) stays on the raw oriented score; only Brier
        # consumes the Platt probabilities (main-table convention).
        rank_p_fail = -test_conf
        y_success = test[label_col].values.astype(int)
        y_fail = 1 - y_success

        m = finalize_metrics(y_fail, rank_p_fail,
                             p_fail_brier=1.0 - p_success_platt)
        if not m:
            continue
        m["method"] = col
        m["category"] = category
        records.append(m)
    return records


# ── M11 (HTC full/L2): pooled logprob features, leave-one-category-out ─────

def load_all_htc_feat(cat_maps: dict[str, dict[str, str]]) -> pd.DataFrame:
    frames = []
    for bench, model in COMBOS:
        p = REPRO / "baseline" / bench / model / "results" / "per_step_scores.csv"
        if not p.is_file():
            continue
        per_step_df = pd.read_csv(p)
        source = f"{bench}/{model}"
        # Public release ships the derived HTC feature table; fall back to raw.
        # Parquet, not CSV: it round-trips float64 and the string task_id /
        # task_group columns exactly, so the derived table is bit-identical to
        # what build_htc_step_dataset() returns from the raw rollout.
        derived = PR / "data" / bench / model / "htc_step_features.parquet"
        if derived.is_file():
            feat_df = pd.read_parquet(derived)
        else:
            verbalized_dir = PR / "data" / bench / model / "raw" / "verbalized"
            feat_df = build_htc_step_dataset(source, per_step_df,
                                             verbalized_dir_override=str(verbalized_dir))
        if feat_df.empty:
            print(f"  [SKIP] HTC unavailable: {source}")
            continue
        src_tag = f"{bench}_{model}"
        id_col = "task_index" if "task_index" in feat_df.columns else "task_id"
        cat_map = cat_maps[src_tag]
        feat_df["category"] = feat_df[id_col].astype(str).map(cat_map).fillna("unknown")
        feat_df["global_task_group"] = src_tag + "::" + feat_df["task_group"].astype(str)
        feat_df["source"] = src_tag
        n_tasks = feat_df["global_task_group"].nunique()
        print(f"  htc   {bench}/{model:10s}  rows={len(feat_df):5d}  tasks={n_tasks:4d}")
        frames.append(feat_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_holdout_m11(feat_df: pd.DataFrame, category: str) -> dict:
    feat_cols = [c for c in feat_df.columns if c.startswith("f") and c[1:].isdigit()]
    train_df = feat_df[feat_df["category"] != category].copy()
    test_df = feat_df[feat_df["category"] == category].copy()
    if train_df.empty or test_df.empty:
        return {}
    y_tr = train_df["success"].values.astype(int)  # 1 = success (baseline role)
    y_te = test_df["success"].values.astype(int)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[feat_cols].values)
    X_te = scaler.transform(test_df[feat_cols].values)
    groups = train_df["global_task_group"].values

    best_C = best_C_for_fold(X_tr, y_tr, groups)
    clf = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs", max_iter=1000, random_state=42)
    step_counts = train_df.groupby("global_task_group")["step"].transform("count")
    clf.fit(X_tr, y_tr, sample_weight=(1.0 / step_counts).values)
    p_success = clf.predict_proba(X_te)[:, 1]
    y_fail = 1 - y_te
    m = finalize_metrics(y_fail, 1.0 - p_success)
    if m:
        m["method"], m["category"] = "m11_htc_full_step", category
    return m


# ── M13 (stacked HTC_L2 over M1-M4/M7) ──────────────────────────────────────

def run_holdout_m13(pss: pd.DataFrame, category: str) -> dict:
    feat_cols = [c for c in BASE_FEATURE_COLS_M13 if c in pss.columns and pss[c].notna().sum() > 0]
    if len(feat_cols) < 2:
        return {}
    sub = pss[pss[feat_cols].notna().any(axis=1)].copy()
    train_df = sub[sub["category"] != category]
    test_df = sub[sub["category"] == category]
    if train_df.empty or test_df.empty:
        return {}
    y_tr = train_df["success"].values.astype(int)
    y_te = test_df["success"].values.astype(int)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {}

    medians = train_df[feat_cols].median()
    X_tr = train_df[feat_cols].fillna(medians).fillna(0.0).values
    X_te = test_df[feat_cols].fillna(medians).fillna(0.0).values
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)
    groups = train_df["global_task_group"].values

    best_C = best_C_for_fold(X_trs, y_tr, groups)
    clf = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_trs, y_tr)
    p_success = clf.predict_proba(X_tes)[:, 1]
    y_fail = 1 - y_te
    m = finalize_metrics(y_fail, 1.0 - p_success)
    if m:
        m["method"], m["category"] = "m13_stacked_htc_l2", category
    return m


# ── Shared metric helper (canonical y_fail / P(fail) convention) ───────────

def finalize_metrics(y_fail, p_fail, n_train_tasks=None, n_test_tasks=None,
                     p_fail_brier=None) -> dict:
    """y_fail: 1=failure. p_fail: P(failure), or any higher=more-predicted-
    failure ranking score -- AUROC/AURC/e-AURC only use its ranks. Every
    caller converts to this convention first (Ours' own "label" is already
    1=failure; M1-M13 are flipped via 1-success / 1-p_success before
    calling here).
    p_fail_brier (2026-08-31): optional separate calibrated P(failure) used
    for the Brier score only. eval_m1_m7_holdout() passes it (held-out
    Platt probabilities) while ranking on the raw oriented score; when
    None, Brier falls back to p_fail (every other caller, whose p_fail is
    already a genuine model probability)."""
    yt = np.asarray(y_fail).astype(int)
    yp = np.asarray(p_fail).astype(float)
    yb = np.asarray(p_fail_brier).astype(float) if p_fail_brier is not None else yp
    valid = ~np.isnan(yp) & ~np.isnan(yb)
    yt, yp, yb = yt[valid], yp[valid], yb[valid]
    if len(yt) < 4 or len(np.unique(yt)) < 2:
        return {}
    auroc = roc_auc_score(yt, yp)
    aurc_val = aurc(yt, yp)[0]
    aurc_oracle_val = aurc_oracle(yt)
    brier = brier_score_loss(yt, np.clip(yb, 1e-7, 1 - 1e-7))
    out = {
        "n_samples": int(len(yt)),
        "auroc": round(float(auroc), 4),
        "aurc": round(float(aurc_val), 4),
        "e_aurc": round(float(aurc_val - aurc_oracle_val), 4),
        "brier": round(float(brier), 4),
    }
    if n_train_tasks is not None:
        out["n_train_tasks"] = n_train_tasks
    if n_test_tasks is not None:
        out["n_test_tasks"] = n_test_tasks
    return out


# ── In-domain reference: pull Ours' already-computed standard CV numbers ───

def load_in_domain_reference() -> pd.DataFrame:
    rows = []
    for bench, model in COMBOS:
        pattern = REPRO / "ours" / f"{bench}_{model}" / "thr_0p3" / "train_risk_model_temp_sca"
        runs = sorted(pattern.glob("*/summary_table.csv"))
        if not runs:
            print(f"  [SKIP] no Ours summary_table.csv: {bench}/{model}")
            continue
        df = pd.read_csv(runs[-1])
        df = df[df["metric"].isin(["auroc", "e_aurc", "brier"])]
        df["source"] = f"{bench}_{model}"
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════
# Figure -- 3-panel AUROC/e-AURC/Brier vs. held-out category, styled after
# results_total/generalization/exp6_cross_website/results/
# rf_generalization_figure.png: two bold, value-annotated "hero" lines
# (cross-category Ours in red, matched in-domain Ours in blue-dashed) drawn
# over a muted cloud of every other baseline, legend in one row below.
# ═════════════════════════════════════════════════════════════════════════

CAT_ORDER = ["devtools", "ecommerce", "health_services", "social_entertain", "travel_navigate"]
CAT_LABELS = ["Dev Tools", "E-Commerce", "Health Svcs", "Social/Entertain.", "Travel/Navigate"]

# thin baseline lines: distinct muted hue + marker per method (M12 dropped)
BASELINE_STYLE = {
    # 2026-08-01 (design decision): the 3 worst-performing baselines by mean
    # cross-category AUROC (m3=0.458, m4=0.475, m2=0.583 -- see
    # results/cross_category_summary.csv) dropped from the figure entirely,
    # not just de-emphasized -- m1/m5/m6/m7 remain as the muted cloud.
    # 2026-08-02: m15 (cumulative action-logprob, the M2->M15 fill-the-gap
    # feature) added -- at 0.628 mean AUROC it's the SECOND-best of the
    # whole M1-M7(+M15) family (only m6 at 0.637 is higher), nowhere near
    # the dropped-3 threshold, so it's kept in the cloud rather than
    # excluded like its at-k parent m2 was.
    "m1_verbalized_conf":                dict(color="#8fb996", marker="o", ls=":",  label="Verbalized Conf."),
    "m5_action_entropy":                 dict(color="#c9986b", marker="P", ls=":",  label="Action Entropy"),
    "m6_cumulative_action_entropy":      dict(color="#7a9d4f", marker="D", ls="-.", label="Cumul. Action Entropy"),
    "m7_conf_cumul_mean":                dict(color="#a05b95", marker="X", ls=":",  label="Conf. Cumul. Mean"),
    "m15_cumulative_action_logprob_mean": dict(color="#4a90d9", marker="^", ls="-.", label="Cumul. Action Logprob"),
    "m13_stacked_htc_l2":                dict(color="#7a746b", marker="*", ls="-.", label="Stacked_L2"),
}


def make_figure(results_df: pd.DataFrame, out_path: Path) -> None:
    metrics = [("auroc", "AUROC ↑"), ("e_aurc", "e-AURC ↓"), ("brier", "Brier ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))

    def series_for(method: str, metric: str) -> np.ndarray | None:
        sub = results_df[results_df.method == method].set_index("category")
        if sub.empty:
            return None
        vals = [sub.loc[c, metric] if c in sub.index else np.nan for c in CAT_ORDER]
        return np.array(vals, dtype=float)

    for ax, (mkey, mtitle) in zip(axes, metrics):
        for method, style in BASELINE_STYLE.items():
            vals = series_for(method, mkey)
            if vals is None or np.all(np.isnan(vals)):
                continue
            ax.plot(range(5), vals, color=style["color"], marker=style["marker"],
                    linestyle=style["ls"], linewidth=1.4, markersize=6, alpha=0.85,
                    label=style["label"])

        m11 = series_for("m11_htc_full_step", mkey)
        if m11 is not None:
            ax.plot(range(5), m11, color="#e07a1e", marker="o", linestyle="-",
                     linewidth=2.0, markersize=6.5, alpha=0.9, label="HTC", zorder=3)

        indomain = series_for("ours_l2_all_indomain_percat", mkey)
        if indomain is not None:
            ax.plot(range(5), indomain, color="#2a78d6", marker="s", linestyle="--",
                     linewidth=2.6, markersize=7, label="In-Domain Ours (matched pop.)", zorder=4)

        macro_only = series_for("ours_l2_macro_only", mkey)
        if macro_only is not None:
            ax.plot(range(5), macro_only, color="#f0876e", marker="^", linestyle="-.",
                     linewidth=2.0, markersize=6.5, alpha=0.9,
                     label="Cross-Category Ours — macro only", zorder=3)

        uq_only = series_for("ours_l2_uq_only", mkey)
        if uq_only is not None:
            ax.plot(range(5), uq_only, color="#8f2a3a", marker="v", linestyle=":",
                     linewidth=2.0, markersize=6.5, alpha=0.9,
                     label="Cross-Category Ours — micro/UQ only", zorder=3)

        cross = series_for("ours_l2_all", mkey)
        if cross is not None:
            ax.plot(range(5), cross, color="#d1372b", marker="D", linestyle="-",
                     linewidth=3.0, markersize=8, label="Cross-Category Ours - M&M", zorder=5)
            span = np.nanmax(cross) - np.nanmin(cross) if np.nanmax(cross) != np.nanmin(cross) else 0.05
            for x, v in enumerate(cross):
                if np.isnan(v):
                    continue
                offset = span * 0.14 if mkey == "auroc" else -span * 0.16
                va = "bottom" if mkey == "auroc" else "top"
                ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                            xytext=(0, 10 if mkey == "auroc" else -10), ha="center", va=va,
                            fontsize=10.5, fontweight="bold", color="#a82418")

        ax.set_title(mtitle, fontsize=15, fontweight="bold")
        ax.set_xticks(range(5))
        ax.set_xticklabels(CAT_LABELS, rotation=18, ha="right", fontsize=10.5)
        ax.set_xlabel("Website Category", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    # hero lines first (Cross-Category Ours variants, M11, In-Domain
    # matched), then the muted baseline cloud, in encounter order
    rank = {"Cross-Category Ours - M&M": 0, "Cross-Category Ours — macro only": 1,
            "Cross-Category Ours — micro/UQ only": 2, "HTC": 3, "In-Domain": 4}
    order = sorted(range(len(labels)),
                    key=lambda i: next((v for k, v in rank.items() if k in labels[i]), 5))
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=7, fontsize=9.5, frameon=True,
               bbox_to_anchor=(0.5, -0.1), title="Method")

    # 2026-08-01 (design decision): title/subtitle removed entirely (were
    # "Cross-Category Ours (L2) Generalization vs. Baselines" +
    # "(website_generalization · 5 website categories, M12 excluded, all 3
    # Ours feature sets shown)") -- rect's top bound loosened from 0.95 to
    # 1 since there's no title reserving space above the axes anymore.
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Loading Ours (feature_matrix.csv, all 10 combos) + category maps...")
    ours_df, cat_maps = load_all_ours()

    print("\nCategory distribution (tasks, across all 10 combos):")
    cat_counts = ours_df.drop_duplicates("global_task_id")["category"].value_counts()
    print(cat_counts.to_string())

    cat_rows = []
    for src in cat_maps:
        sub = ours_df[ours_df.source == src].drop_duplicates("global_task_id")
        for cat in WEBSITE_CATEGORIES + ["unknown", "misc"]:
            n = (sub["category"] == cat).sum()
            if n > 0:
                cat_rows.append({"source": src, "category": cat, "n_tasks": int(n)})
    pd.DataFrame(cat_rows).to_csv(out_dir / "category_stats.csv", index=False)

    print("\n" + "=" * 70)
    print("Loading per_step_scores.csv (M1-M7 + M15 raw scores, all 10 combos)...")
    pss = load_all_per_step_scores(cat_maps)

    print("\n" + "=" * 70)
    print("Loading HTC feature dataset (M11 logprob features)...")
    htc_df = load_all_htc_feat(cat_maps)

    records = []

    print("\n" + "=" * 70)
    print("Cross-category Ours (Logistic_L2)...")
    for fs_name, feat_cols in FEATURE_SETS.items():
        for category in WEBSITE_CATEGORIES:
            res = run_holdout_ours(ours_df, category, feat_cols)
            if not res:
                print(f"  [SKIP] ours/{fs_name}/{category}: insufficient data")
                continue
            res["method"] = f"ours_l2_{fs_name}"
            res["category"] = category
            records.append(res)
            print(f"  ours/{fs_name:12s} held_out={category:18s} "
                  f"auroc={res['auroc']:.4f} e_aurc={res['e_aurc']:.4f} brier={res['brier']:.4f}")

    print("\n" + "=" * 70)
    print("In-domain-per-category Ours (matched population, 5-fold CV within category)...")
    for fs_name, feat_cols in FEATURE_SETS.items():
        for category in WEBSITE_CATEGORIES:
            res = run_indomain_percat(ours_df, category, feat_cols)
            if not res:
                print(f"  [SKIP] indomain_percat/{fs_name}/{category}: insufficient data")
                continue
            res["method"] = f"ours_l2_{fs_name}_indomain_percat"
            res["category"] = category
            records.append(res)
            print(f"  ours/{fs_name:12s} within={category:18s} "
                  f"auroc={res['auroc']:.4f} e_aurc={res['e_aurc']:.4f} brier={res['brier']:.4f} "
                  f"n_train_tasks={res['n_train_tasks']}")

    print("\n" + "=" * 70)
    print("Cross-domain M1-M7 (no retraining, per-category evaluation)...")
    for category in WEBSITE_CATEGORIES:
        recs = eval_m1_m7_holdout(pss, category)
        records.extend(recs)
        for r in recs:
            print(f"  {r['method']:35s} held_out={category:18s} "
                  f"auroc={r['auroc']:.4f} e_aurc={r['e_aurc']:.4f} brier={r['brier']:.4f}")

    if not htc_df.empty:
        print("\n" + "=" * 70)
        print("Cross-domain M11 (HTC full/L2, leave-one-category-out retrain)...")
        for category in WEBSITE_CATEGORIES:
            m = run_holdout_m11(htc_df, category)
            if not m:
                print(f"  [SKIP] m11/{category}: insufficient data")
                continue
            records.append(m)
            print(f"  {m['method']:25s} held_out={category:18s} "
                  f"auroc={m['auroc']:.4f} e_aurc={m['e_aurc']:.4f} brier={m['brier']:.4f}")
    else:
        print("\n[SKIP] M11: HTC feature dataset empty (no logprobs)")

    print("\n" + "=" * 70)
    print("Cross-domain M13 (stacked HTC_L2, leave-one-category-out retrain)...")
    for category in WEBSITE_CATEGORIES:
        m = run_holdout_m13(pss, category)
        if not m:
            print(f"  [SKIP] m13/{category}: insufficient data")
            continue
        records.append(m)
        print(f"  {m['method']:25s} held_out={category:18s} "
              f"auroc={m['auroc']:.4f} e_aurc={m['e_aurc']:.4f} brier={m['brier']:.4f}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(out_dir / "cross_category_results.csv", index=False)

    summary = results_df.groupby("method")[["auroc", "e_aurc", "brier"]].agg(["mean", "std", "count"])
    summary.to_csv(out_dir / "cross_category_summary.csv")
    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std across 5 held-out categories)")
    print("=" * 70)
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("In-domain reference (Ours L2, whole-combo standard CV)...")
    in_domain = load_in_domain_reference()
    in_domain.to_csv(out_dir / "in_domain_reference.csv", index=False)
    in_domain_summary = in_domain.groupby(["feature_set", "metric"])["mean"].agg(["mean", "std"])
    print(in_domain_summary.to_string())

    print("\n" + "=" * 70)
    print("Rendering figure...")
    fig_path = out_dir / "website_generalization_figure.png"
    make_figure(results_df, fig_path)
    print(f"  saved -> {fig_path}")

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
