import pandas as pd

from analysis.prepare_data.xdf import load_all_subjects
from analysis.features.conventions import (
    DEFAULT_OFFSCREEN_LABEL,
    strip_run_suffix,
)
from analysis.features.eyetracking_data import run_eyetracking

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_aoi_fixations(cfg: dict, eyetracking: dict | None = None,
                      preloaded: dict | None = None) -> dict:
    """Label fixations with AOIs and compute transition matrices."""
    if eyetracking is None:
        preloaded = load_all_subjects(cfg) if preloaded is None else preloaded
        eyetracking = run_eyetracking(cfg, preloaded=preloaded)

    aois          = cfg.get("aoi", [])

    if not aois:
        print("WARNING: no AOIs defined in config — all fixations will be labelled 'offscreen'")

    aoi_results: dict = {}
    for sid, trials in eyetracking.get("fixations", {}).items():
        print(f"  [SUB]  {sid} (AOI processing)")
        aoi_results[sid] = {}

        for trial_id, streams in trials.items():
            fix_df = streams.get("fixations", pd.DataFrame())

            if fix_df.empty:
                aoi_results[sid][trial_id] = {"fix_aoi": pd.DataFrame(), "transitions": pd.DataFrame()}
                continue

            fix_aoi = label_fixations(fix_df, aois)
            trans   = aoi_transition_matrix(fix_aoi, aois)

            aoi_results[sid][trial_id] = {"fix_aoi": fix_aoi, "transitions": trans}

    return aoi_results

# ---------------------------------------------------------------------------
# AOI helpers
# ---------------------------------------------------------------------------

def label_fixations(fix_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    """Vectorized AOI assignment based on bounding boxes."""
    df = fix_df.copy()
    df["aoi"] = DEFAULT_OFFSCREEN_LABEL
    for aoi in aois:
        mask = df["x"].between(aoi["x_min"], aoi["x_max"]) & df["y"].between(aoi["y_min"], aoi["y_max"])
        df.loc[mask, "aoi"] = aoi["name"]
    return df


def aoi_transition_matrix(fix_aoi_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    """Calculate transition counts between sequential AOI fixations."""
    labels   = [a["name"] for a in aois]
    matrix   = pd.DataFrame(0, index=labels, columns=labels)
    sequence = fix_aoi_df[fix_aoi_df["aoi"] != DEFAULT_OFFSCREEN_LABEL]["aoi"].tolist()

    for src, dst in zip(sequence[:-1], sequence[1:]):
        if src in matrix.index and dst in matrix.columns:
            matrix.loc[src, dst] += 1
    return matrix

def aggregate_transitions(aoi_results: dict, aois: list[dict], cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Aggregate transition matrices across all trials."""
    labels = [a["name"] for a in aois]
    if not labels:
        return pd.DataFrame(), {}

    total_trans = pd.DataFrame(0, index=labels, columns=labels)
    by_trial = {}

    for sid, trials in aoi_results.items():
        for trial_id, trial_data in trials.items():
            mat = trial_data["transitions"].reindex(index=labels, columns=labels, fill_value=0)
            total_trans += mat

            base = strip_run_suffix(trial_id, cfg)
            by_trial[base] = by_trial.get(base, pd.DataFrame(0, index=labels, columns=labels)) + mat

    return total_trans, by_trial
