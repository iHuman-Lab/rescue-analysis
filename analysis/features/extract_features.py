import pandas as pd

from analysis.features.conventions import (
    DEFAULT_OFFSCREEN_LABEL,
    DEFAULT_PUPIL_COL,
    best_suffix,
    detect_category,
    subject_label,
)

# ---------------------------------------------------------------------------
# Feature extractors (operate on in-memory DataFrames)
# ---------------------------------------------------------------------------


def extract_fixation_features(
    fix_df: pd.DataFrame, fix_aoi_df: pd.DataFrame, aois: list[dict]
) -> dict:
    features = {
        "n_fixations": len(fix_df),
    }
    if not fix_df.empty:
        features["mean_fixation_dur_ms"] = float(fix_df["duration_ms"].mean())
        features["total_fixation_dur_ms"] = float(fix_df["duration_ms"].sum())
    else:
        features["mean_fixation_dur_ms"] = 0.0
        features["total_fixation_dur_ms"] = 0.0

    labels = [a["name"] for a in aois] + [DEFAULT_OFFSCREEN_LABEL]
    for aoi in labels:
        features[f"{aoi}_pct_dur"] = 0.0
        features[f"n_fixations_{aoi}"] = 0

    if not fix_aoi_df.empty and "aoi" in fix_aoi_df.columns:
        aoi_dur = fix_aoi_df.groupby("aoi")["duration_ms"].sum()
        aoi_counts = fix_aoi_df["aoi"].value_counts()
        total = float(aoi_dur.sum())
        if total > 0:
            for aoi in labels:
                features[f"{aoi}_pct_dur"] = float(aoi_dur.get(aoi, 0.0) / total)
                features[f"n_fixations_{aoi}"] = int(aoi_counts.get(aoi, 0))
    return features


def extract_saccade_features(sac_df: pd.DataFrame) -> dict:
    if sac_df.empty:
        return {
            "n_saccades": 0,
            "mean_saccade_dur_ms": 0.0,
            "mean_saccade_amp_px": 0.0,
            "saccades_total_duration_ms": 0.0,
        }
    return {
        "n_saccades": len(sac_df),
        "mean_saccade_dur_ms": float(sac_df["duration_ms"].mean()),
        "mean_saccade_amp_px": float(sac_df["amplitude"].mean()),
        "saccades_total_duration_ms": float(sac_df["duration_ms"].sum()),
    }


def extract_transition_features(trans_df: pd.DataFrame, aois: list[dict]) -> dict:
    labels = [a["name"] for a in aois]
    features = {f"transitions_{src}_{dst}": 0 for src in labels for dst in labels}

    if not trans_df.empty:
        for src in trans_df.index:
            for dst in trans_df.columns:
                key = f"transitions_{src}_{dst}"
                if key in features:
                    features[key] = int(trans_df.loc[src, dst])
    return features


def extract_pupil_features(eye_df: pd.DataFrame, eye_cfg: dict) -> dict:
    if eye_df.empty:
        return {"std_pupil_diam": 0.0}
    pupil_col = eye_cfg.get("pupil_col", DEFAULT_PUPIL_COL)
    if pupil_col not in eye_df.columns:
        return {"std_pupil_diam": 0.0}
    series = pd.to_numeric(eye_df[pupil_col], errors="coerce")
    missing_val = eye_cfg.get("missing", 0.0)
    series = series.replace(missing_val, pd.NA).dropna()
    if series.empty:
        return {"std_pupil_diam": 0.0}
    return {"std_pupil_diam": float(series.std())}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

from .eye_features import extract_eyetracking_features
from .game_features import extract_game_features


def extract_features(cfg: dict) -> pd.DataFrame:
    """Extract per-trial features for _best trials.

    Args:
        cfg:         Full config dict.
        preloaded:   {subject_id: {trial_id: {"game": df, "eyetracker": df}}}
        eyetracking: Output of run_eyetracking_features() —
                     {"fixations": {sid: {trial_id: {"fixations": df, ...}}},
                      "saccades":  {sid: {trial_id: {"saccades":  df, ...}}},
                      "aoi":       {sid: {trial_id: {"fix_aoi": df, "transitions": df, ...}}}}

    Returns:
        DataFrame with one row per best trial.
    """
    subjects = [str(s) for s in cfg.get("sub", [])]
    trials = cfg["trails"]
    expertise = cfg.get("expertise", {})

    # Load the .h5 file which you saved from previous step
    data = None

    for sid in subjects:
        for trial_id in trials:
            data[sid][trial_id][runs]  # Look how to loop over runs

            game_data = data[sid][trial_id][runs]["game"]
            eye_data = data[sid][trial_id][runs]["eye_tracking"]

            eye_features = extract_eyetracking_features(eye_data, cfg)
            game_features = extract_game_features(game_data, cfg)

            row = {
                "participant": subject_label(sid, cfg),
                "trial": trial_id,
                "category": category,
                "expertise": expertise.get(sid, "unknown"),
            }

            print(
                f"  {sid}  {trial_id:35s}  victims={row.get('saved_victims', '?')}  "
                f"fixations={row.get('n_fixations', '?')}  saccades={row.get('n_saccades', '?')}"
            )

    df = pd.DataFrame(rows)
    return df
