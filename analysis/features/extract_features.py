import pandas as pd

from .eye_features import extract_eyetracking_features
from .game_features import extract_game_features

# ---------------------------------------------------------------------------
# Feature extractors (operate on in-memory DataFrames)
# ---------------------------------------------------------------------------


def extract_fixation_features(
    fix_df: pd.DataFrame,
    fix_aoi_df: pd.DataFrame,
    aois: list[dict],
    offscreen_label: str,
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

    labels = [a["name"] for a in aois] + [offscreen_label]
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
    pupil_col = eye_cfg.get("pupil_col")
    if not pupil_col or pupil_col not in eye_df.columns:
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


def extract_features(data: dict, cfg: dict) -> pd.DataFrame:
    """Extract per-trial features for all subjects, trials, and games.

    Args:
        data: Nested dict with structure data[sub][trial][game].
        cfg:  Full config dict.

    Returns:
        DataFrame with one row per game.
    """
    expertise = cfg.get("expertise", {})

    rows = []
    for sub in data:
        for trial in data[sub]:
            for game in data[sub][trial]:
                game_data = data[sub][trial][game].get("game")
                eye_data = data[sub][trial][game].get("eye_tracking")

                eye_features = extract_eyetracking_features(eye_data, cfg)
                game_features = extract_game_features(game_data)
                game_features.pop("victims_per_step", None)

                row = {
                    "participant": sub,
                    "trial": trial,
                    "game": game,
                    "expertise": expertise.get(sub, "unknown"),
                    **eye_features,
                    **game_features,
                }

                print(
                    f"  {sub}  {trial}  {game}  "
                    f"victims={row.get('saved_victims', '?')}  "
                    f"fixations={row.get('n_fixations', '?')}  "
                    f"saccades={row.get('n_saccades', '?')}"
                )

                rows.append(row)

    return pd.DataFrame(rows)
