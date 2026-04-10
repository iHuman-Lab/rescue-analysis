from pathlib import Path

import pandas as pd

from analysis.features.conventions import (
    DEFAULT_PUPIL_COL,
    best_suffix,
    detect_category,
    subject_label,
)
from analysis.features.eyetracking_features import run_eyetracking_features


# ---------------------------------------------------------------------------
# Feature extractors (operate on in-memory DataFrames)
# ---------------------------------------------------------------------------

def extract_fixation_features(fix_df: pd.DataFrame, fix_aoi_df: pd.DataFrame) -> dict:
    features = {
        "n_fixations":           len(fix_df),
        "mean_fixation_dur_ms":  fix_df["duration_ms"].mean() if not fix_df.empty else 0.0,
        "total_fixation_dur_ms": fix_df["duration_ms"].sum()  if not fix_df.empty else 0.0,
    }
    if not fix_aoi_df.empty and "aoi" in fix_aoi_df.columns:
        aoi_dur = fix_aoi_df.groupby("aoi")["duration_ms"].sum()
        total   = aoi_dur.sum()
        for aoi, dur in aoi_dur.items():
            features[f"{aoi}_pct_dur"]      = dur / total if total > 0 else 0.0
            features[f"n_fixations_{aoi}"]  = int((fix_aoi_df["aoi"] == aoi).sum())
    return features


def extract_saccade_features(sac_df: pd.DataFrame) -> dict:
    if sac_df.empty:
        return {}
    return {
        "n_saccades":                 len(sac_df),
        "mean_saccade_dur_ms":        sac_df["duration_ms"].mean(),
        "mean_saccade_amp_px":        sac_df["amplitude"].mean(),
        "saccades_total_duration_ms": sac_df["duration_ms"].sum(),
    }


def extract_transition_features(trans_df: pd.DataFrame) -> dict:
    features = {}
    for src in trans_df.index:
        for dst in trans_df.columns:
            features[f"transitions_{src}_{dst}"] = trans_df.loc[src, dst]
    return features


def extract_pupil_features(eye_df: pd.DataFrame, eye_cfg: dict) -> dict:
    pupil_col = eye_cfg.get("pupil_col", DEFAULT_PUPIL_COL)
    if pupil_col not in eye_df.columns:
        return {}
    series = pd.to_numeric(eye_df[pupil_col], errors="coerce")
    missing_val = eye_cfg.get("missing", 0.0)
    series = series.replace(missing_val, pd.NA).dropna()
    if series.empty:
        return {}
    return {"std_pupil_diam": float(series.std())}


def extract_game_features(game_df: pd.DataFrame) -> dict:
    reward = pd.to_numeric(game_df["reward"], errors="coerce") if "reward" in game_df.columns else pd.Series(dtype=float)
    features = {
        "n_actions":    int(game_df["action"].notna().sum()),
        "n_llm_calls":  int(game_df["llm_response"].notna().sum()) if "llm_response" in game_df.columns else 0,
        "saved_victims": int(game_df["saved_victims"].max()),
        "mean_reward":  float(reward.mean()) if not reward.empty else 0.0,
        "total_reward": float(reward.sum()) if not reward.empty else 0.0,
    }
    max_steps = game_df["step_count"].max()
    features["victims_per_step"] = (features["saved_victims"] / max_steps) if max_steps else 0.0
    return features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_extract_features(cfg: dict, preloaded: dict, eyetracking: dict,
                         processed_dir: Path) -> pd.DataFrame:
    """Extract per-trial features for _best trials and save best_features.csv.

    Args:
        cfg:         Full config dict.
        preloaded:   {subject_id: {trial_id: {"game": df, "eyetracker": df}}}
        eyetracking: Output of run_eyetracking_features() —
                     {"fixations": {sid: {trial_id: {"fixations": df, ...}}},
                      "saccades":  {sid: {trial_id: {"saccades":  df, ...}}},
                      "aoi":       {sid: {trial_id: {"fix_aoi": df, "transitions": df, ...}}}}
        processed_dir: Output directory for best_features.csv.

    Returns:
        DataFrame with one row per best trial.
    """
    subjects  = [str(s) for s in cfg.get("sub", [])]
    expertise = cfg.get("expertise", {})

    eye_cfg    = cfg.get("eyetracker", {})
    fix_by_sub = eyetracking.get("fixations", {})
    sac_by_sub = eyetracking.get("saccades",  {})
    aoi_by_sub = eyetracking.get("aoi",       {})

    rows = []
    suffix = best_suffix(cfg)
    for sid in subjects:
        trials = preloaded.get(sid, {})
        for trial_id, streams in trials.items():
            if not trial_id.endswith(suffix):
                continue

            game_df = streams.get("game",        pd.DataFrame())
            eye_df  = streams.get("eyetracker",  pd.DataFrame())
            fix_res = fix_by_sub.get(sid, {}).get(trial_id, {})
            sac_res = sac_by_sub.get(sid, {}).get(trial_id, {})
            aoi_res = aoi_by_sub.get(sid, {}).get(trial_id, {})

            fix_df      = fix_res.get("fixations",   pd.DataFrame())
            sac_df      = sac_res.get("saccades",    pd.DataFrame())
            fix_aoi_df  = aoi_res.get("fix_aoi",     pd.DataFrame())
            trans_df    = aoi_res.get("transitions",  pd.DataFrame())

            category = detect_category(trial_id, cfg)

            row = {
                "participant": subject_label(sid, cfg),
                "trial":       trial_id,
                "category":    category,
                "expertise":   expertise.get(sid, "unknown"),
            }
            if not game_df.empty:
                row.update(extract_game_features(game_df))
            if not eye_df.empty:
                row.update(extract_pupil_features(eye_df, eye_cfg))
            row.update(extract_fixation_features(fix_df, fix_aoi_df))
            if not sac_df.empty:
                row.update(extract_saccade_features(sac_df))
            if not trans_df.empty:
                row.update(extract_transition_features(trans_df))

            rows.append(row)
            print(f"  {sid}  {trial_id:35s}  victims={row.get('saved_victims', '?')}  "
                  f"fixations={row.get('n_fixations', '?')}  saccades={row.get('n_saccades', '?')}")

    df = pd.DataFrame(rows)
    if not df.empty:
        out = processed_dir / "best_features.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nbest_features -> {out}")
    return df
