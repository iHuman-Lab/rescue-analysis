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

def extract_fixation_features(fix_df: pd.DataFrame, fix_aoi_df: pd.DataFrame, aois: list[dict]) -> dict:
    features = {
        "n_fixations":           len(fix_df),
    }
    if not fix_df.empty:
        features["mean_fixation_dur_ms"]  = float(fix_df["duration_ms"].mean())
        features["total_fixation_dur_ms"] = float(fix_df["duration_ms"].sum())
    else:
        features["mean_fixation_dur_ms"]  = 0.0
        features["total_fixation_dur_ms"] = 0.0

    labels = [a["name"] for a in aois] + [DEFAULT_OFFSCREEN_LABEL]
    for aoi in labels:
        features[f"{aoi}_pct_dur"] = 0.0
        features[f"n_fixations_{aoi}"] = 0

    if not fix_aoi_df.empty and "aoi" in fix_aoi_df.columns:
        aoi_dur = fix_aoi_df.groupby("aoi")["duration_ms"].sum()
        aoi_counts = fix_aoi_df["aoi"].value_counts()
        total   = float(aoi_dur.sum())
        if total > 0:
            for aoi in labels:
                features[f"{aoi}_pct_dur"]     = float(aoi_dur.get(aoi, 0.0) / total)
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
        "n_saccades":                 len(sac_df),
        "mean_saccade_dur_ms":        float(sac_df["duration_ms"].mean()),
        "mean_saccade_amp_px":        float(sac_df["amplitude"].mean()),
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


def _reward_by_unique_progress_state(game_df: pd.DataFrame) -> pd.Series:
    """Count reward once per step using the last nonzero reward in that step."""
    if "reward" not in game_df.columns:
        return pd.Series(dtype=float)

    reward_df = game_df.copy()
    reward_df["reward"] = pd.to_numeric(reward_df["reward"], errors="coerce")
    if "step_count" not in reward_df.columns:
        return reward_df["reward"].dropna()

    reward_df["step_count"] = pd.to_numeric(reward_df["step_count"], errors="coerce")
    reward_df = reward_df.loc[reward_df["step_count"].notna()].copy()
    if reward_df.empty:
        return pd.Series(dtype=float)

    def one_reward_per_step(step_rewards: pd.Series) -> float:
        nonzero = step_rewards[step_rewards.ne(0)].dropna()
        if not nonzero.empty:
            return float(nonzero.iloc[-1])
        last = step_rewards.dropna()
        return float(last.iloc[-1]) if not last.empty else 0.0

    return reward_df.groupby("step_count", sort=False)["reward"].apply(one_reward_per_step)


def extract_game_features(game_df: pd.DataFrame) -> dict:
    if game_df.empty:
        return {
            "n_actions": 0, "n_llm_calls": 0, "saved_victims": 0,
            "mean_reward": 0.0, "total_reward": 0.0, "cumulative_reward": 0.0, "victims_per_step": 0.0
        }
    reward = pd.to_numeric(game_df["reward"], errors="coerce") if "reward" in game_df.columns else pd.Series(dtype=float)
    deduped_reward = _reward_by_unique_progress_state(game_df)
    cumulative_reward = float(deduped_reward.sum()) if not deduped_reward.empty else 0.0
    features = {
        "n_actions":    int(game_df["action"].notna().sum()),
        "n_llm_calls":  int(game_df["llm_response"].notna().sum()) if "llm_response" in game_df.columns else 0,
        "saved_victims": int(game_df["saved_victims"].max()),
        "mean_reward":  float(reward.mean()) if not reward.empty else 0.0,
        "total_reward": cumulative_reward,
        "cumulative_reward": cumulative_reward,
    }
    max_steps = game_df["step_count"].max()
    features["victims_per_step"] = (features["saved_victims"] / max_steps) if max_steps else 0.0
    return features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_extract_features(cfg: dict, preloaded: dict, eyetracking: dict) -> pd.DataFrame:
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
    subjects  = [str(s) for s in cfg.get("sub", [])]
    expertise = cfg.get("expertise", {})

    eye_cfg    = cfg.get("eyetracker", {})
    fix_by_sub = eyetracking.get("fixations", {})
    sac_by_sub = eyetracking.get("saccades",  {})
    aoi_by_sub = eyetracking.get("aoi",       {})

    rows = []
    suffix = best_suffix(cfg)
    aois = cfg.get("aoi", [])
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
            row.update(extract_game_features(game_df))
            row.update(extract_pupil_features(eye_df, eye_cfg))
            row.update(extract_fixation_features(fix_df, fix_aoi_df, aois))
            row.update(extract_saccade_features(sac_df))
            row.update(extract_transition_features(trans_df, aois))

            rows.append(row)
            print(f"  {sid}  {trial_id:35s}  victims={row.get('saved_victims', '?')}  "
                  f"fixations={row.get('n_fixations', '?')}  saccades={row.get('n_saccades', '?')}")

    df = pd.DataFrame(rows)
    return df
