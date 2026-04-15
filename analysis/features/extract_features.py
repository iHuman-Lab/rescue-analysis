import pandas as pd

from .AOI_fixation import aoi_transition_matrix, label_fixations
from .eyetracking_data import run_eyetracking
from .game_features import extract_game_features


def extract_features(data: dict, cfg: dict) -> pd.DataFrame:
    """Extract per-trial features for all subjects and trials.

    Args:
        data: {sub: {trial_id: {run_num: {"game": DataFrame, "eye_tracking": DataFrame}}}}
        cfg:  Full config dict.

    Returns:
        DataFrame with one row per run.
    """
    aois = cfg.get("aoi", [])
    offscreen_label = cfg.get("analysis", {}).get("offscreen_label", "offscreen")
    eye_cfg = cfg.get("eyetracker", {})
    expertise = cfg.get("expertise", {})

    rows = []
    for sub, trials in data.items():
        for trial_id, runs in trials.items():
            for run_num, streams in runs.items():
                game_data = streams["game"]
                eye_data = streams["eye_tracking"]

                et = run_eyetracking(eye_data, cfg)
                fix_df = et["fixations"]
                sac_df = et["saccades"]
                fix_aoi = label_fixations(fix_df, aois)
                trans = aoi_transition_matrix(fix_aoi, aois)
                aoi_labels = [a["name"] for a in aois] + [offscreen_label]

                aoi_dur = fix_aoi.groupby("aoi")["duration_ms"].sum() if not fix_aoi.empty else pd.Series(dtype=float)
                aoi_counts = fix_aoi["aoi"].value_counts() if not fix_aoi.empty else pd.Series(dtype=int)
                total_dur = float(aoi_dur.sum())

                pupil_col = eye_cfg.get("pupil_col")
                if not eye_data.empty and pupil_col and pupil_col in eye_data.columns:
                    pupil = pd.to_numeric(eye_data[pupil_col], errors="coerce")
                    pupil = pupil.replace(eye_cfg.get("missing", 0.0), pd.NA).dropna()
                    std_pupil = float(pupil.std()) if not pupil.empty else None
                else:
                    std_pupil = None

                eye_features = {
                    "n_fixations": len(fix_df),
                    "mean_fixation_dur_ms": float(fix_df["duration_ms"].mean()) if not fix_df.empty else None,
                    "total_fixation_dur_ms": float(fix_df["duration_ms"].sum()) if not fix_df.empty else None,
                    "n_saccades": len(sac_df),
                    "mean_saccade_dur_ms": float(sac_df["duration_ms"].mean()) if not sac_df.empty else None,
                    "mean_saccade_amp_px": float(sac_df["amplitude"].mean()) if not sac_df.empty else None,
                    "saccades_total_duration_ms": float(sac_df["duration_ms"].sum()) if not sac_df.empty else None,
                    "std_pupil_diam": std_pupil,
                }

                aoi_features = {
                    **{f"{a}_pct_dur": float(aoi_dur.get(a, 0.0) / total_dur) if total_dur > 0 else None for a in aoi_labels},
                    **{f"n_fixations_{a}": int(aoi_counts.get(a, 0)) for a in aoi_labels},
                    **({f"transitions_{src}_{dst}": int(trans.loc[src, dst])
                        for src in aoi_labels
                        for dst in aoi_labels
                        if src in trans.index and dst in trans.columns} if not trans.empty else {}),
                }

                game_features = extract_game_features(game_data)

                row = {
                    "participant": sub,
                    "trial": trial_id,
                    "run": run_num,
                    "expertise": expertise.get(sub, "unknown"),
                    **eye_features,
                    **aoi_features,
                    **game_features,
                }

                print(
                    f"  {sub}  {trial_id}  run{run_num}  "
                    f"victims={row.get('saved_victims', '?')}  "
                    f"fixations={row.get('n_fixations', '?')}  "
                    f"saccades={row.get('n_saccades', '?')}"
                )

                rows.append(row)

    return pd.DataFrame(rows)
