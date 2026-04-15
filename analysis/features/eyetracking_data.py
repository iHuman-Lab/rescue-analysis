import numpy as np
import pandas as pd
from igaze.detectors import fixation_detection, saccade_detection

"The reason we used this: https://link.springer.com/article/10.3758/s13428-013-0422-2"

FIXATION_COLUMNS: list[str] = ["start_ms", "end_ms", "duration_ms", "x", "y"]
SACCADE_COLUMNS: list[str] = [
    "saccade_id", "start_ms", "end_ms", "duration_ms",
    "x_start", "y_start", "x_end", "y_end", "amplitude",
]


def _preprocess_eye(eye_df: pd.DataFrame, eye_cfg: dict):
    """Preprocess eye-tracking DataFrame and return (df, x, y, time) arrays."""
    x_col = eye_cfg["x_col"]
    y_col = eye_cfg["y_col"]
    time_col = eye_cfg["time_col"]

    eye_df = eye_df.copy()
    eye_df[time_col] = (eye_df[time_col] - eye_df[time_col].iloc[0]) * 1000.0
    eye_df = eye_df.dropna(subset=[x_col, y_col])
    eye_df[x_col] = eye_df[x_col] * eye_cfg["screen_w"]
    eye_df[y_col] = eye_df[y_col] * eye_cfg["screen_h"]

    x = eye_df[x_col].to_numpy(dtype=float)
    y = eye_df[y_col].to_numpy(dtype=float)
    time = eye_df[time_col].to_numpy(dtype=float)
    return eye_df, x, y, time


def detect_fixations(eye_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if eye_df is None or eye_df.empty:
        return pd.DataFrame(columns=FIXATION_COLUMNS)

    eye_cfg = cfg.get("eyetracker", {})
    fix_cfg = cfg.get("fixation", {})
    _, x, y, time = _preprocess_eye(eye_df, eye_cfg)

    _, Efix = fixation_detection(
        x=x, y=y, time=time,
        missing=eye_cfg["missing"],
        maxdist=fix_cfg["maxdist"],
        mindur=fix_cfg["mindur"],
    )
    return pd.DataFrame(Efix or [], columns=FIXATION_COLUMNS)


def detect_saccades(eye_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if eye_df is None or eye_df.empty:
        return pd.DataFrame(columns=SACCADE_COLUMNS)

    eye_cfg = cfg.get("eyetracker", {})
    sac_cfg = cfg.get("saccade", {})
    _, x, y, time = _preprocess_eye(eye_df, eye_cfg)

    _, end_saccades = saccade_detection(
        x, y, time,
        missing=eye_cfg["missing"],
        minlen=sac_cfg["minlen"],
        maxvel=sac_cfg["maxvel"],
        maxacc=sac_cfg["maxacc"],
    )

    raw_cols = ["start_ms", "end_ms", "duration_ms", "x_start", "y_start", "x_end", "y_end"]
    sac_df = pd.DataFrame(end_saccades or [], columns=raw_cols)

    if not sac_df.empty:
        sac_df.insert(0, "saccade_id", sac_df.index + 1)
        sac_df["amplitude"] = np.hypot(
            sac_df["x_end"] - sac_df["x_start"],
            sac_df["y_end"] - sac_df["y_start"],
        )
    else:
        sac_df = pd.DataFrame(columns=SACCADE_COLUMNS)

    return sac_df


def run_eyetracking(eye_df: pd.DataFrame, cfg: dict) -> dict:
    """Run fixation and saccade detection for a single trial.

    Returns:
        {"fixations": DataFrame, "saccades": DataFrame}
    """
    return {
        "fixations": detect_fixations(eye_df, cfg),
        "saccades": detect_saccades(eye_df, cfg),
    }
