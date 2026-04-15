import numpy as np
import pandas as pd
from igaze.detectors import fixation_detection, saccade_detection

from analysis.features.conventions import FIXATION_COLUMNS, SACCADE_COLUMNS
from analysis.prepare_data.xdf import load_all_subjects

"The reason we used this: https://link.springer.com/article/10.3758/s13428-013-0422-2"


def _preprocess_eye(eye_df: pd.DataFrame, eye_cfg: dict) -> pd.DataFrame:
    x_col = eye_cfg["x_col"]
    y_col = eye_cfg["y_col"]
    time_col = eye_cfg["time_col"]
    screen_w = eye_cfg["screen_w"]
    screen_h = eye_cfg["screen_h"]

    eye_df = eye_df.copy()
    eye_df[time_col] = (eye_df[time_col] - eye_df[time_col].iloc[0]) * 1000.0
    eye_df = eye_df.dropna(subset=[x_col, y_col])
    eye_df[x_col] = eye_df[x_col] * screen_w
    eye_df[y_col] = eye_df[y_col] * screen_h
    return eye_df


def detect_fixations(cfg: dict, preloaded: dict | None = None) -> dict:
    """Detect fixations for all subjects and trials.

    Args:
        cfg:       Full config dict.
        preloaded: {subject_id: {trial_id: {"eyetracker": DataFrame}}}.
                   If None, loads from data/intermediate via load_all_subjects.

    Returns:
        {subject_id: {trial_id: {"fixations": DataFrame}}}
    """
    if preloaded is None:
        preloaded = load_all_subjects(cfg)

    subjects = [str(s) for s in cfg.get("sub", [])]
    eye_cfg = cfg.get("eyetracker", {})
    fix_cfg = cfg.get("fixation", {})
    missing = eye_cfg["missing"]
    maxdist = fix_cfg["maxdist"]
    mindur = fix_cfg["mindur"]

    results: dict = {}

    for sid in subjects:
        data = preloaded.get(sid, {})
        if not data:
            print(f"[SKIP] {sid}: no trial data")
            continue

        print(f"[SUB]  {sid}")
        results[sid] = {}

        for trial_id, streams in data.items():
            eye_df = streams["eyetracker"]
            if eye_df.empty:
                print(f"         {trial_id:35s}  empty eyetracker — skipping")
                continue

            eye_df = _preprocess_eye(eye_df, eye_cfg)

            _, Efix = fixation_detection(
                x=eye_df[eye_cfg["x_col"]].to_numpy(dtype=float),
                y=eye_df[eye_cfg["y_col"]].to_numpy(dtype=float),
                time=eye_df[eye_cfg["time_col"]].to_numpy(dtype=float),
                missing=missing,
                maxdist=maxdist,
                mindur=mindur,
            )

            fix_df = pd.DataFrame(Efix or [], columns=FIXATION_COLUMNS)
            results[sid][trial_id] = {"fixations": fix_df}

    return results


def detect_saccades(cfg: dict, preloaded: dict | None = None) -> dict:
    """Detect saccades for all subjects and trials.

    Args:
        cfg:       Full config dict.
        preloaded: {subject_id: {trial_id: {"eyetracker": DataFrame}}}.
                   If None, loads from data/intermediate via load_all_subjects.

    Returns:
        {subject_id: {trial_id: {"saccades": DataFrame}}}
    """
    if preloaded is None:
        preloaded = load_all_subjects(cfg)

    subjects = [str(s) for s in cfg.get("sub", [])]
    eye_cfg = cfg.get("eyetracker", {})
    sac_cfg = cfg.get("saccade", {})
    missing = eye_cfg["missing"]
    minlen = sac_cfg["minlen"]
    maxvel = sac_cfg["maxvel"]
    maxacc = sac_cfg["maxacc"]

    results: dict = {}

    for sid in subjects:
        data = preloaded.get(sid, {})
        if not data:
            print(f"[SKIP] {sid}: no trial data")
            continue

        print(f"[SUB]  {sid}")
        results[sid] = {}

        for trial_id, streams in data.items():
            eye_df = streams["eyetracker"]
            if eye_df.empty:
                print(f"         {trial_id:35s}  empty eyetracker — skipping")
                continue

            eye_df = _preprocess_eye(eye_df, eye_cfg)

            _, end_saccades = saccade_detection(
                eye_df[eye_cfg["x_col"]].to_numpy(dtype=float),
                eye_df[eye_cfg["y_col"]].to_numpy(dtype=float),
                eye_df[eye_cfg["time_col"]].to_numpy(dtype=float),
                missing=missing,
                minlen=minlen,
                maxvel=maxvel,
                maxacc=maxacc,
            )

            raw_cols = [
                "start_ms",
                "end_ms",
                "duration_ms",
                "x_start",
                "y_start",
                "x_end",
                "y_end",
            ]
            sac_df = pd.DataFrame(end_saccades or [], columns=raw_cols)

            if not sac_df.empty:
                sac_df.insert(0, "saccade_id", sac_df.index + 1)
                sac_df["amplitude"] = np.hypot(
                    sac_df["x_end"] - sac_df["x_start"],
                    sac_df["y_end"] - sac_df["y_start"],
                )
            else:
                sac_df = pd.DataFrame(columns=SACCADE_COLUMNS)

            results[sid][trial_id] = {"saccades": sac_df}

    return results


def extract_eyetracking_features(eye_data, cfg: dict) -> dict:
    """Run both fixation and saccade detection.

    Returns:
        {"fixations": {...}, "saccades": {...}}
    """
    return {
        "fixations": detect_fixations(eye_data, cfg),
        "saccades": detect_saccades(eye_data, cfg),
        "aoi": aoi_features(eye_data, cfg),
    }
