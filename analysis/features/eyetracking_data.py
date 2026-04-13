
import numpy as np
import pandas as pd
from igaze.detectors import fixation_detection, saccade_detection

from analysis.data.xdf import load_all_subjects
from analysis.features.conventions import (
    FIXATION_COLUMNS,
    SACCADE_COLUMNS,
)

"The reason we used this: https://link.springer.com/article/10.3758/s13428-013-0422-2"

# ---------------------------------------------------------------------------
# Shared eye preprocessingF
# ---------------------------------------------------------------------------

def _preprocess_eye(eye_df: pd.DataFrame, eye_cfg: dict) -> pd.DataFrame:
    x_col    = eye_cfg["x_col"]
    y_col    = eye_cfg["y_col"]
    time_col = eye_cfg["time_col"]
    screen_w = eye_cfg["screen_w"]
    screen_h = eye_cfg["screen_h"]

    eye_df = eye_df.copy()
    eye_df[time_col] = (eye_df[time_col] - eye_df[time_col].iloc[0]) * 1000.0
    eye_df = eye_df.dropna(subset=[x_col, y_col])
    eye_df[x_col] = eye_df[x_col] * screen_w
    eye_df[y_col] = eye_df[y_col] * screen_h
    return eye_df


def _iter_preprocessed_trials(cfg: dict, preloaded: dict | None = None):
    if preloaded is None:
        preloaded = load_all_subjects(cfg)

    eye_cfg = cfg.get("eyetracker", {})
    subjects = [str(s) for s in cfg.get("sub", [])]

    for sid in subjects:
        data = preloaded.get(sid, {})
        if not data:
            print(f"[SKIP] {sid}: no trial data")
            continue

        print(f"[SUB]  {sid}")
        trials = []
        for trial_id, streams in data.items():
            eye_df = streams["eyetracker"]
            if eye_df.empty:
                print(f"         {trial_id:35s}  empty eyetracker — skipping")
                continue
            trials.append((trial_id, _preprocess_eye(eye_df, eye_cfg)))

        yield sid, eye_cfg, trials


def _eye_arrays(eye_df: pd.DataFrame, eye_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        eye_df[eye_cfg["x_col"]].to_numpy(dtype=float),
        eye_df[eye_cfg["y_col"]].to_numpy(dtype=float),
        eye_df[eye_cfg["time_col"]].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# Fixations
# ---------------------------------------------------------------------------

def fixation_summary(Efix: list) -> dict:
    if not Efix:
        return {"count": 0, "mean_duration_ms": 0.0, "total_duration_ms": 0.0}
    durations = [f[2] for f in Efix]
    return {
        "count":             len(durations),
        "mean_duration_ms":  sum(durations) / len(durations),
        "total_duration_ms": sum(durations),
    }


def run_fixations(cfg: dict, preloaded: dict | None = None) -> dict:
    """Detect fixations for all subjects and trials.

    Args:
        cfg:       Full config dict.
        preloaded: {subject_id: {trial_id: {"eyetracker": DataFrame}}}.
                   If None, loads from data/intermediate via load_all_subjects.

    Returns:
        {subject_id: {trial_id: {"fixations": DataFrame, "summary": dict}}}
    """
    fix_cfg = cfg.get("fixation", {})
    params = {
        "missing": cfg.get("eyetracker", {})["missing"],
        "maxdist": fix_cfg["maxdist"],
        "mindur": fix_cfg["mindur"],
    }

    results: dict = {}

    for sid, eye_cfg, trials in _iter_preprocessed_trials(cfg, preloaded=preloaded):
        results[sid] = {}

        for trial_id, eye_df in trials:
            x, y, time = _eye_arrays(eye_df, eye_cfg)

            _, Efix = fixation_detection(
                x=x,
                y=y,
                time=time,
                missing=params["missing"],
                maxdist=params["maxdist"],
                mindur=params["mindur"],
            )
            summary = fixation_summary(Efix)

            print(f"         {trial_id:35s}  "
                  f"fixations={summary['count']:>4}  "
                  f"mean_dur={summary['mean_duration_ms']:>7.1f}ms")

            fix_df = (
                pd.DataFrame(Efix, columns=FIXATION_COLUMNS)
                if Efix else pd.DataFrame(columns=FIXATION_COLUMNS)
            )
            results[sid][trial_id] = {"fixations": fix_df, "summary": summary}

    return results


# ---------------------------------------------------------------------------
# Saccades
# ---------------------------------------------------------------------------

def run_saccades(cfg: dict, preloaded: dict | None = None) -> dict:
    """Detect saccades for all subjects and trials.

    Args:
        cfg:       Full config dict.
        preloaded: {subject_id: {trial_id: {"eyetracker": DataFrame}}}.
                   If None, loads from data/intermediate via load_all_subjects.

    Returns:
        {subject_id: {trial_id: {"saccades": DataFrame, "summary": dict}}}
    """
    sac_cfg = cfg.get("saccade", {})
    params = {
        "missing": cfg.get("eyetracker", {})["missing"],
        "minlen": sac_cfg["minlen"],
        "maxvel": sac_cfg["maxvel"],
        "maxacc": sac_cfg["maxacc"],
    }

    results: dict = {}

    for sid, eye_cfg, trials in _iter_preprocessed_trials(cfg, preloaded=preloaded):
        results[sid] = {}

        for trial_id, eye_df in trials:
            x, y, time = _eye_arrays(eye_df, eye_cfg)

            _, end_saccades = saccade_detection(
                x,
                y,
                time,
                missing=params["missing"],
                minlen=params["minlen"],
                maxvel=params["maxvel"],
                maxacc=params["maxacc"],
            )

            rows = []
            for sac_id, sac in enumerate(end_saccades, start=1):
                start_t, end_t, duration, xs, ys, xe, ye = sac
                rows.append({
                    "saccade_id":  sac_id,
                    "start_ms":    float(start_t),
                    "end_ms":      float(end_t),
                    "duration_ms": float(duration),
                    "x_start":     float(xs),
                    "y_start":     float(ys),
                    "x_end":       float(xe),
                    "y_end":       float(ye),
                    "amplitude":   float(((xe - xs) ** 2 + (ye - ys) ** 2) ** 0.5),
                })

            n        = len(rows)
            total_ms = sum(r["duration_ms"] for r in rows)
            mean_dur = total_ms / n if n > 0 else 0.0
            mean_amp = float(np.mean([r["amplitude"] for r in rows])) if rows else 0.0

            print(f"         {trial_id:35s}  "
                  f"saccades={n:>4}  "
                  f"mean_dur={mean_dur:>6.1f}ms  "
                  f"mean_amp={mean_amp:>6.1f}px")

            sac_df = (
                pd.DataFrame(rows)
                if rows else pd.DataFrame(columns=SACCADE_COLUMNS)
            )
            results[sid][trial_id] = {
                "saccades": sac_df,
                "summary": {
                    "n_saccades":        n,
                    "total_duration_ms": total_ms,
                    "mean_duration_ms":  mean_dur,
                    "mean_amplitude_px": mean_amp,
                },
            }

    return results


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def run_eyetracking(cfg: dict, preloaded: dict | None = None) -> dict:
    """Run both fixation and saccade detection.

    Returns:
        {"fixations": {...}, "saccades": {...}}
    """
    return {
        "fixations": run_fixations(cfg, preloaded=preloaded),
        "saccades":  run_saccades(cfg, preloaded=preloaded),
    }