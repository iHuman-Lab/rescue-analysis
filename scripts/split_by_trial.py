"""
split_by_trial.py

Reads subjects from configs/config_analysis.yml, finds their XDF files
under data/raw/, and saves per-trial streams to data/processed/.

Output structure:
    data/processed/
        sub-P001/
            trial_dummy/
                game.csv
                eyetracker.csv
            trial_001/
                game.csv
                eyetracker.csv
        sub-P002/
            ...

Usage:
    python scripts/split_by_trial.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
import yaml

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent
CONFIG     = ROOT / "configs" / "config_analysis.yml"
RAW_DIR    = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

GAME_STREAM = "SARGame"
EYE_STREAM  = "TobiiEyeTracker"
TRIAL_FIELD = "trial_id"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_xdf(subject_id: str) -> Path | None:
    """Search data/raw recursively for an XDF file matching subject_id."""
    for f in RAW_DIR.rglob("*.xdf"):
        if subject_id.lower() in f.name.lower():
            return f
    return None


def parse_game(stream: dict) -> pd.DataFrame:
    """Flatten JSON game samples, drop array fields (grid, image)."""
    rows = []
    for ts, v in zip(stream["time_stamps"], stream["time_series"]):
        try:
            d = json.loads(v[0] if isinstance(v, (list, tuple)) else v)
        except (json.JSONDecodeError, TypeError):
            continue
        flat = {"timestamp": float(ts)}
        for k, val in d.items():
            if isinstance(val, (str, int, float, bool)) or val is None:
                flat[k] = val
        rows.append(flat)
    return pd.DataFrame(rows)


def parse_eye(stream: dict) -> pd.DataFrame:
    """Parse eyetracker float samples into a DataFrame."""
    data = np.array(stream["time_series"], dtype=float)
    # replace zeros with NaN for gaze columns (invalid samples)
    cols = [f"ch{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=cols)
    df.insert(0, "timestamp", stream["time_stamps"])
    return df


def split_by_trial(game_df: pd.DataFrame,
                   eye_df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """Split game and eye data into per-trial segments."""
    if TRIAL_FIELD not in game_df.columns:
        raise ValueError(f"'{TRIAL_FIELD}' not in game stream columns")

    result = {}
    for trial_id, group in game_df.groupby(TRIAL_FIELD, sort=False):
        t_start = group["timestamp"].iloc[0]
        t_end   = group["timestamp"].iloc[-1]
        mask    = (eye_df["timestamp"] >= t_start) & (eye_df["timestamp"] <= t_end)
        result[trial_id] = {
            "game":       group.reset_index(drop=True),
            "eyetracker": eye_df[mask].reset_index(drop=True),
        }
    return result


def get_stream(streams: list, name: str) -> dict | None:
    for s in streams:
        if s["info"].get("name", [""])[0] == name:
            return s
    return None


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(subject_id: str) -> None:
    xdf_path = find_xdf(subject_id)
    if xdf_path is None:
        print(f"  [SKIP] {subject_id}: no XDF found in {RAW_DIR}")
        return

    print(f"  [LOAD] {subject_id}: {xdf_path.relative_to(ROOT)}")
    streams, _ = pyxdf.load_xdf(str(xdf_path))

    game_stream = get_stream(streams, GAME_STREAM)
    eye_stream  = get_stream(streams, EYE_STREAM)

    if game_stream is None:
        print(f"  [SKIP] {subject_id}: '{GAME_STREAM}' stream not found")
        return
    if eye_stream is None:
        print(f"  [SKIP] {subject_id}: '{EYE_STREAM}' stream not found")
        return

    game_df = parse_game(game_stream)
    eye_df  = parse_eye(eye_stream)
    trials  = split_by_trial(game_df, eye_df)

    for trial_id, data in trials.items():
        out = OUTPUT_DIR / f"sub-{subject_id}" / trial_id
        out.mkdir(parents=True, exist_ok=True)

        data["game"].to_csv(out / "game.csv", index=False)
        data["eyetracker"].to_csv(out / "eyetracker.csv", index=False)

        print(f"         {trial_id:30s}  "
              f"game={len(data['game']):>5} rows  "
              f"eye={len(data['eyetracker']):>7} rows  "
              f"-> data/processed/sub-{subject_id}/{trial_id}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    subjects = cfg.get("sub", [])
    print(f"Processing {len(subjects)} subject(s) from {CONFIG.relative_to(ROOT)}\n")

    for sid in subjects:
        process_subject(str(sid))

    print("\nDone.")
