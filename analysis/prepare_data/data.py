"""analysis/prepare_data/data.py — Subject processing and trial iteration."""

import warnings
from pathlib import Path

import pandas as pd
import pyxdf

from analysis.prepare_data.utils import get_stream, split_by_trial


def process_subject(subject_id: str, cfg: dict) -> dict[str, dict[int, dict]]:
    """Parse one subject's XDF and return trial data split by episode.

    Returns:
        {trial_id: {run_num: {"game": DataFrame, "eye_tracking": DataFrame}}}
    """
    xdf_path = Path(
        f"{cfg['paths']['raw']}/sub-{subject_id}/ses-S001/sarmissiong/sub-{subject_id}_ses-S001_task-Default_run-001_sarmissiong.xdf"
    )
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    game_stream = get_stream(streams, cfg["xdf"]["game_stream"])
    eye_stream = get_stream(streams, cfg["xdf"]["eye_stream"])
    return split_by_trial(game_stream, eye_stream, cfg)


def split_by_trail(cfg: dict) -> dict:
    """Process all subjects and return a nested dict saved to a single HDF5 file.

    Structure: {subject_id: {trail: {run_num: {"game": DataFrame, "eye_tracking": DataFrame}}}}
    Saved to cfg["paths"]["processed"]/data.h5 with keys
    /{subject_id}/{trail}/run_{n}/game and /{subject_id}/{trail}/run_{n}/eye_tracking.
    """
    subjects = [str(s) for s in cfg.get("sub", [])]
    trails = [str(t) for t in cfg.get("trails", [])]

    processed_dir = Path(cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    h5_path = processed_dir / "data.h5"

    result: dict = {}

    for sid in subjects:
        print(f"[SUB]  {sid}")
        result[sid] = {}
        for trial_id, runs in process_subject(sid, cfg).items():
            trail = next((t for t in trails if t in trial_id), None)
            if trail is None:
                continue
            result[sid][trail] = runs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.HDFStore(str(h5_path), mode="w") as store:
            for sid, trail_data in result.items():
                for trail, runs in trail_data.items():
                    for run_num, streams in runs.items():
                        store[f"/{sid}/{trail}/run_{run_num}/game"] = streams["game"]
                        store[f"/{sid}/{trail}/run_{run_num}/eye_tracking"] = streams[
                            "eye_tracking"
                        ]

    print(f"[SAVE] {h5_path}")
    return result
