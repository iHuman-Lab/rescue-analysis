import pandas as pd

from .aoi_fixation import build_aoi_features, run_aoi
from .eyetracking_data import build_eye_features, run_eyetracking
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
                aoi = run_aoi(et["fixations"], aois)

                eye_features = build_eye_features(
                    et["fixations"], et["saccades"], eye_data, eye_cfg
                )
                aoi_features = build_aoi_features(
                    aoi["fix_aoi"], aoi["transitions"], aois, offscreen_label
                )
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
