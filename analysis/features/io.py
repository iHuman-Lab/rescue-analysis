from pathlib import Path

import pandas as pd

from analysis.features.conventions import output_file


def save_aoi_results(aoi_results: dict, processed_dir: Path, cfg: dict) -> None:
    for sid, trials in aoi_results.items():
        for trial_id, trial_data in trials.items():
            fix_aoi = trial_data.get("fix_aoi", pd.DataFrame())
            if fix_aoi.empty:
                continue
            out = processed_dir / sid / trial_id / output_file(cfg, "fixation_aoi_file")
            out.parent.mkdir(parents=True, exist_ok=True)
            fix_aoi.to_csv(out, index=False)


def save_aggregated_transitions(
    total_trans: pd.DataFrame, by_trial: dict, processed_dir: Path, cfg: dict
) -> None:
    if not total_trans.empty:
        out = processed_dir / output_file(cfg, "aoi_transitions_all_file")
        out.parent.mkdir(parents=True, exist_ok=True)
        total_trans.to_csv(out)

    by_trial_template = output_file(cfg, "aoi_transitions_by_trial_template")
    for trial_id, trans_df in by_trial.items():
        if trans_df.empty:
            continue
        out = processed_dir / by_trial_template.format(trial=trial_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        trans_df.to_csv(out)


def save_glmm_results(results: pd.DataFrame, processed_dir: Path, cfg: dict) -> Path:
    filename = cfg.get("glmm2", {}).get("output_file", "glmm2_results.csv")
    out = processed_dir / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    return out


def save_best_features(best_features: pd.DataFrame, processed_dir: Path, cfg: dict) -> Path:
    out = processed_dir / output_file(cfg, "best_features_file")
    out.parent.mkdir(parents=True, exist_ok=True)
    best_features.to_csv(out, index=False)
    return out
