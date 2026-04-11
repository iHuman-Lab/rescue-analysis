import warnings
from pathlib import Path

import pandas as pd

from analysis.data.xdf import load_all_subjects
from analysis.features.conventions import (
    DEFAULT_OFFSCREEN_LABEL,
    strip_run_suffix,
    subject_label,
)
from analysis.features.eyetracking_data import run_eyetracking


# ---------------------------------------------------------------------------
# Core AOI functions
# ---------------------------------------------------------------------------

def assign_aoi(x_px: float, y_px: float, aois: list[dict]) -> str:
    """Return the name of the first AOI that contains (x_px, y_px), else 'offscreen'."""
    for aoi in aois:
        if aoi["x_min"] <= x_px <= aoi["x_max"] and aoi["y_min"] <= y_px <= aoi["y_max"]:
            return aoi["name"]
    return DEFAULT_OFFSCREEN_LABEL


def label_fixations(fix_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    """Add an 'aoi' column to a fixations DataFrame (needs x, y columns in pixels)."""
    fix_df = fix_df.copy()
    fix_df["aoi"] = fix_df.apply(lambda r: assign_aoi(r["x"], r["y"], aois), axis=1)
    return fix_df


def aoi_transition_matrix(fix_aoi_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    """Count consecutive AOI-to-AOI transitions (excluding offscreen)."""
    labels   = [a["name"] for a in aois]
    matrix   = pd.DataFrame(0, index=labels, columns=labels)
    sequence = fix_aoi_df[fix_aoi_df["aoi"] != DEFAULT_OFFSCREEN_LABEL]["aoi"].tolist()
    for src, dst in zip(sequence[:-1], sequence[1:]):
        if src in matrix.index and dst in matrix.columns:
            matrix.loc[src, dst] += 1
    return matrix


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_aoi_fixations(cfg: dict, eyetracking: dict | None = None,
                      preloaded: dict | None = None,
                      root: Path | None = None) -> dict:
    """Label fixations with AOIs for every subject / trial.

    Returns:
        {subject_id: {trial_id: {"fix_aoi": DataFrame, "transitions": DataFrame}}}
    """
    if eyetracking is None:
        if preloaded is None:
            preloaded = load_all_subjects(cfg)
        eyetracking = run_eyetracking(cfg, preloaded=preloaded)

    if root is None:
        root = Path(__file__).resolve().parents[2]
    subjects      = [str(s) for s in cfg.get("sub", [])]
    aois          = cfg.get("aoi", [])
    processed_dir = root / cfg["paths"]["processed"]
    fix_by_sub    = eyetracking.get("fixations", {})

    if not aois:
        print("WARNING: no AOIs defined in config — all fixations will be labelled 'offscreen'")

    aoi_results: dict = {}
    for sid in subjects:
        fix_by_trial = fix_by_sub.get(sid, {})
        if not fix_by_trial:
            print(f"  [SKIP] {sid}: no fixation data")
            continue

        print(f"  [SUB]  {sid}")
        aoi_results[sid] = {}
        for trial_id, streams in fix_by_trial.items():
            fix_df = streams["fixations"]

            if fix_df.empty:
                print(f"         {trial_id:35s}  no fixations — skipping")
                aoi_results[sid][trial_id] = {
                    "fix_aoi": pd.DataFrame(), "transitions": pd.DataFrame(),
                }
                continue

            fix_aoi = label_fixations(fix_df, aois)
            trans   = aoi_transition_matrix(fix_aoi, aois)

            # Save per-trial outputs
            out_dir = processed_dir / subject_label(sid, cfg) / trial_id
            out_dir.mkdir(parents=True, exist_ok=True)
            fix_aoi.to_csv(out_dir / "fixations_aoi.csv", index=False)
            trans.to_csv(out_dir / "aoi_transitions.csv")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                fix_aoi.to_hdf(out_dir / "fixations_aoi.h5", key="fixations_aoi", mode="w")

            n_total = len(fix_aoi)
            n_off   = int((fix_aoi["aoi"] == DEFAULT_OFFSCREEN_LABEL).sum())
            print(f"         {trial_id:35s}  fixations={n_total}  offscreen={n_off}")

            aoi_results[sid][trial_id] = {"fix_aoi": fix_aoi, "transitions": trans}

    # Aggregate transition matrices
    _save_aggregated_transitions(aoi_results, aois, cfg, processed_dir, root)

    return aoi_results


def _save_aggregated_transitions(aoi_results, aois, cfg, processed_dir, root):
    """Sum transition matrices across all subjects, total and by trial base name."""
    aoi_names = [a["name"] for a in aois]
    if not aoi_names:
        return
    total = pd.DataFrame(0, index=aoi_names, columns=aoi_names)
    by_trial: dict[str, pd.DataFrame] = {}

    for trials in aoi_results.values():
        for trial_id, trial_data in trials.items():
            mat = trial_data["transitions"]
            if mat.empty:
                continue
            mat = mat.reindex(index=aoi_names, columns=aoi_names, fill_value=0)
            total += mat
            base = strip_run_suffix(trial_id, cfg)
            if base not in by_trial:
                by_trial[base] = pd.DataFrame(0, index=aoi_names, columns=aoi_names)
            by_trial[base] += mat

    total.to_csv(processed_dir / "aoi_transitions_all.csv")
    print(f"Aggregated transitions (all) -> {(processed_dir / 'aoi_transitions_all.csv').relative_to(root)}")
    for base, mat in sorted(by_trial.items()):
        out = processed_dir / f"aoi_transitions_{base}.csv"
        mat.to_csv(out)
        print(f"Aggregated transitions ({base}) -> {out.relative_to(root)}")
