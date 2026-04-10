
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
# AOI helpers
# ---------------------------------------------------------------------------

def assign_aoi(x_px: float, y_px: float, aois: list[dict]) -> str:
    """Return the name of the first AOI that contains (x_px, y_px), else 'offscreen'."""
    for aoi in aois:
        if aoi["x_min"] <= x_px <= aoi["x_max"] and aoi["y_min"] <= y_px <= aoi["y_max"]:
            return aoi["name"]
    return DEFAULT_OFFSCREEN_LABEL


def aoi_transition_matrix(fix_aoi_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    labels   = [a["name"] for a in aois]
    matrix   = pd.DataFrame(0, index=labels, columns=labels)
    sequence = fix_aoi_df[fix_aoi_df["aoi"] != DEFAULT_OFFSCREEN_LABEL]["aoi"].tolist()
    for src, dst in zip(sequence[:-1], sequence[1:]):
        if src in matrix.index and dst in matrix.columns:
            matrix.loc[src, dst] += 1
    return matrix


def label_fixations(fix_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    """Add an 'aoi' column to a fixations DataFrame (columns: x, y in pixels)."""
    fix_df = fix_df.copy()
    fix_df["aoi"] = fix_df.apply(lambda r: assign_aoi(r["x"], r["y"], aois), axis=1)
    return fix_df


# ---------------------------------------------------------------------------
# Per-subject processing (in-memory)
# ---------------------------------------------------------------------------

def _process_subject(subject_id: str,
                     fix_by_trial: dict,
                     processed_dir: Path,
                     aois: list[dict],
                     cfg: dict) -> dict:
    """Label fixations with AOIs for one subject using in-memory fixation data.

    Args:
        subject_id:   e.g. 'P001'
        fix_by_trial: {trial_id: {"fixations": DataFrame, "summary": dict}}
                      (one subject's slice of run_fixations() output)
        processed_dir: root processed path from config
        aois:         list of AOI dicts from config

    Returns:
        {trial_id: {"fix_aoi": DataFrame, "transitions": DataFrame, "summary": dict}}
    """
    print(f"  [SUB]  {subject_id}")
    results = {}

    for trial_id, streams in fix_by_trial.items():
        fix_df = streams["fixations"]

        if fix_df.empty:
            print(f"         {trial_id:35s}  no fixations — skipping AOI labelling")
            results[trial_id] = {
                "fix_aoi":    pd.DataFrame(),
                "transitions": pd.DataFrame(),
                "summary": {
                    "n_fixations": 0,
                    "n_fixations_offscreen": 0,
                    **{f"n_fixations_{a['name']}": 0 for a in aois},
                },
            }
            continue

        fix_aoi = label_fixations(fix_df, aois)
        trans   = aoi_transition_matrix(fix_aoi, aois)

        out_dir = processed_dir / subject_label(subject_id, cfg) / trial_id
        out_dir.mkdir(parents=True, exist_ok=True)
        fix_aoi.to_csv(out_dir / "fixations_aoi.csv", index=False)
        trans.to_csv(out_dir / "aoi_transitions.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            fix_aoi.to_hdf(out_dir / "fixations_aoi.h5", key="fixations_aoi", mode="w")

        counts     = fix_aoi["aoi"].value_counts().to_dict()
        dur_by_aoi = fix_aoi.groupby("aoi")["duration_ms"].agg(["sum", "mean"])
        total_dur  = fix_aoi["duration_ms"].sum()
        n_total    = len(fix_aoi)
        n_off      = counts.get(DEFAULT_OFFSCREEN_LABEL, 0)

        aoi_stats = {}
        for a in aois:
            n   = counts.get(a["name"], 0)
            dur = float(dur_by_aoi.loc[a["name"], "sum"])  if a["name"] in dur_by_aoi.index else 0.0
            avg = float(dur_by_aoi.loc[a["name"], "mean"]) if a["name"] in dur_by_aoi.index else 0.0
            pct = round(dur / total_dur * 100, 2) if total_dur > 0 else 0.0
            aoi_stats[f"n_fixations_{a['name']}"]  = n
            aoi_stats[f"{a['name']}_total_dur_ms"] = round(dur, 2)
            aoi_stats[f"{a['name']}_mean_dur_ms"]  = round(avg, 2)
            aoi_stats[f"{a['name']}_pct_dur"]      = pct

        aoi_parts = [
            f"{a['name']}={aoi_stats['n_fixations_' + a['name']]}({aoi_stats[a['name'] + '_pct_dur']}%)"
            for a in aois
        ]
        print(f"         {trial_id:35s}  total={n_total:>4}  offscreen={n_off:>3}  "
              + "  ".join(aoi_parts))

        off_dur = (
            float(dur_by_aoi.loc[DEFAULT_OFFSCREEN_LABEL, "sum"])
            if DEFAULT_OFFSCREEN_LABEL in dur_by_aoi.index else 0.0
        )
        off_pct = round(off_dur / total_dur * 100, 2) if total_dur > 0 else 0.0

        results[trial_id] = {
            "fix_aoi":    fix_aoi,
            "transitions": trans,
            "summary": {
                "subject":               subject_id,
                "trial":                 trial_id,
                "n_fixations":           n_total,
                "n_fixations_offscreen": n_off,
                "offscreen_pct_dur":     off_pct,
                **aoi_stats,
            },
        }

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_aoi_fixations(cfg: dict, eyetracking: dict | None = None,
                      preloaded: dict | None = None,
                      root: Path | None = None) -> dict:
    """Label fixations with AOIs for all subjects and save results.

    Args:
        cfg:         Full config dict.
        eyetracking: Output of run_eyetracking() —
                     {"fixations": {subject_id: {trial_id: {"fixations": df, ...}}}, ...}
                     If None, runs eyetracking detection first from preloaded.
        preloaded:   Passed to run_eyetracking() if eyetracking is None.

    Returns:
        List of per-trial AOI summary dicts.
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
        aoi_results[sid] = _process_subject(sid, fix_by_trial, processed_dir, aois, cfg)

    all_summaries = [
        trial_data["summary"]
        for sid, trials in aoi_results.items()
        for trial_data in trials.values()
    ]
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        out_path   = processed_dir / "aoi_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"\nAOI summary -> {out_path.relative_to(root)}")

    # Aggregate transition matrices
    aoi_names   = [a["name"] for a in aois]
    total_trans = pd.DataFrame(0, index=aoi_names, columns=aoi_names)
    by_trial: dict[str, pd.DataFrame] = {}

    for sid in subjects:
        sub_proc = processed_dir / subject_label(sid, cfg)
        if not sub_proc.exists():
            continue
        for trial_dir in sorted(sub_proc.iterdir()):
            t_file = trial_dir / "aoi_transitions.csv"
            if not t_file.exists():
                continue
            mat = pd.read_csv(t_file, index_col=0)
            mat = mat.reindex(index=aoi_names, columns=aoi_names, fill_value=0)
            total_trans += mat
            base = strip_run_suffix(trial_dir.name, cfg)
            if base not in by_trial:
                by_trial[base] = pd.DataFrame(0, index=aoi_names, columns=aoi_names)
            by_trial[base] += mat

    out_trans = processed_dir / "aoi_transitions_all.csv"
    total_trans.to_csv(out_trans)
    print(f"Aggregated transitions (all) -> {out_trans.relative_to(root)}")

    for base, mat in sorted(by_trial.items()):
        out = processed_dir / f"aoi_transitions_{base}.csv"
        mat.to_csv(out)
        print(f"Aggregated transitions ({base}) -> {out.relative_to(root)}")

    return aoi_results
