"""analysis/prepare_data/xdf.py — Config loading and subject data access."""

from pathlib import Path

import pandas as pd
import yaml

from analysis.prepare_data.data import process_subject


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Legacy column remapping
# ---------------------------------------------------------------------------


def _remap_legacy_eye_columns(
    eye_df: pd.DataFrame, channels: list[str] | None
) -> pd.DataFrame:
    """Repair older intermediate eye files saved with a shifted channel list.

    Older configs omitted the leading Tobii device timestamp channel, which caused
    every subsequent eye column to be shifted left by one and left the final field
    as ch8. When that exact pattern is present, reassign columns by position.
    """
    if not channels:
        return eye_df

    data_columns = [col for col in eye_df.columns if col != "timestamp"]
    if data_columns == channels:
        return eye_df

    legacy_columns = list(channels[1:]) + [f"ch{len(channels) - 1}"]
    if data_columns == legacy_columns:
        remapped = eye_df.copy()
        remapped.columns = ["timestamp", *channels]
        return remapped

    return eye_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_all_subjects(
    cfg: dict, root: Path | None = None
) -> dict[str, dict[str, dict]]:
    """Load all subjects from data/intermediate (no XDF processing).

    Use this when the XDF step was skipped but downstream steps still need
    the data in memory instead of each re-reading from disk independently.

    Returns:
        {subject_id: {trial_id: {"eyetracker": DataFrame, "game": DataFrame}}}
    """
    if root is None:
        root = Path(__file__).resolve().parents[2]

    intermediate = root / cfg["paths"]["intermediate"]
    sub_tmpl = (
        cfg["xdf"].get("output", {}).get("subject_dir_template", "sub-{subject_id}")
    )
    all_data: dict[str, dict[str, dict]] = {}

    for subject_id in cfg.get("sub", []):
        sid = str(subject_id)
        sub_dir = intermediate / sub_tmpl.format(subject_id=sid)
        if not sub_dir.exists():
            print(f"[SKIP] {sid}: not found in {intermediate.relative_to(root)}")
            continue

        channels = cfg.get("eyetracker", {}).get("channels")

        trials: dict[str, dict] = {}
        for trial_dir in sorted(sub_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            h5 = trial_dir / "eyetracker.h5"
            if not h5.exists():
                continue
            eye_df = pd.read_hdf(h5, key="eyetracker")
            eye_df = _remap_legacy_eye_columns(eye_df, channels)
            if channels:
                rename = {
                    f"ch{i}": name
                    for i, name in enumerate(channels)
                    if f"ch{i}" in eye_df.columns
                }
                if rename:
                    eye_df = eye_df.rename(columns=rename)
            trial_data: dict = {"eyetracker": eye_df}
            game_h5_name = (
                cfg["xdf"].get("output", {}).get("files", {}).get("game_h5", "game.h5")
            )
            game_h5 = trial_dir / game_h5_name
            if game_h5.exists():
                trial_data["game"] = pd.read_hdf(game_h5, key="game")
            trials[trial_dir.name] = trial_data

        if trials:
            print(
                f"[LOAD] {sid}: {len(trials)} trials from {sub_dir.relative_to(root)}"
            )
        all_data[sid] = trials

    return all_data


def collect_subjects(cfg: dict, root: Path | None = None) -> dict[str, dict[str, dict]]:
    """Process all subjects from XDF files, save to disk, return data in memory.

    Returns:
        {subject_id: {trial_id: {"game": DataFrame, "eyetracker": DataFrame}}}
    """
    if root is None:
        root = Path(__file__).resolve().parents[2]
    all_data: dict[str, dict[str, dict]] = {}
    for subject_id in cfg.get("sub", []):
        sid = str(subject_id)
        print(f"[XDF] {sid}")
        all_data[sid] = process_subject(sid, cfg, root)
    return all_data


def run_from_config(cfg: dict, root: Path | None = None) -> None:
    collect_subjects(cfg, root)
