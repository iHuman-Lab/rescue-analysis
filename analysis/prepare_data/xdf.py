"""analysis/data/xdf.py — XDF → per-trial CSV / HDF5. Called from main.py."""
 
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
import yaml
 
 
def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
 
 
def _paths(cfg: dict, root: Path) -> dict[str, Path]:
    return {k: root / cfg["paths"][k] for k in ("raw", "intermediate", "processed")}
 
 
# ---------------------------------------------------------------------------
# Stream parsing
# ---------------------------------------------------------------------------
 
def _get_stream(streams: list, name: str) -> dict | None:
    return next((s for s in streams if s["info"].get("name", [""])[0] == name), None)
 
 
def _parse_game(stream: dict) -> pd.DataFrame:
    rows = []
    for ts, v in zip(stream["time_stamps"], stream["time_series"]):
        try:
            d = json.loads(v[0] if isinstance(v, (list, tuple)) else v)
        except (json.JSONDecodeError, TypeError):
            continue
        rows.append({"timestamp": float(ts), **{k: w for k, w in d.items() if not isinstance(w, (dict, list))}})
    return pd.DataFrame(rows)
 
 
def _parse_eye(stream: dict, channels: list[str] | None = None) -> pd.DataFrame:
    """Column names come from config (eyetracker.channels); fallback to ch0, ch1, …
    If channels has fewer entries than data columns, named channels are used for the
    first N and generic ch{N} names for the remainder."""
    data = np.array(stream["time_series"], dtype=float)
    n = data.shape[1]
    if channels:
        cols = list(channels[:n]) + [f"ch{i}" for i in range(len(channels), n)]
    else:
        cols = [f"ch{i}" for i in range(n)]
    df = pd.DataFrame(data, columns=cols)
    df.insert(0, "timestamp", stream["time_stamps"])
    return df


def _remap_legacy_eye_columns(eye_df: pd.DataFrame, channels: list[str] | None) -> pd.DataFrame:
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
# Trial splitting + best-run selection
 
def _split_by_trial(
    game_df: pd.DataFrame,
    eye_df: pd.DataFrame,
    xcfg: dict,
    run_tmpl: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    tfield    = xcfg["trial_field"]
    rfield    = xcfg.get("reset_field")
    threshold = xcfg.get("reset_drop_threshold", 5)
    afield    = xcfg.get("active_status_field")
    avalue    = xcfg.get("active_status_value")
 
    if tfield not in game_df.columns:
        raise ValueError(f"'{tfield}' not in game columns: {list(game_df.columns)}")
 
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for tid, group in game_df.groupby(tfield, sort=False):
        group = group.reset_index(drop=True)
 
        # detect retries: step_count drops → new run
        if rfield and rfield in group.columns:
            sc = pd.to_numeric(group[rfield], errors="coerce").fillna(0)
            bounds = [0] + list(group.index[sc.diff() < -threshold]) + [len(group)]
        else:
            bounds = [0, len(group)]
 
        n = len(bounds) - 1
        for i in range(n):
            slc = group.iloc[bounds[i]:bounds[i + 1]].reset_index(drop=True)
            key = run_tmpl.format(trial_id=tid, run_index=i + 1) if n > 1 else str(tid)
 
            active = (
                slc[slc[afield] == avalue]
                if afield and avalue is not None and afield in slc.columns
                else slc
            )
            if active.empty:
                active = slc
 
            t0, t1 = active["timestamp"].iloc[0], active["timestamp"].iloc[-1]
            result[key] = {
                "game":       slc,
                "eyetracker": eye_df[(eye_df["timestamp"] >= t0) & (eye_df["timestamp"] <= t1)].reset_index(drop=True),
            }
    return result
 
 
def _victim_count(game_df: pd.DataFrame, victim_cfg: dict) -> int:
    for col in victim_cfg.get("columns", []):
        if col in game_df.columns:
            return int(pd.to_numeric(game_df[col], errors="coerce").max())
    norm = {c.lower(): c for c in game_df.columns}
    for tokens in victim_cfg.get("contains", [["victim", "saved"], ["victim"]]):
        for nk, ok in norm.items():
            if all(t in nk for t in tokens):
                return int(pd.to_numeric(game_df[ok], errors="coerce").max())
    return 0
 
 
def _best_runs(
    trials: dict[str, dict[str, pd.DataFrame]],
    suffix_re: re.Pattern,
    victim_cfg: dict,
) -> dict[str, dict[str, pd.DataFrame]]:
    groups: dict[str, list[str]] = {}
    for tid in trials:
        groups.setdefault(suffix_re.sub("", tid), []).append(tid)
 
    best: dict[str, dict[str, pd.DataFrame]] = {}
    for base, tids in groups.items():
        if len(tids) == 1:
            best[base] = trials[tids[0]]
        else:
            scores = {t: _victim_count(trials[t]["game"], victim_cfg) for t in tids}
            winner = max(scores, key=scores.__getitem__)
            print(f"    {base}: best={winner} (victims={scores[winner]})")
            best[base] = trials[winner]
    return best
 
 
# ---------------------------------------------------------------------------
# I/O
 
def _hdf_ok() -> bool:
    try:
        import tables  # noqa: F401
        return True
    except ImportError:
        return False
 
 
def _save(out: Path, data: dict[str, pd.DataFrame], files: dict[str, str], *, csv: bool, hdf: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if csv:
        data["game"].to_csv(out / files["game_csv"], index=False)
        data["eyetracker"].to_csv(out / files["eyetracker_csv"], index=False)
    if hdf:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            data["game"].to_hdf(out / files["game_h5"], key="game", mode="w")
            data["eyetracker"].to_hdf(out / files["eyetracker_h5"], key="eyetracker", mode="w")
 
 
# ---------------------------------------------------------------------------
# Public API
 
def run_from_config(cfg: dict, root: Path | None = None) -> None:
    collect_subjects(cfg, root)


def process_subject(subject_id: str, cfg: dict, root: Path) -> dict[str, dict]:
    """Process one subject's XDF, save to disk, and return all trial data in memory.

    Returns:
        {trial_id: {"game": DataFrame, "eyetracker": DataFrame}}
        Includes both individual runs and _best entries.
    """
    paths = _paths(cfg, root)
    xcfg  = cfg["xdf"]
    ocfg  = xcfg.get("output", {})
    fcfg  = ocfg.get("files", {})

    files = {
        "game_csv":       fcfg.get("game_csv",       "game.csv"),
        "eyetracker_csv": fcfg.get("eyetracker_csv", "eyetracker.csv"),
        "game_h5":        fcfg.get("game_h5",        "game.h5"),
        "eyetracker_h5":  fcfg.get("eyetracker_h5",  "eyetracker.h5"),
    }
    do_csv    = ocfg.get("write_csv", True)
    do_hdf    = ocfg.get("write_hdf", True) and _hdf_ok()
    run_tmpl  = ocfg.get("run_key_template",     "{trial_id}_run{run_index}")
    sub_tmpl  = ocfg.get("subject_dir_template", "sub-{subject_id}")
    best_tmpl = ocfg.get("best_dir_template",    "{base}_best")
    suffix_re = re.compile(ocfg.get("run_suffix_pattern", r"_run\d+$"))

    xdf_path = next(paths["raw"].rglob(f"*{subject_id}*.xdf"), None)
    if xdf_path is None:
        print(f"  [SKIP] {subject_id}: no XDF under {paths['raw'].relative_to(root)}")
        return {}
    print(f"  [LOAD] {subject_id}: {xdf_path.relative_to(root)}")

    streams, _ = pyxdf.load_xdf(str(xdf_path))
    game_s = _get_stream(streams, xcfg["game_stream"])
    eye_s  = _get_stream(streams, xcfg["eye_stream"])
    if game_s is None:
        raise ValueError(f"Game stream '{xcfg['game_stream']}' not in {xdf_path.name}")
    if eye_s is None:
        raise ValueError(f"Eye stream '{xcfg['eye_stream']}' not in {xdf_path.name}")

    channels = cfg.get("eyetracker", {}).get("channels")
    trials   = _split_by_trial(_parse_game(game_s), _parse_eye(eye_s, channels), xcfg, run_tmpl)
    sub_dir  = paths["intermediate"] / sub_tmpl.format(subject_id=subject_id)

    all_trials: dict[str, dict] = {}

    for tid, data in trials.items():
        d = sub_dir / tid
        _save(d, data, files, csv=do_csv, hdf=do_hdf)
        print(f"    {tid:30s}  game={len(data['game']):>5}  eye={len(data['eyetracker']):>7}  -> {d.relative_to(root)}/")
        all_trials[tid] = data

    for base, data in _best_runs(trials, suffix_re, xcfg.get("victim_count", {})).items():
        best_key = best_tmpl.format(base=base)
        _save(sub_dir / best_key, data, files, csv=do_csv, hdf=do_hdf)
        all_trials[best_key] = data

    return all_trials


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


def load_all_subjects(cfg: dict, root: Path | None = None) -> dict[str, dict[str, dict]]:
    """Load all subjects from data/intermediate (no XDF processing).

    Use this when the XDF step was skipped but downstream steps still need
    the data in memory instead of each re-reading from disk independently.

    Returns:
        {subject_id: {trial_id: {"eyetracker": DataFrame, "game": DataFrame}}}
    """
    import pandas as pd

    if root is None:
        root = Path(__file__).resolve().parents[2]

    intermediate = root / cfg["paths"]["intermediate"]
    sub_tmpl     = cfg["xdf"].get("output", {}).get("subject_dir_template", "sub-{subject_id}")
    all_data: dict[str, dict[str, dict]] = {}

    for subject_id in cfg.get("sub", []):
        sid     = str(subject_id)
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
            # Rename generic ch0, ch1, … columns to configured channel names when
            # the HDF was saved before the channel list was added or matched.
            if channels:
                rename = {f"ch{i}": name for i, name in enumerate(channels) if f"ch{i}" in eye_df.columns}
                if rename:
                    eye_df = eye_df.rename(columns=rename)
            trial_data: dict = {"eyetracker": eye_df}
            game_h5_name = cfg["xdf"].get("output", {}).get("files", {}).get("game_h5", "game.h5")
            game_h5 = trial_dir / game_h5_name
            if game_h5.exists():
                trial_data["game"] = pd.read_hdf(game_h5, key="game")
            trials[trial_dir.name] = trial_data

        if trials:
            print(f"[LOAD] {sid}: {len(trials)} trials from {sub_dir.relative_to(root)}")
        all_data[sid] = trials

    return all_data
 
 