"""HDF5 and CSV I/O helpers."""

import warnings
from pathlib import Path

import pandas as pd


def _hdf_ok() -> bool:
    try:
        import tables  # noqa: F401

        return True
    except ImportError:
        return False


def _save(
    out: Path,
    data: dict[str, pd.DataFrame],
    files: dict[str, str],
    *,
    csv: bool,
    hdf: bool,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if csv:
        data["game"].to_csv(out / files["game_csv"], index=False)
        data["eyetracker"].to_csv(out / files["eyetracker_csv"], index=False)
    if hdf:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            data["game"].to_hdf(out / files["game_h5"], key="game", mode="w")
            data["eyetracker"].to_hdf(
                out / files["eyetracker_h5"], key="eyetracker", mode="w"
            )
