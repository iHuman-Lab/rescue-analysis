import pandas as pd

DEFAULT_OFFSCREEN_LABEL: str = "offscreen"


def label_fixations(fix_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    """Vectorized AOI assignment based on bounding boxes."""
    df = fix_df.copy()
    df["aoi"] = DEFAULT_OFFSCREEN_LABEL
    for aoi in aois:
        mask = df["x"].between(aoi["x_min"], aoi["x_max"]) & df["y"].between(
            aoi["y_min"], aoi["y_max"]
        )
        df.loc[mask, "aoi"] = aoi["name"]
    return df


def aoi_transition_matrix(fix_aoi_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    """Calculate transition counts between sequential AOI fixations."""
    labels = [a["name"] for a in aois]
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    sequence = fix_aoi_df[fix_aoi_df["aoi"] != DEFAULT_OFFSCREEN_LABEL]["aoi"].tolist()

    for src, dst in zip(sequence[:-1], sequence[1:]):
        if src in matrix.index and dst in matrix.columns:
            matrix.loc[src, dst] += 1
    return matrix
