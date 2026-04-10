
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

ROOT   = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "config_analysis.yml"

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

SCREEN_W   = cfg["eyetracker"]["screen_w"]
SCREEN_H   = cfg["eyetracker"]["screen_h"]
AOIS       = cfg.get("aoi", [])
PROCESSED  = ROOT / cfg["paths"]["processed"]
AOI_COLORS = {"game_area": "#4C72B0", "info_panel": "#DD8452", "chat_panel": "#55A868"}
SCREENSHOT  = ROOT / "analysis" / "game_screenshot.png"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fixations(subjects, trial_filter=None):
    """Load fixations_aoi.csv for given subjects, optionally filtered by trial."""
    frames = []
    for sid in subjects:
        sub_dir = PROCESSED / f"sub-{sid}"
        if not sub_dir.exists():
            continue
        for trial_dir in sorted(sub_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            if trial_filter and trial_dir.name != trial_filter:
                continue
            f = trial_dir / "fixations_aoi.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df["subject"] = sid
            df["trial"]   = trial_dir.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_heatmap(fix_df, sigma=30):
    """Build a duration-weighted heatmap using scipy gaussian_filter.
    Values below the mean of non-zero cells are masked as NaN (transparent).
    """
    canvas = np.zeros((SCREEN_H, SCREEN_W), dtype=float)
    valid  = fix_df.dropna(subset=["x", "y"])
    for _, row in valid.iterrows():
        xi = int(np.clip(row["x"], 0, SCREEN_W - 1))
        yi = int(np.clip(row["y"], 0, SCREEN_H - 1))
        canvas[yi, xi] += float(row.get("duration_ms", 1.0))
    hmap = gaussian_filter(canvas, sigma=sigma)
    # mask near-zero values so background shows through
    threshold = np.percentile(hmap[hmap > 0], 30) if np.any(hmap > 0) else 0
    hmap[hmap < threshold] = np.nan
    return hmap


def _hot_transparent():
    """'hot' colormap with NaN rendered as fully transparent."""
    cmap = plt.cm.hot.copy()
    cmap.set_bad(alpha=0)
    return cmap


def save_figure(out_path: Path):
    """Save current figure as both PNG and PDF."""
    plt.savefig(out_path, dpi=150)
    # rasterize all image artists so the PDF matches the PNG visually
    for ax in plt.gcf().axes:
        for artist in ax.get_images():
            artist.set_rasterized(True)
    plt.savefig(out_path.with_suffix(".pdf"), dpi=150)


def draw_aois(ax, skip=("game_area",)):
    """Draw AOI dashed boundaries, skipping any names in `skip`."""
    for aoi in AOIS:
        if aoi["name"] in skip:
            continue
        color = AOI_COLORS.get(aoi["name"], "white")
        rect  = mpatches.Rectangle(
            (aoi["x_min"], aoi["y_min"]),
            aoi["x_max"] - aoi["x_min"],
            aoi["y_max"] - aoi["y_min"],
            linewidth=2, edgecolor=color, facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(aoi["x_min"] + 6, aoi["y_min"] + 20, aoi["name"],
                color=color, fontsize=9, fontweight="bold")


def plot_heatmap(fix_df, title, out_path):
    """Full-screen heatmap with game screenshot background and panel AOI outlines."""
    if fix_df.empty:
        print(f"  [SKIP] no fixations for: {title}")
        return

    hmap = make_heatmap(fix_df)

    fig, ax = plt.subplots(figsize=(12, 7))

    # background screenshot
    if SCREENSHOT.exists():
        import matplotlib.image as mpimg
        bg = mpimg.imread(str(SCREENSHOT))
        ax.imshow(bg, origin="upper", extent=[0, SCREEN_W, SCREEN_H, 0],
                  aspect="auto", zorder=0)
        ax.imshow(hmap, cmap=_hot_transparent(), origin="upper",
                  extent=[0, SCREEN_W, SCREEN_H, 0], aspect="auto",
                  alpha=0.55, zorder=1)
    else:
        ax.imshow(hmap, cmap=_hot_transparent(), origin="upper",
                  extent=[0, SCREEN_W, SCREEN_H, 0], aspect="auto")

    # panel outlines only — game_area wall skipped
    draw_aois(ax, skip=("game_area",))

    ax.set_xlim(0, SCREEN_W)
    ax.set_ylim(SCREEN_H, 0)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    n   = len(fix_df)
    dur = fix_df["duration_ms"].sum() / 1000
    ax.text(10, SCREEN_H - 15, f"n={n} fixations  |  total={dur:.1f}s",
            color="white", fontsize=9)

    plt.tight_layout()
    save_figure(out_path)
    plt.close()
    print(f"  Saved -> {out_path.relative_to(ROOT)}  +  .pdf")


def plot_heatmap_by_aoi(fix_df, title, out_path):
    """One subplot per AOI, heatmap cropped to that AOI's region."""
    if fix_df.empty:
        print(f"  [SKIP] no fixations for: {title}")
        return

    full_hmap = make_heatmap(fix_df)
    n_aois    = len(AOIS)

    fig, axes = plt.subplots(1, n_aois, figsize=(6 * n_aois, 5))
    if n_aois == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)

    for ax, aoi in zip(axes, AOIS):
        x0, x1 = aoi["x_min"], aoi["x_max"]
        y0, y1 = aoi["y_min"], aoi["y_max"]

        # crop heatmap to AOI bounding box
        crop = full_hmap[y0:y1, x0:x1]
        color = AOI_COLORS.get(aoi["name"], "white")

        ax.imshow(crop, cmap=_hot_transparent(), origin="upper",
                  extent=[x0, x1, y1, y0], aspect="auto")
        ax.set_title(aoi["name"], color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        # fixations in this AOI
        aoi_fix = fix_df[fix_df["aoi"] == aoi["name"]]
        n   = len(aoi_fix)
        pct = round(n / len(fix_df) * 100, 1) if len(fix_df) > 0 else 0
        dur = aoi_fix["duration_ms"].sum() / 1000
        ax.text(x0 + 4, y1 - 4, f"n={n} ({pct}%)  {dur:.1f}s",
                color="yellow", fontsize=8, va="bottom")

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.tight_layout()
    save_figure(out_path)
    plt.close()
    print(f"  Saved -> {out_path.relative_to(ROOT)}  +  .pdf")


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def load_transitions(subjects, trial_filter=None):
    """Aggregate AOI transition matrices across subjects/trials."""
    aoi_names = [a["name"] for a in AOIS]
    total     = pd.DataFrame(0, index=aoi_names, columns=aoi_names)
    for sid in subjects:
        sub_dir = PROCESSED / f"sub-{sid}"
        if not sub_dir.exists():
            continue
        for trial_dir in sorted(sub_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            if trial_filter and trial_dir.name != trial_filter:
                continue
            f = trial_dir / "aoi_transitions.csv"
            if not f.exists():
                continue
            mat = pd.read_csv(f, index_col=0)
            # align to expected AOI labels
            mat = mat.reindex(index=aoi_names, columns=aoi_names, fill_value=0)
            total += mat
    return total


def plot_transitions(matrix, title, out_path, normalize=True):
    """Plot AOI transition matrix as an annotated heatmap."""
    if matrix.values.sum() == 0:
        print(f"  [SKIP] no transitions for: {title}")
        return

    if normalize:
        row_sums = matrix.sum(axis=1).replace(0, 1)
        display  = matrix.div(row_sums, axis=0)
        fmt, cbar_label = ".2f", "Transition probability"
    else:
        display  = matrix
        fmt, cbar_label = "d", "Transition count"

    _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(display.values, cmap="YlOrRd", vmin=0,
                   vmax=1 if normalize else None, aspect="auto")

    labels = list(matrix.index)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("To AOI")
    ax.set_ylabel("From AOI")
    ax.set_title(title, fontsize=12)

    # annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = display.values[i, j]
            txt = f"{val:{fmt}}" if fmt == ".2f" else str(int(val))
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if val < 0.6 else "white", fontsize=10)

    plt.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    save_figure(out_path)
    plt.close()
    print(f"  Saved -> {out_path.relative_to(ROOT)}  +  .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub",      default=None, help="Subject ID, e.g. P001")
    parser.add_argument("--trial",    default=None, help="Trial folder name")
    parser.add_argument("--by-trial", action="store_true", help="Separate heatmap per trial")
    parser.add_argument("--by-aoi",      action="store_true", help="Show heatmap cropped per AOI")
    parser.add_argument("--transitions", action="store_true", help="Plot AOI transition matrix")
    args = parser.parse_args()

    subjects   = [args.sub] if args.sub else [str(s) for s in cfg.get("sub", [])]
    out_dir    = ROOT / "analysis" / "heatmaps"
    out_dir.mkdir(exist_ok=True)

    if args.transitions:
        if args.by_trial:
            for sid in subjects:
                fix_all = load_fixations([sid])
                if fix_all.empty:
                    continue
                for trial_name in fix_all["trial"].unique():
                    mat = load_transitions([sid], trial_filter=trial_name)
                    plot_transitions(mat, f"{sid} — {trial_name}",
                                     out_dir / f"transitions_{sid}_{trial_name}.png")
        else:
            mat   = load_transitions(subjects, trial_filter=args.trial)
            label = "_".join(subjects) if len(subjects) <= 3 else f"{len(subjects)}subjects"
            plot_transitions(mat, "AOI transitions", out_dir / f"transitions_{label}.png")
        return

    plot_fn    = plot_heatmap_by_aoi if args.by_aoi else plot_heatmap
    aoi_suffix = "_byAOI" if args.by_aoi else ""

    if args.by_trial:
        for sid in subjects:
            fix_all = load_fixations([sid])
            if fix_all.empty:
                continue
            for trial_name, grp in fix_all.groupby("trial"):
                fname = f"{sid}_{trial_name}{aoi_suffix}.png"
                plot_fn(grp, f"{sid} — {trial_name}", out_dir / fname)
    else:
        fix_all = load_fixations(subjects, trial_filter=args.trial)
        label   = "_".join(subjects) if len(subjects) <= 3 else f"{len(subjects)}subjects"
        if args.trial:
            label += f"_{args.trial}"
        title   = f"Fixation heatmap  ({', '.join(subjects) if len(subjects)<=3 else str(len(subjects))+' subjects'})"
        if args.trial:
            title += f"\n{args.trial}"
        plot_fn(fix_all, title, out_dir / f"heatmap_{label}{aoi_suffix}.png")


if __name__ == "__main__":
    main()
