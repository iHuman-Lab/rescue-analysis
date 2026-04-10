"""Quick visualization of AOI panels defined in config_analysis.yml."""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import yaml

ROOT   = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "config_analysis.yml"

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

screen_w = cfg["eyetracker"]["screen_w"]
screen_h = cfg["eyetracker"]["screen_h"]
aois     = cfg.get("aoi", [])

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, screen_w)
ax.set_ylim(screen_h, 0)          # y-axis: top = 0 (screen convention)
ax.set_aspect("equal")
ax.set_facecolor("#f0f0f0")
ax.set_title(f"AOI panels  ({screen_w}×{screen_h} screen)", fontsize=13)
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")

for i, aoi in enumerate(aois):
    x      = aoi["x_min"]
    y      = aoi["y_min"]
    w      = aoi["x_max"] - aoi["x_min"]
    h      = aoi["y_max"] - aoi["y_min"]
    color  = COLORS[i % len(COLORS)]
    rect   = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=2",
        linewidth=2, edgecolor=color,
        facecolor=color, alpha=0.35,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, aoi["name"],
            ha="center", va="center", fontsize=11, color=color, fontweight="bold")
    # label dimensions in corner
    ax.text(x + 4, y + 14, f"({x},{y})–({aoi['x_max']},{aoi['y_max']})",
            fontsize=7, color=color, va="top")

plt.tight_layout()
out = ROOT / "analysis" / "aoi_panels.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()