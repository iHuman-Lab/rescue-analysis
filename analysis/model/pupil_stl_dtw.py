"""Dynamic time warping between STL robustness trajectories, expert vs novice.

Uses the per-run robustness traces written by pupil_stl.py. Per condition, every pair of runs
from different participants gets a DTW distance; these are averaged to a participant x
participant matrix D. The novice-to-expert distance is compared with the expert-to-expert
distance, with an exact permutation test over participant labels.

Two variants: "raw" (level and shape) and "shape" (each run z-scored first).
"""

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GROUP_COLORS = {"expert": "#d95f02", "novice": "#1b9e77"}


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """DTW distance with |a_i - b_j| cost, normalized by (len(a) + len(b))."""
    n, m = len(a), len(b)
    cost = np.abs(a[:, None] - b[None, :])
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        prev, cur = D[i - 1], D[i]
        for j in range(1, m + 1):
            cur[j] = cost[i - 1, j - 1] + min(prev[j], prev[j - 1], cur[j - 1])
    return float(D[n, m] / (n + m))


def run_series(traces: pd.DataFrame, bin_s: int, znorm: bool) -> dict:
    """{(participant, trial, run): robustness series, block-averaged to bin_s seconds}."""
    out = {}
    for key, g in traces.groupby(["participant", "trial", "run"]):
        r = g.sort_values("t_s").dropna(subset=["robustness"])
        if len(r) < 2 * bin_s:
            continue
        x = r["robustness"].to_numpy()
        x = x[: len(x) // bin_s * bin_s].reshape(-1, bin_s).mean(axis=1)
        out[key] = (x - x.mean()) / x.std() if znorm else x
    return out


def participant_matrix(series: dict, info: pd.DataFrame) -> pd.DataFrame:
    """Participant x participant mean DTW distance between runs of different participants."""
    keys = list(series)
    sums: dict = {}
    for k1, k2 in combinations(keys, 2):
        if k1[0] == k2[0]:
            continue
        pair = tuple(sorted((k1[0], k2[0])))
        sums.setdefault(pair, []).append(dtw(series[k1], series[k2]))
    parts = sorted({k[0] for k in keys})
    D = pd.DataFrame(np.nan, index=parts, columns=parts)
    for (p, q), v in sums.items():
        D.loc[p, q] = D.loc[q, p] = np.mean(v)
    return D


def group_stat(D: np.ndarray, is_exp: np.ndarray) -> tuple[float, float, float]:
    """(novice->expert, expert->expert, difference) mean distances, excluding self-pairs."""
    exp_idx = np.flatnonzero(is_exp)
    nov_idx = np.flatnonzero(~is_exp)
    n2e = np.nanmean([np.nanmean(D[i, exp_idx]) for i in nov_idx])
    e2e = np.nanmean([np.nanmean(D[i, exp_idx[exp_idx != i]]) for i in exp_idx])
    return n2e, e2e, n2e - e2e


def analyze(traces: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    pcfg = cfg.get("pupil_stl", {})
    bin_s = pcfg.get("dtw_bin_s", 5)
    info = traces.groupby("participant")["expertise"].first()
    rows, mats = [], {}
    for variant, znorm in (("raw", False), ("shape", True)):
        for cond, d in traces.groupby("condition"):
            series = run_series(d, bin_s, znorm)
            D = participant_matrix(series, info)
            is_exp = (info.loc[D.index] == "expert").to_numpy()
            n2e, e2e, diff = group_stat(D.to_numpy(), is_exp)
            perm = []
            for idx in combinations(range(len(D)), int(is_exp.sum())):
                m = np.zeros(len(D), bool)
                m[list(idx)] = True
                perm.append(group_stat(D.to_numpy(), m)[2])
            p = float((np.array(perm) >= diff - 1e-12).mean())  # one-sided: novices farther
            rows.append((variant, cond, int((~is_exp).sum()), int(is_exp.sum()), n2e, e2e, diff, p))
            mats[(variant, cond)] = (D, is_exp)
    cols = ["variant", "condition", "n_novice", "n_expert", "novice_to_expert", "expert_to_expert", "difference", "p_perm"]
    return pd.DataFrame(rows, columns=cols), mats


def plot(mats: dict, stats: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for j, cond in enumerate(("no_llm", "llm")):
        D, is_exp = mats[("raw", cond)]
        order = np.argsort(~is_exp, kind="stable")  # experts first
        ax = axes[0, j]
        im = ax.imshow(D.to_numpy()[np.ix_(order, order)], cmap="viridis")
        labels = [f"{D.index[i]}{'*' if is_exp[i] else ''}" for i in order]
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(f"{'no LLM' if cond == 'no_llm' else 'LLM'}: DTW distance (* = expert)")
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, j]
        Dv = D.to_numpy()
        exp_idx = np.flatnonzero(is_exp)
        for k, (grp, color) in enumerate(GROUP_COLORS.items()):
            idx = np.flatnonzero(is_exp == (grp == "expert"))
            vals = [np.nanmean(Dv[i, exp_idx[exp_idx != i]]) for i in idx]
            ax.scatter(np.full(len(vals), k) + np.linspace(-0.08, 0.08, len(vals)), vals, color=color, s=30)
            ax.hlines(np.mean(vals), k - 0.2, k + 0.2, color=color)
        row = stats[(stats.variant == "raw") & (stats.condition == cond)].iloc[0]
        ax.set_xticks([0, 1], [f"expert (n={row.n_expert})", f"novice (n={row.n_novice})"])
        ax.set_ylabel("mean DTW distance to experts")
        ax.set_title(f"permutation p = {row.p_perm:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def run(cfg: dict) -> pd.DataFrame:
    processed = ROOT / cfg["paths"]["processed"]
    traces = pd.read_csv(processed / "pupil_stl_traces.csv")
    stats, mats = analyze(traces, cfg)
    stats.to_csv(processed / "pupil_stl_dtw.csv", index=False)
    print(stats.round(4).to_string(index=False))
    plot(mats, stats, processed / "pupil_stl_dtw.png")
    print(f"\nfigure -> {processed / 'pupil_stl_dtw.png'}")
    return stats
