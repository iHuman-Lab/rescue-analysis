"""Pupil-size workload trajectories and probabilistic STL robustness, expert vs novice.

Pipeline per run (data.h5 eye_tracking):
  1. pupil diameter -> 10 Hz bins (missing samples dropped, short gaps interpolated)
  2. rolling pupil SD (window_s, step 1 s) with a within-window bootstrap range
  3. calibrated workload probability: logistic ensemble (cluster-bootstrapped over
     participants, leave-one-participant-out) mapping log pupil SD -> P(high workload),
     label = LLM condition. Gives the interval trace p_t = [p_lo, p_hi] (Thrust 1b).
  4. interval predicate semantics (Thrust 2a): mu = "workload not high" has satisfaction
     interval [1 - p_hi, 1 - p_lo]; only the conservative lower bound 1 - p_hi is kept.
  5. phi = Historically_[0,W] mu, monitored with a log-sum-exp soft-min over the past W
     seconds -> probabilistic robustness r_t in [0, 1] (near 0 likely violation, near 1
     likely satisfaction). Spec threshold tau: phi is satisfied iff max p_hi <= tau,
     i.e. r_t >= 1 - tau (checked with the exact min).
"""

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import expit, logsumexp
from scipy.stats import mannwhitneyu, rankdata

ROOT = Path(__file__).resolve().parents[2]
BIN_S = 0.1
MAX_GAP_BINS = 3  # interpolate gaps up to 0.3 s (blinks)
METRICS = ("mean_sd", "mean_robustness", "min_robustness", "frac_violated")
GROUP_COLORS = {"expert": "#d95f02", "novice": "#1b9e77"}


def included_subjects(cfg: dict) -> set[str]:
    """Participants in cfg["sub"] minus pupil_stl.exclude_participants."""
    drop = {str(x) for x in cfg.get("pupil_stl", {}).get("exclude_participants", [])}
    return {str(x) for x in cfg["sub"]} - drop


def pupil_bins(eye: pd.DataFrame, pupil_col: str, missing: float) -> np.ndarray:
    """Mean valid pupil diameter per BIN_S bin, blinks interpolated."""
    t = eye["timestamp"].to_numpy(dtype=float)
    t = t - t[0]
    pupil = pd.to_numeric(eye[pupil_col], errors="coerce").replace(missing, np.nan)
    idx = (t / BIN_S).astype(int)
    series = pupil.groupby(idx).mean().reindex(range(idx.max() + 1))
    return series.interpolate(limit=MAX_GAP_BINS, limit_area="inside").to_numpy()


FEATURES = {"sd": "sd", "mean": "pm", "slope": "sl"}  # name -> column prefix


def rolling_features(x: np.ndarray, window_s: float, rng, n_boot: int) -> pd.DataFrame:
    """Pupil SD, mean and slope (mm/s) with bootstrap [lo, hi], in windows ending every second."""
    w = int(window_s / BIN_S)
    step = int(1 / BIN_S)
    rows = []
    for end in range(w, len(x) + 1, step):
        full = x[end - w : end]
        ok = ~np.isnan(full)
        if ok.sum() < 0.7 * w:
            continue
        seg, tt = full[ok], np.arange(w)[ok] * BIN_S
        idx = rng.integers(0, len(seg), (n_boot, len(seg)))
        xs, ts = seg[idx], tt[idx]
        tvar = np.maximum(ts.var(axis=1), 1e-12)
        boots = {
            "sd": xs.std(axis=1, ddof=1),
            "pm": xs.mean(axis=1),
            "sl": ((ts * xs).mean(axis=1) - ts.mean(axis=1) * xs.mean(axis=1)) / tvar,
        }
        point = {"sd": seg.std(ddof=1), "pm": seg.mean(), "sl": np.polyfit(tt, seg, 1)[0]}
        row = [end * BIN_S]
        for k in ("sd", "pm", "sl"):
            lo, hi = np.percentile(boots[k], [10, 90])
            row += [point[k], lo, hi]
        rows.append(row)
    cols = ["t"] + [f"{k}{suf}" for k in ("sd", "pm", "sl") for suf in ("", "_lo", "_hi")]
    return pd.DataFrame(rows, columns=cols)


def historically_min(x: np.ndarray, n: int) -> np.ndarray:
    """Sliding min over the last n samples (STL Always over the past window)."""
    return pd.Series(x).rolling(n, min_periods=1).min().to_numpy()


def build_windows(cfg: dict) -> pd.DataFrame:
    """Per-run rolling pupil features (no calibration yet)."""
    eye_cfg = cfg["eyetracker"]
    pcfg = cfg.get("pupil_stl", {})
    rng = np.random.default_rng(pcfg.get("seed", 0))
    expertise = cfg["expertise"]
    cond_map = cfg["analysis"]["condition_by_category"]
    subjects = included_subjects(cfg)

    frames = []
    with pd.HDFStore(str(ROOT / cfg["paths"]["processed"] / "data.h5"), mode="r") as store:
        for key in sorted(k for k in store.keys() if k.endswith("/eye_tracking")):
            _, sid, trial, run, _ = key.split("/")
            if sid not in subjects or trial not in cond_map:
                continue
            eye = store[key]
            if eye.empty:
                print(f"skip {sid}/{trial}/{run}: no eye samples")
                continue
            x = pupil_bins(eye, eye_cfg["pupil_col"], eye_cfg["missing"])
            tr = rolling_features(x, pcfg.get("window_s", 5), rng, pcfg.get("n_boot", 100))
            tr["t_abs"] = float(eye["timestamp"].iloc[0]) + tr["t"]  # same LSL clock as the game stream
            tr = tr.assign(
                participant=sid, trial=trial, run=run,
                condition=cond_map[trial], expertise=expertise[sid],
            )
            frames.append(tr)
    return pd.concat(frames, ignore_index=True)


def build_traces(cfg: dict, windows: pd.DataFrame | None = None) -> pd.DataFrame:
    pcfg = cfg.get("pupil_stl", {})
    if windows is None:
        windows = build_windows(cfg)
    df = calibrate(windows.copy(), pcfg, np.random.default_rng(pcfg.get("seed", 0)))
    return add_robustness(df, int(pcfg.get("spec_window_s", 30)), pcfg.get("beta", 50.0))


def add_robustness(df: pd.DataFrame, n_hist: int, beta: float) -> pd.DataFrame:
    """Monitor phi = Historically_[0, n_hist] (workload not high); 1 trace sample per second."""
    df = df.copy()
    df["t_s"] = df["t"].round().astype(int)
    df = df.sort_values(["participant", "trial", "run", "t_s"])
    mon = df.groupby(["participant", "trial", "run"])
    df["robustness"] = np.concatenate([monitor(g, n_hist, beta) for _, g in mon])
    df["robustness_exact"] = np.concatenate([monitor(g, n_hist, beta, soft=False) for _, g in mon])
    return df


def fit_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Weighted 1-D logistic regression by Newton/IRLS. Returns (intercept, slope)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    pen = np.diag([0.0, ridge])
    for _ in range(25):
        p = expit(X @ beta)
        grad = X.T @ (w * (p - y)) + pen @ beta
        hess = (X * (w * p * (1 - p))[:, None]).T @ X + pen
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def calibrate(df: pd.DataFrame, pcfg: dict, rng, feat: str = "sd", norm: str = "raw") -> pd.DataFrame:
    """Leave-one-participant-out logistic ensemble -> p, p_lo, p_hi (monotone in the feature).

    feat: pupil feature behind the predicate (sd | mean | slope).
    norm: raw = one global standardization; subject = each participant standardized against
    their own no_llm windows (per-subject shift and scale).
    """
    n_models = pcfg.get("n_models", 50)
    c = FEATURES[feat]
    x = df[[c, f"{c}_lo", f"{c}_hi"]].to_numpy()
    if feat == "sd":
        x = np.log(x)
    parts = df["participant"].to_numpy()
    if norm == "subject":
        base = (df["condition"] == "no_llm").to_numpy()
        for pid in np.unique(parts):
            rows = parts == pid
            ref = x[rows & base, 0] if (rows & base).any() else x[rows, 0]
            x[rows] = (x[rows] - ref.mean()) / ref.std()
    else:
        x = (x - x[:, 0].mean()) / x[:, 0].std()
    y = (df["condition"] == "llm").to_numpy(dtype=float)
    by_p = {p: np.flatnonzero(parts == p) for p in np.unique(parts)}

    out = np.zeros((len(df), 3))
    for held, held_idx in by_p.items():
        train = [p for p in by_p if p != held]
        preds = []
        for _ in range(n_models):
            idx = np.concatenate([by_p[p] for p in rng.choice(train, len(train))])
            yb = y[idx]
            w = np.where(yb == 1, 0.5 / yb.mean(), 0.5 / (1 - yb.mean()))
            b0, b1 = fit_logistic(x[idx, 0], yb, w)
            preds.append(expit(b0 + b1 * x[held_idx]))  # (n, 3): sd, sd_lo, sd_hi
        preds = np.stack(preds)  # (B, n, 3)
        lo = np.minimum(preds[..., 1], preds[..., 2])
        hi = np.maximum(preds[..., 1], preds[..., 2])
        p = preds[..., 0].mean(axis=0)
        out[held_idx, 0] = p
        out[held_idx, 1] = np.minimum(np.percentile(lo, 10, axis=0), p)
        out[held_idx, 2] = np.maximum(np.percentile(hi, 90, axis=0), p)
    df[["p", "p_lo", "p_hi"]] = out
    return df


def monitor(g: pd.DataFrame, n: int, beta: float, soft: bool = True) -> np.ndarray:
    """Historically_[0,n] of the predicate lower bound (1 - p_hi), 1 sample per second.

    soft: log-sum-exp soft-min (differentiable, biased low by <= ln(n)/beta);
    otherwise exact min. NaN until a full window of history exists.
    """
    sat = pd.Series(1.0 - g["p_hi"].to_numpy(), index=g["t_s"].to_numpy())
    sat = sat[~sat.index.duplicated()].reindex(range(sat.index.min(), sat.index.max() + 1))
    a = sat.to_numpy()
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        win = sliding_window_view(a, n)
        valid = ~np.isnan(win)
        filled = np.where(valid, win, 10.0)  # missing windows drop out of the min
        r = (-logsumexp(-beta * filled, axis=1) / beta) if soft else filled.min(axis=1)
        r[valid.mean(axis=1) < 0.5] = np.nan
        out[n - 1 :] = r
    # map back onto the (possibly gappy) original rows
    return pd.Series(out, index=sat.index).reindex(g["t_s"].to_numpy()).to_numpy()


def run_summary(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """One row per run: fraction of monitored time violating phi (exact min), robustness stats."""
    g = df.groupby(["participant", "expertise", "condition", "trial", "run"])
    return g.agg(
        mean_sd=("sd", "mean"),
        mean_p=("p", "mean"),
        mean_robustness=("robustness", "mean"),
        min_robustness=("robustness", "min"),
        frac_violated=("robustness_exact", lambda r: float((r[r.notna()] < 1 - tau).mean())),
        duration_s=("t", "max"),
    ).reset_index()


def compare_groups(runs: pd.DataFrame) -> pd.DataFrame:
    """Expert vs novice per condition on participant-level means (avoids pseudo-replication)."""
    out = []
    for cond, d in runs.groupby("condition"):
        by_p = d.groupby(["participant", "expertise"], as_index=False).mean(numeric_only=True)
        for metric in METRICS:
            e = by_p.loc[by_p.expertise == "expert", metric]
            n = by_p.loc[by_p.expertise == "novice", metric]
            p = mannwhitneyu(e, n).pvalue if len(e) and len(n) else np.nan
            out.append((cond, metric, len(e), len(n), e.mean(), n.mean(), p))
    # within-participant LLM effect, expert vs novice
    by_pc = runs.groupby(["participant", "expertise", "condition"], as_index=False).mean(numeric_only=True)
    wide = by_pc.pivot(index=["participant", "expertise"], columns="condition", values=list(METRICS))
    for metric in METRICS:
        delta = (wide[(metric, "llm")] - wide[(metric, "no_llm")]).dropna().reset_index(name="d")
        e = delta.loc[delta.expertise == "expert", "d"]
        n = delta.loc[delta.expertise == "novice", "d"]
        p = mannwhitneyu(e, n).pvalue if len(e) and len(n) else np.nan
        out.append(("llm_minus_no_llm", metric, len(e), len(n), e.mean(), n.mean(), p))
    return pd.DataFrame(
        out, columns=["condition", "metric", "n_expert", "n_novice", "expert_mean", "novice_mean", "p_mwu"]
    )


def group_curve(d: pd.DataFrame, col: str, min_runs: int, rng, n_boot: int = 200):
    """Mean over participants per time, 95% bootstrap CI across participants.

    Per-participant mean first (runs averaged), so each participant counts once.
    """
    per_p = d.groupby(["participant", "t"])[col].mean().unstack("t")
    per_p = per_p.loc[:, per_p.count() >= min_runs]
    mean = per_p.mean()
    boots = np.array([
        per_p.iloc[rng.integers(0, len(per_p), len(per_p))].mean().to_numpy()
        for _ in range(n_boot)
    ])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return mean.index.to_numpy(), mean.to_numpy(), lo, hi


def plot(df: pd.DataFrame, cfg: dict, out: Path) -> None:
    pcfg = cfg.get("pupil_stl", {})
    tau = pcfg.get("tau", 0.75)
    min_p = pcfg.get("min_participants", 3)
    rng = np.random.default_rng(pcfg.get("seed", 0))
    conds = ["no_llm", "llm"]
    rows = [("sd", "Pupil SD (mm)"), ("p", "P(high workload), calibrated"),
            ("robustness", f"probabilistic robustness r_t\nH[0,{pcfg.get('spec_window_s', 30)}s] (workload not high)")]
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 9), sharex="col", sharey="row")
    for j, cond in enumerate(conds):
        for i, (col, label) in enumerate(rows):
            ax = axes[i, j]
            for grp, color in GROUP_COLORS.items():
                d = df[(df.condition == cond) & (df.expertise == grp)]
                if d.empty:
                    continue
                n = d.participant.nunique()
                t, m, lo, hi = group_curve(d, col, min_p, rng)
                ax.plot(t, m, color=color, label=f"{grp} (n={n})")
                ax.fill_between(t, lo, hi, color=color, alpha=0.2)
            if col == "robustness":
                ax.axhline(1 - tau, color="k", lw=0.8, ls="--")  # r_t below this: tau exceeded
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title("no LLM" if cond == "no_llm" else "LLM")
            if i == len(rows) - 1:
                ax.set_xlabel("time since run start (s)")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


PICKUP_ACTION = 3
PICKUP_SUCCESS_REWARD = 10.0  # +10 = victim picked up; -20 / -10 / 0 = wrong or failed pickup


def pickup_events(cfg: dict) -> pd.DataFrame:
    """One row per pickup attempt: time (game clock) and whether a victim was picked up."""
    cond_map = cfg["analysis"]["condition_by_category"]
    subjects = included_subjects(cfg)
    rows = []
    with pd.HDFStore(str(ROOT / cfg["paths"]["processed"] / "data.h5"), mode="r") as store:
        for key in sorted(k for k in store.keys() if k.endswith("/game")):
            _, sid, trial, run, _ = key.split("/")
            if sid not in subjects or trial not in cond_map:
                continue
            g = store[key]
            g = g[g["action"] == PICKUP_ACTION]
            # several render frames per step; the reward lands on a later frame of the step
            for _, step in g.groupby("step_count"):
                nonzero = step["reward"][step["reward"] != 0]
                reward = float(nonzero.iloc[0]) if len(nonzero) else 0.0
                rows.append((sid, trial, run, float(step["timestamp"].min()), reward))
    ev = pd.DataFrame(rows, columns=["participant", "trial", "run", "t_abs", "reward"])
    ev["picked"] = ev["reward"] == PICKUP_SUCCESS_REWARD
    return ev


def event_locked(df: pd.DataFrame, ev: pd.DataFrame, pre: int, post: int, flag: str = "picked") -> pd.DataFrame:
    """p_t and r_t on a 1 Hz grid from -pre to +post seconds around each event.

    flag: boolean column of ev carried through (picked for pickups, alt for help requests).
    """
    offsets = np.arange(-pre, post + 1)
    traces = {k: g.sort_values("t_abs") for k, g in df.groupby(["participant", "trial", "run"])}
    rows = []
    for e in ev.itertuples():
        tr = traces.get((e.participant, e.trial, e.run))
        if tr is None:
            continue
        t = tr["t_abs"].to_numpy()
        grid = e.t_abs + offsets
        inside = (grid >= t[0]) & (grid <= t[-1])
        out = {"p": np.full(len(grid), np.nan), "robustness": np.full(len(grid), np.nan)}
        for col in out:
            ok = inside & ~np.isnan(np.interp(grid, t, tr[col].to_numpy()))
            out[col][ok] = np.interp(grid, t, tr[col].to_numpy())[ok]
        rows.append(pd.DataFrame({
            "participant": e.participant, "trial": e.trial, "run": e.run,
            "condition": tr["condition"].iloc[0], "expertise": tr["expertise"].iloc[0],
            flag: getattr(e, flag), "event_t": e.t_abs, "t": offsets, **out,
        }))
    return pd.concat(rows, ignore_index=True)


def event_stats(locked: pd.DataFrame, flag: str = "picked", names: tuple = ("picked", "not_picked")) -> pd.DataFrame:
    """Robustness at the event moment (t = 0): expert vs novice, by event type and condition."""
    at = locked[locked["t"] == 0].dropna(subset=["robustness"])
    out = []
    for cond, d in [("all", at)] + list(at.groupby("condition")):
        by_p = d.groupby(["participant", "expertise", flag], as_index=False)["robustness"].mean()
        for val, lab in ((True, names[0]), (False, names[1])):
            b = by_p[by_p[flag] == val]
            e = b.loc[b.expertise == "expert", "robustness"]
            n = b.loc[b.expertise == "novice", "robustness"]
            p = mannwhitneyu(e, n).pvalue if len(e) > 1 and len(n) > 1 else np.nan
            out.append((cond, lab, len(e), len(n), e.mean(), n.mean(), p))
    return pd.DataFrame(
        out, columns=["condition", "outcome", "n_expert", "n_novice", "expert_mean", "novice_mean", "p_mwu"]
    )


def plot_event_locked(
    locked: pd.DataFrame, cfg: dict, out: Path, flag: str = "picked",
    titles: tuple = ("victim picked up", "not picked up (wrong / failed)"), xlabel: str = "time relative to pickup (s)",
) -> None:
    pcfg = cfg.get("pupil_stl", {})
    tau = pcfg.get("tau", 0.75)
    rng = np.random.default_rng(pcfg.get("seed", 0))
    rows = [("p", "P(high workload), calibrated"),
            ("robustness", f"probabilistic robustness r_t\nH[0,{pcfg.get('spec_window_s', 30)}s] (workload not high)")]
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 6.5), sharex=True, sharey="row")
    for j, (val, title) in enumerate(((True, titles[0]), (False, titles[1]))):
        for i, (col, label) in enumerate(rows):
            ax = axes[i, j]
            for grp, color in GROUP_COLORS.items():
                d = locked[(locked[flag] == val) & (locked.expertise == grp)].dropna(subset=[col])
                if d.empty:
                    continue
                n = d.participant.nunique()
                t, m, lo, hi = group_curve(d, col, pcfg.get("min_participants", 3), rng)
                ax.plot(t, m, color=color, label=f"{grp} (n={n})")
                ax.fill_between(t, lo, hi, color=color, alpha=0.2)
            ax.axvline(0, color="k", lw=0.8, ls=":")
            if col == "robustness":
                ax.axhline(1 - tau, color="k", lw=0.8, ls="--")
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(title)
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def window_sweep(df: pd.DataFrame, ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Robustness at pickup vs the spec look-back window W (spec H[0,W], plotted span -W..0)."""
    pcfg = cfg.get("pupil_stl", {})
    rows = []
    for w in pcfg.get("sweep_spec_window_s", [5, 10, 15, 30, 60, 120]):
        dw = add_robustness(df, int(w), pcfg.get("beta", 50.0))
        locked = event_locked(dw, ev, int(w), 0)
        at = locked[locked["t"] == 0].dropna(subset=["robustness"])
        by_p = at.groupby(["participant", "expertise", "picked"], as_index=False)["robustness"].mean()
        for picked, lab in ((True, "picked"), (False, "not_picked")):
            b = by_p[by_p.picked == picked]
            e = b.loc[b.expertise == "expert", "robustness"]
            n = b.loc[b.expertise == "novice", "robustness"]
            p = mannwhitneyu(e, n).pvalue if len(e) > 1 and len(n) > 1 else np.nan
            rows.append((w, lab, len(e), len(n), e.mean(), n.mean(), e.mean() - n.mean(), p))
    return pd.DataFrame(
        rows, columns=["window_s", "outcome", "n_expert", "n_novice", "expert_mean", "novice_mean", "gap", "p_mwu"]
    )


def plot_window_sweep(sw: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (outcome, title) in zip(axes, (("picked", "victim picked up"), ("not_picked", "not picked up"))):
        d = sw[sw.outcome == outcome]
        ax.plot(d.window_s, d.expert_mean, "o-", color=GROUP_COLORS["expert"], label="expert")
        ax.plot(d.window_s, d.novice_mean, "o-", color=GROUP_COLORS["novice"], label="novice")
        for w, gap, p in zip(d.window_s, d.expert_mean, d.p_mwu):
            ax.annotate(f"p={p:.2f}", (w, gap), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax.set_xscale("log")
        ax.set_xticks(d.window_s, [str(int(w)) for w in d.window_s])
        ax.set_title(title)
        ax.set_xlabel("spec window W (s): H[0,W], robustness at pickup")
    axes[0].set_ylabel("probabilistic robustness at pickup")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    r = rankdata(score)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _group_gap(by_p: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """Expert minus novice (participant level): difference, Cohen's d, Mann-Whitney p."""
    e = by_p.loc[by_p.expertise == "expert", col]
    n = by_p.loc[by_p.expertise == "novice", col]
    pooled = np.sqrt(((len(e) - 1) * e.var() + (len(n) - 1) * n.var()) / (len(e) + len(n) - 2))
    return e.mean() - n.mean(), (e.mean() - n.mean()) / pooled, mannwhitneyu(e, n).pvalue


def predicate_comparison(windows: pd.DataFrame, ev: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare predicate definitions: pupil feature x baseline handling, then a tau sweep.

    Fixed comparison set (all variants reported). Per variant:
      auc_*: how well p separates LLM from no_llm windows (leave-one-participant-out)
      llm_* / pickup_*: expert minus novice in mean robustness (LLM runs / at pickup)
    """
    pcfg = cfg.get("pupil_stl", {})
    tau0 = pcfg.get("tau", 0.75)
    taus = pcfg.get("sweep_tau", [0.5, 0.55, 0.6, 0.65, 0.75])
    variants, tau_rows = [], []
    for feat in FEATURES:
        for norm in ("raw", "subject"):
            d = calibrate(windows.copy(), pcfg, np.random.default_rng(pcfg.get("seed", 0)), feat, norm)
            d = add_robustness(d, int(pcfg.get("spec_window_s", 30)), pcfg.get("beta", 50.0))
            y = (d["condition"] == "llm").to_numpy(dtype=float)
            auc_by_p = np.mean([_auc(g["p"].to_numpy(), y[g.index]) for _, g in d.reset_index(drop=True).groupby("participant")
                                if 0 < y[g.index].sum() < len(g)])
            row = {"feature": feat, "norm": norm, "auc_pooled": _auc(d["p"].to_numpy(), y), "auc_by_participant": auc_by_p}

            runs = run_summary(d, tau0)
            llm = runs[runs.condition == "llm"].groupby(["participant", "expertise"], as_index=False).mean(numeric_only=True)
            row["llm_gap"], row["llm_d"], row["llm_p"] = _group_gap(llm, "mean_robustness")

            at = event_locked(d, ev, 0, 0).dropna(subset=["robustness"])
            at = at.groupby(["participant", "expertise"], as_index=False)["robustness"].mean()
            row["pickup_gap"], row["pickup_d"], row["pickup_p"] = _group_gap(at, "robustness")
            variants.append(row)

            # tau sweep: share of monitored time with phi violated, LLM runs
            for tau in taus:
                viol = (
                    d[d.condition == "llm"].dropna(subset=["robustness_exact"])
                    .assign(v=lambda x: x["robustness_exact"] < 1 - tau)
                    .groupby(["participant", "expertise"], as_index=False)["v"].mean()
                )
                gap, dd, pv = _group_gap(viol, "v")
                e = viol.loc[viol.expertise == "expert", "v"].mean()
                n = viol.loc[viol.expertise == "novice", "v"].mean()
                tau_rows.append((feat, norm, tau, e, n, gap, dd, pv))
    tau_df = pd.DataFrame(
        tau_rows, columns=["feature", "norm", "tau", "expert_frac_violated", "novice_frac_violated", "gap", "d", "p_mwu"]
    )
    return pd.DataFrame(variants), tau_df


def tau_sweep(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Share of monitored time phi is violated vs the threshold tau, expert vs novice.

    Per participant and condition: fraction of seconds with exact robustness < 1 - tau.
    p_mwu is per tau; p_familywise is a one-sided exact permutation test (all expert/novice
    relabelings of the participants) on max over tau of (novice - expert) violation, so
    picking the best tau afterwards cannot inflate significance.
    """
    pcfg = cfg.get("pupil_stl", {})
    lo, hi, step = pcfg.get("tau_grid", [0.5, 0.9, 0.02])
    taus = np.round(np.arange(lo, hi + 1e-9, step), 4)
    rows = []
    for cond, d in df.dropna(subset=["robustness_exact"]).groupby("condition"):
        parts = sorted(d.participant.unique())
        grp = np.array([d.loc[d.participant == q, "expertise"].iloc[0] for q in parts])
        viol = np.array([
            (d.loc[d.participant == q, "robustness_exact"].to_numpy()[:, None] < 1 - taus).mean(axis=0)
            for q in parts
        ])  # (participants, taus)
        is_exp = grp == "expert"
        gap = viol[~is_exp].mean(axis=0) - viol[is_exp].mean(axis=0)  # novice - expert
        stat = gap.max()
        perm = []
        for idx in combinations(range(len(parts)), int(is_exp.sum())):
            m = np.zeros(len(parts), bool)
            m[list(idx)] = True
            perm.append((viol[~m].mean(axis=0) - viol[m].mean(axis=0)).max())
        p_fw = float((np.array(perm) >= stat - 1e-12).mean())
        for j, tau in enumerate(taus):
            pv = mannwhitneyu(viol[is_exp, j], viol[~is_exp, j]).pvalue
            rows.append((cond, tau, viol[is_exp, j].mean(), viol[~is_exp, j].mean(), gap[j], pv, p_fw))
    return pd.DataFrame(
        rows, columns=["condition", "tau", "expert", "novice", "gap_novice_minus_expert", "p_mwu", "p_familywise"]
    )


def plot_tau_sweep(sw: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for j, cond in enumerate(("no_llm", "llm")):
        d = sw[sw.condition == cond]
        ax = axes[0, j]
        ax.plot(d.tau, d.expert, color=GROUP_COLORS["expert"], label="expert")
        ax.plot(d.tau, d.novice, color=GROUP_COLORS["novice"], label="novice")
        ax.set_title(f"{'no LLM' if cond == 'no_llm' else 'LLM'}  (familywise p = {d.p_familywise.iloc[0]:.3f})")
        ax.set_ylabel("share of time phi violated")
        ax2 = axes[1, j]
        ax2.plot(d.tau, d.p_mwu, "k.-", ms=3)
        ax2.axhline(0.05, color="r", lw=0.8, ls="--")
        ax2.set_yscale("log")
        ax2.set_ylabel("p (per tau, MWU)")
        ax2.set_xlabel("threshold tau")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def alt_events(cfg: dict) -> pd.DataFrame:
    """Help-request events from the help_vs_auto classification: Alt press vs automatic request."""
    hva = cfg["help_vs_auto"]
    path = ROOT / cfg["paths"]["processed"] / "help_vs_auto_events_stepwin.csv"
    e = pd.read_csv(path).drop_duplicates(["subject", "trial", "run", "step", "label", "event_timestamp"])
    e = e[e["subject"].isin(included_subjects(cfg)) & ~e["subject"].isin(hva.get("exclude_participants", []))]
    return pd.DataFrame({
        "participant": e["subject"], "trial": e["trial"], "run": "run_" + e["run"].astype(int).astype(str),
        "t_abs": e["event_timestamp"], "alt": e["label"] == 1,
    }).reset_index(drop=True)


def alt_discrimination(locked: pd.DataFrame) -> pd.DataFrame:
    """Does workload at the event separate Alt presses from automatic requests?

    Per participant with both event types: AUC of the value at t = 0 for predicting Alt.
    Then a sign test / Wilcoxon of participant AUCs against 0.5.
    """
    from scipy.stats import wilcoxon

    at = locked[locked["t"] == 0]
    rows = []
    for col in ("robustness", "p"):
        d = at.dropna(subset=[col])
        aucs = []
        for _, g in d.groupby("participant"):
            y = g["alt"].to_numpy(dtype=float)
            if 0 < y.sum() < len(y):
                aucs.append(_auc(g[col].to_numpy(), y))
        aucs = np.array(aucs)
        pv = wilcoxon(aucs - 0.5).pvalue if len(aucs) > 5 else np.nan
        rows.append((col, len(aucs), aucs.mean(), np.median(aucs), pv))
    return pd.DataFrame(rows, columns=["signal", "n_participants", "mean_auc", "median_auc", "p_wilcoxon"])


def run(cfg: dict) -> pd.DataFrame:
    processed = ROOT / cfg["paths"]["processed"]
    windows = build_windows(cfg)
    df = build_traces(cfg, windows)
    df.to_csv(processed / "pupil_stl_traces.csv", index=False)

    runs = run_summary(df, cfg.get("pupil_stl", {}).get("tau", 0.75))
    runs.to_csv(processed / "pupil_stl_runs.csv", index=False)
    stats = compare_groups(runs)
    stats.to_csv(processed / "pupil_stl_group_stats.csv", index=False)
    print(stats.round(4).to_string(index=False))

    pcfg = cfg.get("pupil_stl", {})
    ev = pickup_events(cfg)
    locked = event_locked(df, ev, pcfg.get("event_pre_s", 30), pcfg.get("event_post_s", 15))
    locked.to_csv(processed / "pupil_stl_event_locked.csv", index=False)
    ev_stats = event_stats(locked)
    ev_stats.to_csv(processed / "pupil_stl_event_stats.csv", index=False)
    print(f"\npickup events: {len(ev)} ({int(ev.picked.sum())} picked up)")
    print(ev_stats.round(4).to_string(index=False))
    plot_event_locked(locked, cfg, processed / "pupil_stl_event_locked.png")
    print(f"figure -> {processed / 'pupil_stl_event_locked.png'}")

    alt = alt_events(cfg)
    alt_locked = event_locked(df, alt, pcfg.get("event_pre_s", 30), pcfg.get("event_post_s", 15), flag="alt")
    alt_locked.to_csv(processed / "pupil_stl_alt_locked.csv", index=False)
    alt_stats = event_stats(alt_locked, "alt", ("alt_press", "auto_request"))
    alt_stats.to_csv(processed / "pupil_stl_alt_stats.csv", index=False)
    alt_auc = alt_discrimination(alt_locked)
    alt_auc.to_csv(processed / "pupil_stl_alt_discrimination.csv", index=False)
    print(f"\nhelp-request events: {len(alt)} ({int(alt.alt.sum())} Alt presses, {int((~alt.alt).sum())} automatic)")
    print(alt_stats.round(4).to_string(index=False))
    print("\nAlt vs auto discrimination from robustness / p at the event:")
    print(alt_auc.round(4).to_string(index=False))
    plot_event_locked(
        alt_locked, cfg, processed / "pupil_stl_alt_locked.png", "alt",
        ("Alt press (help requested)", "automatic request (no Alt)"), "time relative to LLM request (s)",
    )

    sw = window_sweep(df, ev, cfg)
    sw.to_csv(processed / "pupil_stl_window_sweep.csv", index=False)
    print("\nspec-window sweep (robustness at pickup):")
    print(sw.round(4).to_string(index=False))
    plot_window_sweep(sw, processed / "pupil_stl_window_sweep.png")

    ts = tau_sweep(df, cfg)
    ts.to_csv(processed / "pupil_stl_tau_sweep_sd.csv", index=False)
    plot_tau_sweep(ts, processed / "pupil_stl_tau_sweep.png")
    print("\ntau sweep, SD predicate (familywise p per condition):")
    print(ts.groupby("condition")["p_familywise"].first().round(4).to_string())
    print(ts[ts.tau.isin([0.5, 0.6, 0.65, 0.7, 0.75, 0.8])].round(4).to_string(index=False))

    variants, tau_df = predicate_comparison(windows, ev, cfg)
    variants.to_csv(processed / "pupil_stl_predicate_variants.csv", index=False)
    tau_df.to_csv(processed / "pupil_stl_tau_sweep.csv", index=False)
    print("\npredicate variants (gap = expert - novice robustness; d = Cohen's d):")
    print(variants.round(4).to_string(index=False))
    print("\ntau sweep (LLM runs, share of time phi violated):")
    print(tau_df.round(4).to_string(index=False))

    fig_path = processed / "pupil_stl_trajectories.png"
    plot(df, cfg, fig_path)
    print(f"\nfigure -> {fig_path}")
    return df
