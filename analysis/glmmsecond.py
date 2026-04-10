from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning




def prepare_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add condition and expertise columns, keep only FEATURES + id cols."""
    expertise_map = {str(k): str(v) for k, v in cfg.get("expertise", {}).items()}
    df = df.copy()
    df["condition"] = df["category"].apply(
        lambda c: "no_llm" if c == "dummy" else "llm"
    )
    df["expertise"] = df["participant"].str.replace("sub-", "", regex=False).map(expertise_map).fillna("unknown")
    df["condition"] = pd.Categorical(df["condition"], categories=["no_llm", "llm"])
    df["expertise"] = pd.Categorical(df["expertise"], categories=["novice", "expert"])
    features = cfg["glmm2"]["continuous"] + cfg["glmm2"]["count"]
    keep = ["participant", "category", "condition", "expertise"] + [
        f for f in features if f in df.columns
    ]
    return df[keep]


def run_glmm(df: pd.DataFrame, outcome: str):
    """Fit a mixed LM for one outcome: condition * expertise, random intercept per participant."""
    clean = df.dropna(subset=[outcome]).copy()
    if clean.empty:
        return None
    formula = (
        f"{outcome} ~ C(condition, Treatment('no_llm'))"
        " * C(expertise, Treatment('novice'))"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            model = smf.mixedlm(formula, clean, groups=clean["participant"]).fit()
        except np.linalg.LinAlgError:
            warnings.warn(f"Skipping '{outcome}': singular matrix.", RuntimeWarning)
            return None
    return model


def run_all(cfg: dict, root: Path,
            dataframes: dict, verbose: bool = True) -> pd.DataFrame:
    """Run GLMM on all datasets.

    Args:
        cfg:        Full config dict.
        root:       Project root path (from main.py).
        dataframes: {name: DataFrame} from run_extract_features().
        verbose:    Print progress.
    """
    processed_dir = root / cfg["paths"]["processed"]
    features      = cfg["glmm2"]["continuous"] + cfg["glmm2"]["count"]
    rows = []
    for name, df in dataframes.items():
        prepared = prepare_df(df, cfg)
        if verbose:
            print(f"\n=== {name} ===")
            print(prepared[["condition", "expertise"]].value_counts().to_string())
        for outcome in features:
            if outcome not in prepared.columns:
                continue
            model = run_glmm(prepared, outcome)
            if model is None:
                continue
            for term in model.params.index:
                rows.append({
                    "dataset":  name,
                    "outcome":  outcome,
                    "term":     term,
                    "coef":     model.params[term],
                    "se":       model.bse.get(term),
                    "p_value":  model.pvalues.get(term),
                })
            if verbose:
                print(f"  {outcome}: fitted")

    results = pd.DataFrame(rows)
    if not results.empty:
        valid = results["p_value"].notna()
        _, fdr, _, _ = multipletests(results.loc[valid, "p_value"], method="fdr_bh")
        results["p_value_fdr"] = np.nan
        results.loc[valid, "p_value_fdr"] = fdr
        out = processed_dir / "glmm2_results.csv"
        results.to_csv(out, index=False)
        if verbose:
            print(f"\nResults saved -> {out.relative_to(root)}")
            print(results.to_string(index=False))
    return results


