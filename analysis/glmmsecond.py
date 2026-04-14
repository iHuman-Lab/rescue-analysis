from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from analysis.features.conventions import condition_for_category


def prepare_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add condition and expertise columns, keep only FEATURES + id cols."""
    expertise_map = {str(k): str(v) for k, v in cfg.get("expertise", {}).items()}
    df = df.copy()
    derived_condition = df["category"].apply(
        lambda c: condition_for_category(c, cfg)
    )
    if "condition" in df.columns:
        df["condition"] = df["condition"].fillna(derived_condition)
    else:
        df["condition"] = derived_condition

    derived_expertise = (
        df["participant"]
        .str.replace("sub-", "", regex=False)
        .map(expertise_map)
        .fillna("unknown")
    )
    if "expertise" in df.columns:
        df["expertise"] = df["expertise"].fillna(derived_expertise)
    else:
        df["expertise"] = derived_expertise

    df["condition"] = pd.Categorical(df["condition"], categories=["no_llm", "llm"])
    df["expertise"] = pd.Categorical(df["expertise"], categories=["novice", "expert"])
    features = cfg["glmm2"]["continuous"] + cfg["glmm2"]["count"]
    keep = ["participant", "category", "condition", "expertise"] + [
        f for f in features if f in df.columns
    ]
    return df[keep]


def run_glmm(df: pd.DataFrame, features: list[str]):
    """Fit one mixed LM using all features together."""
    available = [feature for feature in features if feature in df.columns]
    if not available:
        return None

    clean = (
        df.melt(
            id_vars=["participant", "category", "condition", "expertise"],
            value_vars=available,
            var_name="feature",
            value_name="value",
        )
        .dropna(subset=["value"])
        .copy()
    )
    if clean.empty:
        return None

    formula = (
        "value ~ C(feature)"
        " + C(condition, Treatment('no_llm'))"
        " * C(expertise, Treatment('novice'))"
    )
    try:
        model = smf.mixedlm(formula, clean, groups=clean["participant"]).fit()
    except np.linalg.LinAlgError:
        print("Skipping pooled model: singular matrix.")
        return None
    return model


def run_all(
    cfg: dict,
    dataframes: dict,
) -> pd.DataFrame:
    """Run one pooled GLMM per dataset using all configured features together."""
    features = cfg["glmm2"]["continuous"] + cfg["glmm2"]["count"]
    rows = []

    for name, df in dataframes.items():
        prepared = prepare_df(df, cfg)
        model = run_glmm(prepared, features)
        if model is None:
            continue

        for term in model.params.index:
            rows.append(
                {
                    "dataset": name,
                    "outcome": "all_features",
                    "term": term,
                    "coef": model.params[term],
                    "se": model.bse.get(term),
                    "p_value": model.pvalues.get(term),
                }
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        valid = results["p_value"].notna()
        _, fdr, _, _ = multipletests(results.loc[valid, "p_value"], method="fdr_bh")
        results["p_value_fdr"] = np.nan
        results.loc[valid, "p_value_fdr"] = fdr

    return results
