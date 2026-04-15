from math import erfc, sqrt

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
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


def _normal_approx_p_value(coef: float, se: float) -> float:
    if se is None or pd.isna(se) or se == 0:
        return np.nan
    z_score = abs(coef / se)
    return erfc(z_score / sqrt(2.0))


def run_glmm(df: pd.DataFrame, outcome: str, count_features: list[str]):
    """Fit an LMM for continuous outcomes and a Poisson GLMM for count outcomes."""
    if outcome not in df.columns:
        return None

    clean = df.dropna(subset=[outcome]).copy()
    if clean.empty:
        return None

    formula = (
        f"{outcome} ~ C(condition, Treatment('no_llm'))"
        " * C(expertise, Treatment('novice'))"
    )
    try:
        if outcome in count_features:
            if (clean[outcome] < 0).any():
                return None
            random = {"participant": "0 + C(participant)"}
            model = PoissonBayesMixedGLM.from_formula(formula, random, clean).fit_vb()
            return {
                "model_type": "poisson_glmm",
                "terms": [
                    {
                        "term": term,
                        "coef": coef,
                        "se": se,
                        "p_value": _normal_approx_p_value(coef, se),
                    }
                    for term, coef, se in zip(model.model.exog_names, model.fe_mean, model.fe_sd)
                ],
            }

        model = smf.mixedlm(formula, clean, groups=clean["participant"]).fit()
        return {
            "model_type": "linear_mixedlm",
            "terms": [
                {
                    "term": term,
                    "coef": model.params[term],
                    "se": model.bse.get(term),
                    "p_value": model.pvalues.get(term),
                }
                for term in model.params.index
            ],
        }
    except (np.linalg.LinAlgError, ValueError):
        return None


def run_all(
    cfg: dict,
    dataframes: dict,
) -> pd.DataFrame:
    """Run one mixed-effects model per feature for each dataset."""
    continuous_features = cfg["glmm2"]["continuous"]
    count_features = cfg["glmm2"]["count"]
    features = continuous_features + count_features
    rows = []

    for name, df in dataframes.items():
        if df is None or df.empty:
            continue
        prepared = prepare_df(df, cfg)
        for outcome in features:
            result = run_glmm(prepared, outcome, count_features)
            if result is None:
                continue

            for term_result in result["terms"]:
                rows.append(
                    {
                        "dataset": name,
                        "outcome": outcome,
                        "model_type": result["model_type"],
                        "term": term_result["term"],
                        "coef": term_result["coef"],
                        "se": term_result["se"],
                        "p_value": term_result["p_value"],
                    }
                )

    results = pd.DataFrame(rows)
    if not results.empty:
        valid = results["p_value"].notna()
        _, fdr, _, _ = multipletests(results.loc[valid, "p_value"], method="fdr_bh")
        results["p_value_fdr"] = np.nan
        results.loc[valid, "p_value_fdr"] = fdr

    return results
