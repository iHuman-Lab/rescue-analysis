from math import erfc, sqrt

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
from statsmodels.stats.multitest import multipletests


def prepare_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Select the best run per participant/trial and encode model columns."""
    condition_map = cfg.get("analysis", {}).get("condition_by_category", {})
    metric = cfg.get("glmm2", {}).get("best_run_metric", "saved_victims")
    count_features = cfg["glmm2"]["count"]
    features = cfg["glmm2"]["continuous"] + count_features

    best = (
        df.sort_values(metric, ascending=False)
        .groupby(["participant", "trial"], as_index=False)
        .first()
        .rename(columns={"trial": "category"})
    )

    best["condition"] = best["category"].map(condition_map).fillna("unknown")
    best["condition"] = pd.Categorical(best["condition"], categories=["no_llm", "llm"])
    best["expertise"] = pd.Categorical(
        best["expertise"], categories=["novice", "expert"]
    )
    keep = ["participant", "category", "condition", "expertise"] + features
    return best[keep].reset_index(drop=True)


def run_glmm(df: pd.DataFrame, outcome: str, count_features: list[str]) -> dict:
    """Fit an LMM for continuous outcomes and a Poisson GLMM for count outcomes."""
    model_df = df[
        ["participant", "condition", "expertise", outcome]
    ].dropna().reset_index(drop=True)
    if model_df.empty:
        return {"model_type": "skipped", "terms": []}

    formula = (
        f"{outcome} ~ C(condition, Treatment('no_llm'))"
        " * C(expertise, Treatment('novice'))"
    )

    if outcome in count_features:
        model = PoissonBayesMixedGLM.from_formula(
            formula, {"participant": "0 + C(participant)"}, model_df
        ).fit_vb()
        return {
            "model_type": "poisson_glmm",
            "terms": [
                {
                    "term": term,
                    "coef": coef,
                    "se": se,
                    "p_value": np.nan if (se is None or pd.isna(se) or se == 0)
                               else erfc(abs(coef / se) / sqrt(2.0)),
                }
                for term, coef, se in zip(model.model.exog_names, model.fe_mean, model.fe_sd)
            ],
        }

    model = smf.mixedlm(formula, model_df, groups=model_df["participant"]).fit()
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


def run_all(cfg: dict, dataframes: dict) -> pd.DataFrame:
    """Select best run per participant/category, then run one mixed model per feature."""
    count_features = cfg["glmm2"]["count"]
    features = cfg["glmm2"]["continuous"] + count_features
    rows = []

    for name, df in dataframes.items():
        if df is None or df.empty:
            continue

        # Select best run per participant per trial, rename trial → category
        prepared = prepare_df(df, cfg)

        for outcome in features:
            result = run_glmm(prepared, outcome, count_features)
            rows.extend(
                {"dataset": name, "outcome": outcome, "model_type": result["model_type"], **term}
                for term in result["terms"]
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        valid = results["p_value"].notna()
        _, fdr, _, _ = multipletests(results.loc[valid, "p_value"], method="fdr_bh")
        results["p_value_fdr"] = np.nan
        results.loc[valid, "p_value_fdr"] = fdr

    return results
