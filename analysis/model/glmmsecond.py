from math import erfc, sqrt

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
from statsmodels.stats.multitest import multipletests
from analysis.features.conventions import condition_for_category


def prepare_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep only FEATURES + id cols."""
    df = df.copy()

    df["condition"] = df["category"].apply(lambda c: condition_for_category(c, cfg))
    df["condition"] = pd.Categorical(df["condition"], categories=["no_llm", "llm"])
    df["expertise"] = pd.Categorical(df["expertise"], categories=["novice", "expert"])
    features = cfg["glmm2"]["continuous"] + cfg["glmm2"]["count"]
    keep = ["participant", "category", "condition", "expertise"] + features
    return df[keep]


def _normal_approx_p_value(coef: float, se: float) -> float:
    if se is None or pd.isna(se) or se == 0:
        return np.nan
    z_score = abs(coef / se)
    return erfc(z_score / sqrt(2.0))


def run_glmm(df: pd.DataFrame, outcome: str, count_features: list[str]):
    """Fit an LMM for continuous outcomes and a Poisson GLMM for count outcomes."""
    clean = df.copy()

    formula = (
        f"{outcome} ~ C(condition, Treatment('no_llm'))"
        " * C(expertise, Treatment('novice'))"
    )

    if outcome in count_features:
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
                for term, coef, se in zip(
                    model.model.exog_names,
                    model.fe_mean,
                    model.fe_sd,
                )
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


def _result_rows(dataset: str, outcome: str, result: dict) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "outcome": outcome,
            "model_type": result["model_type"],
            **term_result,
        }
        for term_result in result["terms"]
    ]


def run_all(
    cfg: dict,
    dataframes: dict,
) -> pd.DataFrame:
    """Run one mixed-effects model per feature for each dataset."""
    count_features = cfg["glmm2"]["count"]
    features = cfg["glmm2"]["continuous"] + count_features
    rows = []

    for name, df in dataframes.items():
        if df is None or df.empty:
            continue
        prepared = prepare_df(df, cfg)
        for outcome in features:
            result = run_glmm(prepared, outcome, count_features)
            rows.extend(_result_rows(name, outcome, result))

    results = pd.DataFrame(rows)
    if not results.empty:
        valid = results["p_value"].notna()
        _, fdr, _, _ = multipletests(results.loc[valid, "p_value"], method="fdr_bh")
        results["p_value_fdr"] = np.nan
        results.loc[valid, "p_value_fdr"] = fdr

    return results
