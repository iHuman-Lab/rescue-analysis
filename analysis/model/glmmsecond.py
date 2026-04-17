import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def prepare_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Select the best run per participant/trial and encode model columns."""
    condition_map = cfg.get("analysis", {}).get("condition_by_category", {})
    metric = cfg.get("glmm2", {}).get("best_run_metric")
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


def _fixed_effect_formula(outcome: str) -> str:
    return (
        f"{outcome} ~ C(condition, Treatment('no_llm'))"
        " * C(expertise, Treatment('novice'))"
    )


def run_glmm(df: pd.DataFrame, outcome: str) -> dict:
    """Fit a linear mixed-effects model with participant and category random intercepts."""
    model_df = (
        df[["participant", "category", "condition", "expertise", outcome]]
        .dropna()
        .reset_index(drop=True)
    )
    if model_df.empty:
        return {"model_type": "skipped", "terms": [], "variance_terms": []}

    formula = _fixed_effect_formula(outcome)

    model = smf.mixedlm(
        formula,
        model_df,
        groups=model_df["participant"],
        re_formula="1",
        vc_formula={"category": "0 + C(category)"},
    ).fit()
    return {
        "model_type": "linear_mixedlm",
        "terms": [
            {
                "term": term,
                "coef": model.params[term],
                "se": model.bse.get(term),
                "p_value": model.pvalues.get(term),
            }
            for term in model.fe_params.index
        ],
        "variance_terms": [
            {
                "term": "participant_random_intercept_var",
                "coef": float(model.cov_re.iloc[0, 0]),
                "se": np.nan,
                "p_value": np.nan,
            },
            {
                "term": "category_random_intercept_var",
                "coef": float(model.vcomp[0]) if len(model.vcomp) else np.nan,
                "se": np.nan,
                "p_value": np.nan,
            },
        ],
    }


def run_all(cfg: dict, dataframes: dict) -> pd.DataFrame:
    """Select best runs and fit one linear mixed-effects model per outcome."""
    count_features = cfg["glmm2"]["count"]
    features = cfg["glmm2"]["continuous"] + count_features
    rows = []

    for name, df in dataframes.items():
        if df is None or df.empty:
            continue

        # Select best run per participant per trial, rename trial → category
        prepared = prepare_df(df, cfg)

        for outcome in features:
            result = run_glmm(prepared, outcome)
            rows.extend(
                {
                    "dataset": name,
                    "outcome": outcome,
                    "model_type": result["model_type"],
                    **term,
                }
                for term in result["terms"]
            )
            rows.extend(
                {
                    "dataset": name,
                    "outcome": outcome,
                    "model_type": result["model_type"],
                    **term,
                }
                for term in result.get("variance_terms", [])
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        valid = results["p_value"].notna()
        _, fdr, _, _ = multipletests(results.loc[valid, "p_value"], method="fdr_bh")
        results["p_value_fdr"] = np.nan
        results.loc[valid, "p_value_fdr"] = fdr

    return results
