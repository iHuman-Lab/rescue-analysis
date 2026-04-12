import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def get_glmm_cfg(cfg: dict) -> dict:
    """Return GLMM configuration with safe defaults."""
    glmm_cfg = cfg.get("glmm2", {})
    return {
        "continuous": glmm_cfg.get("continuous", []),
        "count": glmm_cfg.get("count", []),
        "group_col": glmm_cfg.get("group_col", "participant"),
        "output_file": glmm_cfg.get("output_file", "glmm2_results.csv"),
        "verbose": glmm_cfg.get("verbose", True),
    }


def get_all_features(cfg: dict) -> list[str]:
    """Return every configured outcome column name."""
    glmm_cfg = get_glmm_cfg(cfg)
    return glmm_cfg["continuous"] + glmm_cfg["count"]


def prepare_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add condition and expertise columns, keep only FEATURES + id cols."""
    expertise_map = {str(k): str(v) for k, v in cfg.get("expertise", {}).items()}
    df = df.copy()
    derived_condition = df["category"].apply(
        lambda c: "no_llm" if c == "dummy" else "llm"
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
    features = get_all_features(cfg)
    keep = ["participant", "category", "condition", "expertise"] + [
        f for f in features if f in df.columns
    ]
    return df[keep]


def get_feature_groups(cfg: dict) -> dict:
    """Return outcome names grouped by model family from config."""
    return {
        "continuous": get_glmm_cfg(cfg)["continuous"],
        "count": get_glmm_cfg(cfg)["count"],
    }


def make_formula(outcome: str) -> str:
    """Build the shared fixed-effects formula for all outcomes."""
    return (
        f"{outcome} ~ C(condition, Treatment('no_llm'))"
        " * C(expertise, Treatment('novice'))"
    )


def fit_continuous_glmm(clean: pd.DataFrame, formula: str, group_col: str):
    """Fit a Gaussian mixed model with participant random intercepts."""
    return smf.mixedlm(formula, clean, groups=clean[group_col]).fit()


def fit_count_glmm(clean: pd.DataFrame, formula: str, group_col: str):
    """Fit a Poisson mixed model with participant random intercepts."""
    model = PoissonBayesMixedGLM.from_formula(
        formula,
        {group_col: f"0 + C({group_col})"},
        clean,
    )
    return model.fit_map()


def summarize_model(model, outcome_type: str) -> list[dict]:
    """Convert fitted model output into a row-per-term summary."""
    if outcome_type == "continuous":
        return [
            {
                "term": term,
                "coef": model.params[term],
                "se": model.bse.get(term),
                "p_value": model.pvalues.get(term),
            }
            for term in model.params.index
        ]

    rows = []
    for term, coef, se in zip(model.model.exog_names, model.fe_mean, model.fe_sd):
        z_value = coef / se if se else np.nan
        p_value = 2 * norm.sf(abs(z_value)) if se else np.nan
        rows.append(
            {
                "term": term,
                "coef": coef,
                "se": se,
                "p_value": p_value,
            }
        )

    for term, coef in zip(model.model.vcp_names, model.vcp_mean):
        rows.append(
            {
                "term": term,
                "coef": coef,
                "se": np.nan,
                "p_value": np.nan,
            }
        )
    return rows


def run_glmm(df: pd.DataFrame, outcome: str, outcome_type: str, group_col: str):
    """Fit the configured mixed model for one outcome."""
    clean = df.dropna(subset=[outcome]).copy()
    if clean.empty:
        return None
    formula = make_formula(outcome)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            if outcome_type == "continuous":
                model = fit_continuous_glmm(clean, formula, group_col)
            elif outcome_type == "count":
                model = fit_count_glmm(clean, formula, group_col)
            else:
                raise ValueError(f"Unknown outcome type '{outcome_type}' for '{outcome}'.")
        except (np.linalg.LinAlgError, ValueError):
            warnings.warn(f"Skipping '{outcome}': singular matrix.", RuntimeWarning)
            return None
    return summarize_model(model, outcome_type)


def iter_model_rows(prepared: pd.DataFrame, feature_groups: dict, group_col: str):
    """Yield model summaries for configured outcomes present in the data."""
    for outcome_type, features in feature_groups.items():
        for outcome in features:
            if outcome not in prepared.columns:
                continue
            model_rows = run_glmm(prepared, outcome, outcome_type, group_col)
            if model_rows is None:
                continue
            yield outcome_type, outcome, model_rows


def append_result_rows(rows: list[dict], dataset: str, outcome: str,
                       outcome_type: str, model_rows: list[dict]) -> None:
    """Append summarized model rows to the results list."""
    for row in model_rows:
        rows.append(
            {
                "dataset": dataset,
                "outcome": outcome,
                "outcome_type": outcome_type,
                "term": row["term"],
                "coef": row["coef"],
                "se": row["se"],
                "p_value": row["p_value"],
            }
        )


def finalize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Add FDR-corrected p-values to the results table."""
    if results.empty:
        return results
    valid = results["p_value"].notna()
    _, fdr, _, _ = multipletests(results.loc[valid, "p_value"], method="fdr_bh")
    results = results.copy()
    results["p_value_fdr"] = np.nan
    results.loc[valid, "p_value_fdr"] = fdr
    return results


def run_all(
    cfg: dict, root: Path, dataframes: dict, verbose: bool | None = None
) -> pd.DataFrame:
    """Run GLMM on all datasets.

    Args:
        cfg:        Full config dict.
        root:       Project root path (from main.py).
        dataframes: {name: DataFrame} from run_extract_features().
        verbose:    Print progress.
    """
    glmm_cfg = get_glmm_cfg(cfg)
    feature_groups = get_feature_groups(cfg)
    group_col = glmm_cfg["group_col"]
    verbose = glmm_cfg["verbose"] if verbose is None else verbose
    rows = []
    for name, df in dataframes.items():
        prepared = prepare_df(df, cfg)
        if verbose:
            print(f"\n=== {name} ===")
            print(prepared[["condition", "expertise"]].value_counts().to_string())
        for outcome_type, outcome, model_rows in iter_model_rows(
            prepared, feature_groups, group_col
        ):
            append_result_rows(rows, name, outcome, outcome_type, model_rows)
            if verbose:
                print(f"  {outcome} ({outcome_type}): fitted")

    results = finalize_results(pd.DataFrame(rows))
    if not results.empty:
        if verbose:
            print(results.to_string(index=False))
    return results
