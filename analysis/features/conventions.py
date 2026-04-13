import re


DEFAULT_BEST_SUFFIX = "_best"
DEFAULT_CATEGORY_NAMES = ("gemini", "openai", "dummy")
DEFAULT_OFFSCREEN_LABEL = "offscreen"
DEFAULT_PUPIL_COL = "avg_pupil_diam"
DEFAULT_CONDITION_BY_CATEGORY = {
    "dummy": "no_llm",
    "gemini": "llm",
    "openai": "llm",
}
DEFAULT_OUTPUT_FILES = {
    "best_features_file": "best_features.csv",
    "fixation_aoi_file": "fixations_aoi.csv",
    "aoi_transitions_file": "aoi_transitions.csv",
    "aoi_transitions_all_file": "aoi_transitions_all.csv",
    "aoi_transitions_by_trial_template": "aoi_transitions_{trial}.csv",
}

FIXATION_COLUMNS = ["start_ms", "end_ms", "duration_ms", "x", "y"]
SACCADE_COLUMNS = [
    "saccade_id",
    "start_ms",
    "end_ms",
    "duration_ms",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "amplitude",
]


def best_suffix(cfg: dict) -> str:
    template = (
        cfg.get("xdf", {})
        .get("output", {})
        .get("best_dir_template", "{base}_best")
    )
    return template.replace("{base}", "") if "{base}" in template else DEFAULT_BEST_SUFFIX


def category_names(cfg: dict) -> list[str]:
    names = cfg.get("analysis", {}).get("categories", DEFAULT_CATEGORY_NAMES)
    return [str(name) for name in names]


def condition_for_category(category: str, cfg: dict) -> str:
    mapping = cfg.get("analysis", {}).get(
        "condition_by_category", DEFAULT_CONDITION_BY_CATEGORY
    )
    return str(mapping.get(category, "llm"))


def output_file(cfg: dict, key: str) -> str:
    outputs = cfg.get("analysis", {}).get("outputs", {})
    return str(outputs.get(key, DEFAULT_OUTPUT_FILES[key]))


def detect_category(trial_id: str, cfg: dict) -> str:
    suffix = best_suffix(cfg)
    base_name = trial_id[: -len(suffix)] if suffix and trial_id.endswith(suffix) else trial_id
    return next((name for name in category_names(cfg) if name in base_name), "unknown")


def subject_label(subject_id: str, cfg: dict) -> str:
    template = (
        cfg.get("xdf", {})
        .get("output", {})
        .get("subject_dir_template", "sub-{subject_id}")
    )
    return template.format(subject_id=subject_id)


def run_suffix_pattern(cfg: dict) -> str:
    return (
        cfg.get("xdf", {})
        .get("output", {})
        .get("run_suffix_pattern", r"_run\d+$")
    )


def strip_run_suffix(name: str, cfg: dict) -> str:
    return re.sub(run_suffix_pattern(cfg), "", name)
