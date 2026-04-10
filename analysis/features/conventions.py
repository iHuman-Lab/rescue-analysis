import re


DEFAULT_BEST_SUFFIX = "_best"
DEFAULT_CATEGORY_NAMES = ("gemini", "openai", "dummy")
DEFAULT_OFFSCREEN_LABEL = "offscreen"
DEFAULT_PUPIL_COL = "avg_pupil_diam"

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
    return cfg.get("naming", {}).get("best_suffix", DEFAULT_BEST_SUFFIX)


def category_names(cfg: dict) -> list[str]:
    names = cfg.get("naming", {}).get("categories", DEFAULT_CATEGORY_NAMES)
    return [str(name) for name in names]


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
