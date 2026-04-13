from analysis.features.eyetracking_data import run_fixations, run_saccades
from analysis.features.AOI_fixation import run_aoi_fixations


def run_eyetracking_features(cfg: dict, preloaded: dict | None = None,
                              root=None) -> dict:
    fixations   = run_fixations(cfg, preloaded=preloaded)
    saccades    = run_saccades(cfg, preloaded=preloaded)
    eyetracking = {"fixations": fixations, "saccades": saccades}
    eyetracking["aoi"] = run_aoi_fixations(cfg, eyetracking=eyetracking, root=root)
    return eyetracking
