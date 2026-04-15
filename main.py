from pathlib import Path

import yaml
from analysis.utils import skip_run

from analysis.features.extract_features import extract_features
from analysis.model.glmmsecond import run_all as run_glmmsecond
from analysis.prepare_data.data import split_by_trail

with open("configs/analysis.yml") as f:
    cfg = yaml.safe_load(f)


preloaded = {}
with skip_run("run", "split_data") as check, check():
    preloaded = split_by_trail(cfg)


with skip_run("run", "extract_features") as check, check():
    df = extract_features(preloaded, cfg)

# # eyetracking = {}
# with skip_run("skip", "eyetracking") as check, check():
#     eyetracking = run_eyetracking_features(cfg, preloaded=preloaded, root=ROOT)
#     if "aoi" in eyetracking:
#         save_aoi_results(eyetracking["aoi"], processed_dir, cfg)
#         total_trans, by_trial = aggregate_transitions(
#             eyetracking["aoi"], cfg.get("aoi", []), cfg
#         )
#         save_aggregated_transitions(total_trans, by_trial, processed_dir, cfg)

# best_features = None
# with skip_run("skip", "extract_features") as check, check():
#     best_features = run_extract_features(cfg, preloaded, eyetracking)
#     if best_features is not None and not best_features.empty:
#         out = save_best_features(best_features, processed_dir, cfg)
#         print(f"\nbest_features -> {out}")

with skip_run("run", "glmmsecond") as check, check():
    glmm_results = run_glmmsecond(cfg, dataframes={"best_features": df})
    if glmm_results is not None and not glmm_results.empty:
        processed_dir = Path(cfg["paths"]["processed"])
        filename = cfg.get("glmm2", {}).get("output_file", "glmm2_results.csv")
        out = processed_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        glmm_results.to_csv(out, index=False)
        print(f"\nglmmsecond -> {out}")
