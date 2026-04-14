import yaml
from pathlib import Path

from analysis.prepare_data.xdf import collect_subjects, load_all_subjects
from analysis.features.extract_features import run_extract_features
from analysis.features.eyetracking_features import run_eyetracking_features
from analysis.features.AOI_fixation import aggregate_transitions
from analysis.features.io import (
	save_aoi_results,
	save_aggregated_transitions,
	save_best_features,
)
from analysis.glmmsecond import run_all as run_glmmsecond
from analysis.utils import skip_run

ROOT   = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "analysis.yml"

with open(CONFIG) as f:
	cfg = yaml.safe_load(f)
processed_dir = ROOT / cfg["paths"]["processed"]

preloaded = {}



with skip_run("run", "xdf") as check, check():
	preloaded = collect_subjects(cfg)  

if not preloaded:
	preloaded = load_all_subjects(cfg) 


eyetracking = {}
with skip_run("run", "eyetracking") as check, check():
	eyetracking = run_eyetracking_features(cfg, preloaded=preloaded, root=ROOT)
	if "aoi" in eyetracking:
		save_aoi_results(eyetracking["aoi"], processed_dir, cfg)
		total_trans, by_trial = aggregate_transitions(eyetracking["aoi"], cfg.get("aoi", []), cfg)
		save_aggregated_transitions(total_trans, by_trial, processed_dir, cfg)

best_features = None
with skip_run("run", "extract_features") as check, check():
	best_features = run_extract_features(cfg, preloaded, eyetracking)
	if best_features is not None and not best_features.empty:
		out = save_best_features(best_features, processed_dir, cfg)
		print(f"\nbest_features -> {out}")


with skip_run("run", "glmmsecond") as check, check():
	run_glmmsecond(cfg, processed_dir, dataframes={"best_features": best_features}, root=ROOT)
