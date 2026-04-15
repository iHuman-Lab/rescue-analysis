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
	save_glmm_results,
)
from analysis.glmmsecond import run_all as run_glmmsecond
from analysis.multivariate_mixed import run_all as run_multivariate_mixed
from analysis.utils import skip_run

ROOT   = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "analysis.yml"

def main() -> None:
	with open(CONFIG) as f:
		cfg = yaml.safe_load(f)
	processed_dir = ROOT / cfg["paths"]["processed"]

	preloaded = {}
	with skip_run("skip", "xdf") as check, check():
		preloaded = collect_subjects(cfg)

	if not preloaded:
		preloaded = load_all_subjects(cfg)

	eyetracking = {}
	with skip_run("skip", "eyetracking") as check, check():
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

	with skip_run("skip", "glmmsecond") as check, check():
		glmm_results = run_glmmsecond(cfg, dataframes={"best_features": best_features})
		if glmm_results is not None and not glmm_results.empty:
			out = save_glmm_results(glmm_results, processed_dir, cfg)
			print(f"\nglmmsecond -> {out}")

	with skip_run("skip", "glmmsecond_multivariate") as check, check():
		mv_results = run_multivariate_mixed(cfg, dataframes={"best_features": best_features})
		if mv_results is not None and not mv_results.empty:
			out = processed_dir / cfg["glmm2"]["multivariate"]["output_file"]
			mv_results.to_csv(out, index=False)
			print(f"\nglmmsecond_multivariate -> {out}")


if __name__ == "__main__":
	main()
