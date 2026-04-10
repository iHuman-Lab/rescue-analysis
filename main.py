import yaml
from pathlib import Path

from analysis.data.xdf import collect_subjects, load_all_subjects
from analysis.features.extract_features import run_extract_features
from analysis.features.eyetracking_features import run_eyetracking_features
from analysis.glmmsecond import run_all as run_glmmsecond
from analysis.utils import skip_run

ROOT   = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "config_analysis.yml"

if __name__ == "__main__":
	with open(CONFIG) as f:
		cfg = yaml.safe_load(f)
	processed_dir = ROOT / cfg["paths"]["processed"]

	preloaded = {}

	with skip_run("run", "xdf") as check, check():
		preloaded = collect_subjects(cfg)  # XDF → disk + in-memory

	if not preloaded:
		preloaded = load_all_subjects(cfg)  # xdf skipped → load once from data/intermediate

	eyetracking = {}
	with skip_run("run", "eyetracking") as check, check():
		eyetracking = run_eyetracking_features(cfg, preloaded=preloaded, root=ROOT)

	best_features = None
	with skip_run("run", "extract_features") as check, check():
		best_features = run_extract_features(cfg, preloaded, eyetracking, processed_dir=processed_dir)


	with skip_run("run", "glmmsecond") as check, check():
		run_glmmsecond(cfg, ROOT, dataframes={"best_features": best_features})

