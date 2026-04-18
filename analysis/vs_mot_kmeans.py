from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


VISUAL_SEARCH_TOKEN = "visual_search"
MOT_TOKEN = "multi_object_tracking"
SUBJECT_DIR_PATTERN = re.compile(r"sub-[A-Za-z0-9]+$")
SUBJECT_ID_PATTERN = re.compile(r"sub-[A-Za-z0-9]+", re.IGNORECASE)
VS_MOT_DIR_NAMES = {"vs_mot", "vs-mot"}


def _extract_subject_id(path: Path) -> str:
    for parent in [path.parent, *path.parents]:
        if SUBJECT_DIR_PATTERN.search(parent.name):
            return parent.name

    match = SUBJECT_ID_PATTERN.search(path.name)
    if match:
        return match.group(0)

    raise ValueError(f"Could not infer subject id from path: {path}")


def _task_from_name(path: Path) -> str | None:
    name = path.name.lower()
    if VISUAL_SEARCH_TOKEN in name:
        return "vs"
    if MOT_TOKEN in name:
        return "mot"
    return None


def _is_vs_mot_file(path: Path) -> bool:
    return any(part.lower() in VS_MOT_DIR_NAMES for part in path.parts)


def _discover_vs_mot_files(raw_dir: Path) -> dict[str, dict[str, Path]]:
    subject_files: dict[str, dict[str, Path]] = {}

    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        if not _is_vs_mot_file(path):
            continue

        task = _task_from_name(path)
        if task is None:
            continue

        subject_id = _extract_subject_id(path)
        subject_bucket = subject_files.setdefault(subject_id, {})

        existing = subject_bucket.get(task)
        if existing is None or path.stat().st_mtime > existing.stat().st_mtime:
            subject_bucket[task] = path

    return subject_files


def _load_table(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".txt":
        # Some subjects contain a one-line pointer to the real exported CSV path.
        pointer_target = path.read_text(encoding="utf-8").strip()
        if not pointer_target:
            return None

        target_name = Path(pointer_target).name
        local_candidates = list(path.parent.rglob(target_name))
        if not local_candidates:
            local_candidates = list(path.parents[2].rglob(target_name))

        if local_candidates:
            return pd.read_csv(local_candidates[0])

        return None

    return None


def _vs_features(df: pd.DataFrame) -> dict[str, float]:
    clean = df.copy()
    if "correct" not in clean.columns:
        raise KeyError("Visual search file is missing the 'correct' column.")

    clean["correct_num"] = clean["correct"].astype(str).str.lower().map(
        {"true": 1.0, "false": 0.0}
    )
    clean["rt"] = pd.to_numeric(clean.get("rt"), errors="coerce")
    clean["timeout_num"] = (
        clean.get("response", pd.Series(index=clean.index, dtype=object))
        .astype(str)
        .str.lower()
        .eq("timeout")
        .astype(float)
    )

    return {
        "vs_n_trials": float(len(clean)),
        "vs_accuracy_mean": float(clean["correct_num"].mean()),
        "vs_correct_sum": float(clean["correct_num"].sum()),
        "vs_rt_mean": float(clean["rt"].mean()),
        "vs_rt_median": float(clean["rt"].median()),
        "vs_timeout_rate": float(clean["timeout_num"].mean()),
    }


def _mot_features(df: pd.DataFrame) -> dict[str, float]:
    clean = df.copy()
    if "accuracy" not in clean.columns:
        raise KeyError("MOT file is missing the 'accuracy' column.")

    clean["accuracy"] = pd.to_numeric(clean["accuracy"], errors="coerce")
    clean["correct"] = pd.to_numeric(clean.get("correct"), errors="coerce")
    clean["num_targets"] = pd.to_numeric(clean.get("num_targets"), errors="coerce")

    return {
        "mot_n_trials": float(len(clean)),
        "mot_accuracy_mean": float(clean["accuracy"].mean()),
        "mot_accuracy_median": float(clean["accuracy"].median()),
        "mot_correct_sum": float(clean["correct"].sum()),
        "mot_targets_mean": float(clean["num_targets"].mean()),
    }


def build_subject_feature_matrix(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    discovered = _discover_vs_mot_files(raw_dir)

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for subject_id, task_files in sorted(discovered.items()):
        vs_path = task_files.get("vs")
        mot_path = task_files.get("mot")

        if vs_path is None or mot_path is None:
            skipped.append(
                {
                    "subject_id": subject_id,
                    "reason": "missing_vs_or_mot_file",
                    "vs_path": str(vs_path) if vs_path else "",
                    "mot_path": str(mot_path) if mot_path else "",
                }
            )
            continue

        vs_df = _load_table(vs_path)
        mot_df = _load_table(mot_path)

        if vs_df is None or mot_df is None:
            skipped.append(
                {
                    "subject_id": subject_id,
                    "reason": "unreadable_vs_or_mot_file",
                    "vs_path": str(vs_path),
                    "mot_path": str(mot_path),
                }
            )
            continue

        row: dict[str, object] = {
            "subject_id": subject_id,
            "vs_file": str(vs_path),
            "mot_file": str(mot_path),
        }
        row.update(_vs_features(vs_df))
        row.update(_mot_features(mot_df))
        rows.append(row)

    if not rows:
        raise ValueError(
            "No usable VS/MOT subject pairs were found under data/raw. "
            "Expected files inside subject-level vs_mot folders."
        )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def _standardize_features(feature_df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, pd.Series]:
    means = feature_df.mean(axis=0)
    stds = feature_df.std(axis=0).replace(0, 1.0)
    scaled = ((feature_df - means) / stds).fillna(0.0)
    return scaled.to_numpy(dtype=float), means, stds


def _initialize_centers(x: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if x.shape[0] < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} usable subjects for clustering, found {x.shape[0]}."
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(x.shape[0], size=n_clusters, replace=False)
    return x[indices].copy()


def _fit_kmeans(
    x: np.ndarray,
    *,
    n_clusters: int = 2,
    random_seed: int = 42,
    max_iter: int = 100,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    centers = _initialize_centers(x, n_clusters, random_seed)
    rng = np.random.default_rng(random_seed)

    for _ in range(max_iter):
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)

        new_centers = centers.copy()
        for cluster_id in range(n_clusters):
            members = x[labels == cluster_id]
            if len(members) == 0:
                new_centers[cluster_id] = x[rng.integers(0, x.shape[0])]
            else:
                new_centers[cluster_id] = members.mean(axis=0)

        if np.abs(new_centers - centers).max() <= tolerance:
            centers = new_centers
            break
        centers = new_centers

    distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    return labels, centers


def _expert_score(centers_scaled: np.ndarray, columns: list[str]) -> np.ndarray:
    positive = [
        "vs_accuracy_mean",
        "vs_correct_sum",
        "mot_accuracy_mean",
        "mot_accuracy_median",
        "mot_correct_sum",
    ]
    negative = [
        "vs_rt_mean",
        "vs_rt_median",
        "vs_timeout_rate",
    ]

    scores = np.zeros(centers_scaled.shape[0], dtype=float)
    for column in positive:
        if column in columns:
            scores += centers_scaled[:, columns.index(column)]
    for column in negative:
        if column in columns:
            scores -= centers_scaled[:, columns.index(column)]
    return scores


def _label_clusters(centers_scaled: np.ndarray, columns: list[str]) -> dict[int, str]:
    scores = _expert_score(centers_scaled, columns)
    expert_cluster = int(scores.argmax())
    return {
        cluster_id: ("expert" if cluster_id == expert_cluster else "novice")
        for cluster_id in range(len(scores))
    }


def run_vs_mot_kmeans(raw_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    combined, skipped = build_subject_feature_matrix(raw_dir)

    metadata_columns = ["subject_id", "vs_file", "mot_file"]
    feature_columns = [column for column in combined.columns if column not in metadata_columns]
    x_scaled, means, stds = _standardize_features(combined[feature_columns])

    labels, centers_scaled = _fit_kmeans(x_scaled, n_clusters=2, random_seed=42)
    label_map = _label_clusters(centers_scaled, feature_columns)

    subject_clusters = combined.copy()
    subject_clusters["cluster_id"] = labels
    subject_clusters["predicted_expertise"] = subject_clusters["cluster_id"].map(label_map)
    subject_clusters = subject_clusters.sort_values("subject_id").reset_index(drop=True)

    centers_original = pd.DataFrame(
        centers_scaled * stds.to_numpy() + means.to_numpy(),
        columns=feature_columns,
    )
    centers_original.insert(0, "predicted_expertise", [label_map[i] for i in range(2)])
    centers_original.insert(0, "cluster_id", [0, 1])

    summary = (
        subject_clusters.groupby(["predicted_expertise"], as_index=False)
        .agg(subject_count=("subject_id", "nunique"))
        .sort_values("predicted_expertise")
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_path = output_dir / "vs_mot_subject_clusters.csv"
    centroids_path = output_dir / "vs_mot_cluster_centroids.csv"
    summary_path = output_dir / "vs_mot_cluster_summary.csv"
    skipped_path = output_dir / "vs_mot_skipped_subjects.csv"

    subject_clusters.to_csv(subject_path, index=False)
    centers_original.to_csv(centroids_path, index=False)
    summary.to_csv(summary_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    return {
        "subject_clusters": subject_clusters,
        "centroids": centers_original,
        "summary": summary,
        "skipped": skipped,
        "paths": pd.DataFrame(
            {
                "artifact": [
                    "subject_clusters",
                    "centroids",
                    "summary",
                    "skipped",
                ],
                "path": [
                    str(subject_path),
                    str(centroids_path),
                    str(summary_path),
                    str(skipped_path),
                ],
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster subjects as expert or novice using raw VS and MOT files."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Root directory containing subject folders and their vs_mot files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where clustering outputs will be written.",
    )
    args = parser.parse_args()

    results = run_vs_mot_kmeans(Path(args.raw_dir), Path(args.output_dir))

    print("\nPredicted subject expertise:")
    printable = results["subject_clusters"][["subject_id", "predicted_expertise", "cluster_id"]]
    print(printable.to_string(index=False))

    if not results["skipped"].empty:
        print("\nSkipped subjects:")
        print(results["skipped"].to_string(index=False))

    print("\nArtifacts:")
    for _, row in results["paths"].iterrows():
        print(f"{row['artifact']} -> {row['path']}")


if __name__ == "__main__":
    main()
