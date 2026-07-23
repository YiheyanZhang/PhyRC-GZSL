"""Descriptive paired uncertainty analysis for the formal PhyRC results."""

import json
from pathlib import Path

import numpy as np


DATASETS = {
    "paviau": {
        "display_name": "PaviaU",
        "baseline": "cada_vae",
        "baseline_name": "CADA-VAE",
        "class_names": [
            "Asphalt", "Meadows", "Gravel", "Trees", "Painted Metal Sheets",
            "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows",
        ],
    },
    "houston": {
        "display_name": "Houston",
        "baseline": "cada_vae",
        "baseline_name": "CADA-VAE",
        "class_names": [
            "Healthy Grass", "Stressed Grass", "Synthetic Grass", "Trees", "Soil",
            "Water", "Residential", "Commercial", "Road", "Highway", "Railway",
            "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track",
        ],
    },
    "longkou": {
        "display_name": "LongKou",
        "baseline": "eszsl",
        "baseline_name": "ESZSL",
        "class_names": [
            "Corn", "Cotton", "Sesame", "Broad-leaf soybean",
            "Narrow-leaf soybean", "Rice", "Water", "Roads and houses", "Mixed weed",
        ],
    },
    "indian_pines": {
        "display_name": "Indian Pines",
        "baseline": "eszsl",
        "baseline_name": "ESZSL",
        "class_names": [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
            "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
            "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
            "Soybean-clean", "Wheat", "Woods",
            "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
        ],
    },
}
SEEDS = range(42, 47)


def hierarchical_paired_bootstrap(differences, resamples=20_000, seed=2027):
    """Bootstrap seeds, then held-out classes within each sampled seed."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 2:
        raise ValueError("differences must be a two-dimensional seed-by-class matrix")

    rng = np.random.default_rng(seed)
    seed_count, class_count = differences.shape
    estimates = np.empty(resamples)
    for index in range(resamples):
        sampled_seeds = rng.integers(seed_count, size=seed_count)
        sampled_classes = rng.integers(class_count, size=(seed_count, class_count))
        estimates[index] = np.mean(
            differences[sampled_seeds[:, None], sampled_classes]
        )

    return {
        "mean": float(differences.mean()),
        "ci95": [float(value) for value in np.percentile(estimates, [2.5, 97.5])],
    }


def align_metric_rows(phyrc_rows, baseline_rows, metric):
    """Return paired metric differences ordered by held-out class identifier."""
    def index(rows):
        indexed = {int(row["unseen"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("duplicate held-out class")
        return indexed

    phyrc, baseline = index(phyrc_rows), index(baseline_rows)
    if phyrc.keys() != baseline.keys():
        raise ValueError("held-out classes differ between paired results")
    unseen = sorted(phyrc)
    differences = np.array([
        phyrc[class_id]["metrics"][metric] - baseline[class_id]["metrics"][metric]
        for class_id in unseen
    ])
    return unseen, differences


def _load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _phyrc_path(checkpoints, dataset, seed):
    if seed == 42:
        return checkpoints / f"{dataset}_phyrc_single_unseen.json"
    return checkpoints / "multiseed" / dataset / f"seed{seed}" / f"{dataset}_phyrc.json"


def _method_rows(checkpoints, dataset, method, seed):
    if dataset == "indian_pines":
        folder = checkpoints / "indian_pines_submission" / "single" / f"seed{seed}"
        return [
            _load(folder / f"{method}_s{class_id}.json")["result"]
            for class_id in range(1, 17)
        ]
    if method == "phyrc":
        return _load(_phyrc_path(checkpoints, dataset, seed))["results"]
    return _load(checkpoints / "baselines" / method / f"{dataset}_seed{seed}.json")["results"]


def build_analysis(project_root):
    checkpoints = Path(project_root) / "checkpoints"
    analysis = {
        "protocol": {
            "status": "descriptive post-hoc uncertainty analysis; not a preregistered confirmatory test",
            "pairing": "seed and held-out class",
            "bootstrap": "resample seeds, then held-out classes within each sampled seed",
            "resamples": 20_000,
            "rng_seed": 2027,
            "baseline_rule": "strongest displayed controlled baseline by aggregate mean H",
        },
        "datasets": {},
    }

    for dataset, metadata in DATASETS.items():
        seed_differences = []
        unseen_accuracy = []
        class_ids = None
        for seed in SEEDS:
            phyrc = _method_rows(checkpoints, dataset, "phyrc", seed)
            baseline = _method_rows(checkpoints, dataset, metadata["baseline"], seed)
            current_ids, differences = align_metric_rows(
                phyrc, baseline, "H"
            )
            if class_ids is not None and current_ids != class_ids:
                raise ValueError(f"held-out classes differ across seeds for {dataset}")
            class_ids = current_ids
            seed_differences.append(differences)
            by_id = {int(row["unseen"]): row for row in phyrc}
            unseen_accuracy.append([
                by_id[class_id]["metrics"]["Unseen_AA"] for class_id in class_ids
            ])

        differences = np.asarray(seed_differences)
        unseen_accuracy = np.asarray(unseen_accuracy)
        bootstrap = hierarchical_paired_bootstrap(differences)
        per_class = []
        for column, class_id in enumerate(class_ids):
            values = unseen_accuracy[:, column]
            per_class.append({
                "class_id": class_id,
                "class_name": metadata["class_names"][class_id - 1],
                "mean_unseen_aa": float(values.mean()),
                "std_unseen_aa": float(values.std()),
                "min_unseen_aa": float(values.min()),
                "max_unseen_aa": float(values.max()),
            })

        analysis["datasets"][dataset] = {
            "display_name": metadata["display_name"],
            "baseline": metadata["baseline_name"],
            "seeds": list(SEEDS),
            "paired_units": int(differences.size),
            "mean_h_difference": bootstrap["mean"],
            "h_difference_ci95": bootstrap["ci95"],
            "phyrc_win_rate": float(np.mean(differences > 0)),
            "tie_rate": float(np.mean(differences == 0)),
            "hardest_three": sorted(per_class, key=lambda row: row["mean_unseen_aa"])[:3],
            "per_class": per_class,
        }
    return analysis


def main():
    project_root = Path(__file__).resolve().parent
    analysis = build_analysis(project_root)
    expected = {
        "paviau": 6.16, "houston": 9.51, "longkou": 10.63,
        "indian_pines": 18.12,
    }
    for dataset, difference in expected.items():
        actual = analysis["datasets"][dataset]["mean_h_difference"]
        if round(actual, 2) != difference:
            raise ValueError(f"{dataset} mean H difference is {actual:.2f}, expected {difference:.2f}")

    output = project_root / "checkpoints" / "statistics" / "phyrc_paired_bootstrap.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    for row in analysis["datasets"].values():
        lower, upper = row["h_difference_ci95"]
        print(
            f'{row["display_name"]}: delta H={row["mean_h_difference"]:.2f}, '
            f'95% CI [{lower:.2f}, {upper:.2f}], win={100 * row["phyrc_win_rate"]:.1f}%'
        )


if __name__ == "__main__":
    main()
