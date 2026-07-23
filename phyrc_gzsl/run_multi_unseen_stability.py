from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from phyrc_gzsl.baseline.evaluate_eszsl import evaluate_split as evaluate_eszsl
from phyrc_gzsl.evaluate_phyrc import evaluate_stage2_split, _validate_backbone_partition
from phyrc_gzsl.run_phyrc_multi_unseen import _atomic_json
from phyrc_gzsl.utils.config import class_id_lists, load_config, set_unseen_classes


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "phyrc_gzsl/checkpoints/multi_unseen_stability"
METRICS = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
DATASETS = {
    "paviau": {
        "config": "phyrc_gzsl/configs/paviau_p1.yaml",
        "attributes": "phyrc_gzsl/data/processed/PaviaU_structured_attributes.json",
        "groups": [[2, 5], [4, 7]],
    },
    "houston": {
        "config": "phyrc_gzsl/configs/houston_p1.yaml",
        "attributes": "phyrc_gzsl/data/processed/Houston_structured_attributes.json",
        "groups": [[2, 6, 10], [4, 8, 12]],
    },
    "indian_pines": {
        "config": "phyrc_gzsl/configs/indian_pines_p1.yaml",
        "attributes": "phyrc_gzsl/data/processed/IndianPines_structured_attributes.json",
        "groups": [[3, 7, 11], [5, 9, 13]],
    },
}


def build_jobs() -> list[dict]:
    jobs = []
    for dataset, settings in DATASETS.items():
        for group, unseen in enumerate(settings["groups"], start=1):
            for seed in range(42, 47):
                folder = f"phyrc_gzsl/checkpoints/multi_unseen_stability/{dataset}/group{group}/seed{seed}"
                jobs.append({
                    "dataset": dataset, "group": group, "seed": seed,
                    "unseen_classes": list(unseen), "config": settings["config"],
                    "attributes": settings["attributes"],
                    "backbone": f"{folder}/backbone.pt",
                    "results": {"phyrc": f"{folder}/phyrc.json", "eszsl": f"{folder}/eszsl.json"},
                })
    return jobs


def _partition(job: dict) -> tuple[list[int], list[int]]:
    config = load_config(ROOT / job["config"])
    set_unseen_classes(config, job["unseen_classes"])
    seen, unseen, _ = class_id_lists(config)
    return seen, unseen


def _checkpoint_complete(job: dict) -> bool:
    path = ROOT / job["backbone"]
    if not path.is_file():
        return False
    try:
        _validate_backbone_partition(path, _partition(job)[0])
        return True
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _result_complete(job: dict, method: str) -> bool:
    try:
        payload = json.loads((ROOT / job["results"][method]).read_text(encoding="utf-8"))
        metrics = payload["result"]["metrics"]
        return (
            payload["method"] == method and payload["dataset"] == job["dataset"]
            and payload["group"] == job["group"] and payload["seed"] == job["seed"]
            and payload["unseen_classes"] == job["unseen_classes"]
            and all(math.isfinite(float(metrics[key])) for key in METRICS)
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _train(job: dict) -> None:
    subprocess.run([
        sys.executable, "-m", "phyrc_gzsl.train_spectral_mae", "--config", job["config"],
        "--backbone", "spectral_morphology", "--checkpoint", str(ROOT / job["backbone"]),
        "--unseen-classes", *map(str, job["unseen_classes"]), "--seed", str(job["seed"]),
    ], cwd=ROOT, check=True)


def _evaluate(job: dict, method: str) -> dict:
    seen, unseen = _partition(job)
    _validate_backbone_partition(ROOT / job["backbone"], seen)
    if method == "phyrc":
        result = evaluate_stage2_split(
            ROOT / job["config"], ROOT / job["backbone"], ROOT / job["attributes"],
            ROOT, unseen, job["seed"], "full",
        )
    else:
        result = evaluate_eszsl(
            ROOT / job["config"], ROOT / job["backbone"], ROOT / job["attributes"],
            ROOT, unseen, job["seed"],
        )
    return {
        "method": method, "dataset": job["dataset"], "group": job["group"],
        "seed": job["seed"], "seen_classes": seen, "unseen_classes": unseen, "result": result,
    }


def _summary(jobs: list[dict]) -> dict:
    output = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for group in (1, 2):
            output[dataset][f"group{group}"] = {}
            for method in ("phyrc", "eszsl"):
                rows = [
                    json.loads((ROOT / job["results"][method]).read_text(encoding="utf-8"))
                    for job in jobs if job["dataset"] == dataset and job["group"] == group
                    and _result_complete(job, method)
                ]
                output[dataset][f"group{group}"][method] = {
                    "seeds": sorted(row["seed"] for row in rows),
                    **({
                        key: {
                            "mean": float(np.mean([row["result"]["metrics"][key] for row in rows])),
                            "std": float(np.std([row["result"]["metrics"][key] for row in rows], ddof=0)),
                        } for key in METRICS
                    } if rows else {}),
                }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Two fixed shifted multi-unseen stability groups.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    jobs = build_jobs()
    _atomic_json(OUTPUT / "manifest.json", {
        "selection_rule": "existing equally spaced class-ID group shifted by -1 and +1",
        "methods": ["phyrc", "eszsl"], "jobs": jobs,
    })
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "jobs": len(jobs)}, indent=2))
        return
    for index, job in enumerate(jobs, start=1):
        if not _checkpoint_complete(job):
            print(f"[{index}/{len(jobs)}] train {job['dataset']} group={job['group']} seed={job['seed']}", flush=True)
            _train(job)
        for method in ("phyrc", "eszsl"):
            if not _result_complete(job, method):
                print(f"[{index}/{len(jobs)}] evaluate {method}", flush=True)
                _atomic_json(ROOT / job["results"][method], _evaluate(job, method))
        _atomic_json(OUTPUT / "summary.json", _summary(jobs))


if __name__ == "__main__":
    main()
