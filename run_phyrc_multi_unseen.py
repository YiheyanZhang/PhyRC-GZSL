from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from evaluate_phyrc import evaluate_stage2_split, _validate_backbone_partition
from utils.config import load_config, set_unseen_classes, class_id_lists


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "checkpoints/multi_unseen"
METRICS = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
DATASETS = {
    "paviau": {
        "config": "configs/paviau_p1.yaml",
        "attributes": "data/processed/PaviaU_structured_attributes.json",
        "unseen_classes": [3, 6],
    },
    "houston": {
        "config": "configs/houston_p1.yaml",
        "attributes": "data/processed/Houston_structured_attributes.json",
        "unseen_classes": [3, 7, 11],
    },
    "longkou": {
        "config": "configs/longkou_p1.yaml",
        "attributes": "data/processed/LongKou_structured_attributes.json",
        "unseen_classes": [3, 6],
    },
}


def build_jobs() -> list[dict]:
    jobs = []
    for dataset, settings in DATASETS.items():
        for seed in range(42, 47):
            folder = f"checkpoints/multi_unseen/{dataset}/seed{seed}"
            jobs.append({
                "dataset": dataset,
                "seed": seed,
                "unseen_classes": list(settings["unseen_classes"]),
                "config": settings["config"],
                "attributes": settings["attributes"],
                "backbone": f"{folder}/backbone.pt",
                "result": f"{folder}/result.json",
            })
    return jobs


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


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


def _result_complete(job: dict) -> bool:
    try:
        payload = json.loads((ROOT / job["result"]).read_text(encoding="utf-8"))
        metrics = payload["result"]["metrics"]
        return (
            payload["dataset"] == job["dataset"]
            and int(payload["seed"]) == job["seed"]
            and payload["unseen_classes"] == job["unseen_classes"]
            and payload["result"]["ablation"] == "full"
            and all(math.isfinite(float(metrics[key])) for key in METRICS)
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def aggregate_results(payloads: list[dict]) -> dict:
    summary = {}
    for dataset in sorted({payload["dataset"] for payload in payloads}):
        rows = sorted(
            (payload for payload in payloads if payload["dataset"] == dataset),
            key=lambda payload: payload["seed"],
        )
        entry = {"seeds": [row["seed"] for row in rows]}
        for key in METRICS:
            values = [float(row["result"]["metrics"][key]) for row in rows]
            entry[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
        seen = rows[0]["seen_classes"]
        unseen = rows[0]["unseen_classes"]
        per_class = {}
        for offset, class_id in enumerate(unseen, start=len(seen)):
            values = [
                float(row["result"]["metrics"]["per_class"][offset])
                if offset in row["result"]["metrics"]["per_class"]
                else float(row["result"]["metrics"]["per_class"][str(offset)])
                for row in rows
            ]
            per_class[str(class_id)] = {
                "mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
            }
        entry["unseen_per_class"] = per_class
        summary[dataset] = entry
    return summary


def _train(job: dict) -> None:
    command = [
        sys.executable, "-m", "train_spectral_mae",
        "--config", job["config"],
        "--backbone", "spectral_morphology",
        "--checkpoint", str(ROOT / job["backbone"]),
        "--unseen-classes", *map(str, job["unseen_classes"]),
        "--seed", str(job["seed"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def _evaluate(job: dict) -> dict:
    seen, unseen = _partition(job)
    result = evaluate_stage2_split(
        ROOT / job["config"], ROOT / job["backbone"], ROOT / job["attributes"],
        ROOT, unseen, job["seed"], "full",
    )
    return {
        "dataset": job["dataset"],
        "seed": job["seed"],
        "seen_classes": seen,
        "unseen_classes": unseen,
        "result": result,
    }


def _write_summary(jobs: list[dict]) -> None:
    payloads = [
        json.loads((ROOT / job["result"]).read_text(encoding="utf-8"))
        for job in jobs if _result_complete(job)
    ]
    _atomic_json(OUTPUT / "summary.json", {
        "completed_jobs": len(payloads),
        "expected_jobs": len(jobs),
        "datasets": aggregate_results(payloads),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Preregistered Full-only multi-Unseen stress test.")
    parser.add_argument("--execute", action="store_true", help="train and evaluate the fixed 15 jobs")
    args = parser.parse_args()
    jobs = build_jobs()
    _atomic_json(OUTPUT / "manifest.json", {
        "design": "docs/superpowers/specs/2026-07-19-phyrc-multi-unseen-stress-test-design.md",
        "jobs": jobs,
    })
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "jobs": len(jobs), "manifest": str(OUTPUT / "manifest.json")}, indent=2))
        return
    for index, job in enumerate(jobs, start=1):
        if _result_complete(job):
            print(f"[{index}/{len(jobs)}] resume {job['dataset']} seed {job['seed']}", flush=True)
            continue
        print(f"[{index}/{len(jobs)}] run {job['dataset']} seed {job['seed']}", flush=True)
        if not _checkpoint_complete(job):
            _train(job)
        _validate_backbone_partition(ROOT / job["backbone"], _partition(job)[0])
        _atomic_json(ROOT / job["result"], _evaluate(job))
        _write_summary(jobs)
    _write_summary(jobs)


if __name__ == "__main__":
    main()
