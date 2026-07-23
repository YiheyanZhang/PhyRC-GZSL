from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from baseline.evaluate_cada_vae import evaluate_split as evaluate_cada_vae
from baseline.evaluate_eszsl import evaluate_split as evaluate_eszsl
from baseline.evaluate_f_clswgan import evaluate_split as evaluate_f_clswgan
from baseline.evaluate_free import STRICT_FREE, evaluate_split as evaluate_free
from baseline.evaluate_sae import evaluate_split as evaluate_sae
from evaluate_phyrc import _validate_backbone_partition
from run_phyrc_multi_unseen import _atomic_json, _partition


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "checkpoints/multi_unseen/baselines"
METRICS = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
METHODS = ("eszsl", "sae", "f_clswgan", "cada_vae", "free")
EVALUATORS = {
    "eszsl": evaluate_eszsl,
    "sae": evaluate_sae,
    "f_clswgan": evaluate_f_clswgan,
    "cada_vae": evaluate_cada_vae,
    "free": evaluate_free,
}


def build_jobs() -> list[dict]:
    full_jobs = json.loads(
        (ROOT / "checkpoints/multi_unseen/manifest.json").read_text(encoding="utf-8")
    )["jobs"]
    return [
        {
            **job,
            "method": method,
            "result": f"checkpoints/multi_unseen/baselines/{method}/{job['dataset']}/seed{job['seed']}.json",
        }
        for method in METHODS for job in full_jobs
    ]


def _complete(job: dict) -> bool:
    try:
        payload = json.loads((ROOT / job["result"]).read_text(encoding="utf-8"))
        metrics = payload["result"]["metrics"]
        return (
            payload["method"] == job["method"]
            and payload["dataset"] == job["dataset"]
            and int(payload["seed"]) == job["seed"]
            and payload["unseen_classes"] == job["unseen_classes"]
            and all(math.isfinite(float(metrics[key])) for key in METRICS)
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _evaluate(job: dict) -> dict:
    seen, unseen = _partition(job)
    _validate_backbone_partition(ROOT / job["backbone"], seen)
    kwargs = STRICT_FREE if job["method"] == "free" else {}
    result = EVALUATORS[job["method"]](
        ROOT / job["config"], ROOT / job["backbone"], ROOT / job["attributes"],
        ROOT, unseen, job["seed"], **kwargs,
    )
    return {
        "method": job["method"], "dataset": job["dataset"], "seed": job["seed"],
        "seen_classes": seen, "unseen_classes": unseen, "result": result,
    }


def _summary(jobs: list[dict]) -> dict:
    payloads = [
        json.loads((ROOT / job["result"]).read_text(encoding="utf-8"))
        for job in jobs if _complete(job)
    ]
    methods = {}
    for method in METHODS:
        datasets = {}
        for dataset in sorted({row["dataset"] for row in payloads if row["method"] == method}):
            rows = [row for row in payloads if row["method"] == method and row["dataset"] == dataset]
            datasets[dataset] = {
                key: {
                    "mean": float(np.mean([row["result"]["metrics"][key] for row in rows])),
                    "std": float(np.std([row["result"]["metrics"][key] for row in rows], ddof=0)),
                }
                for key in METRICS
            }
            datasets[dataset]["seeds"] = sorted(row["seed"] for row in rows)
        methods[method] = datasets
    return {"completed_jobs": len(payloads), "expected_jobs": len(jobs), "methods": methods}


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled multi-Unseen baseline comparison.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    jobs = build_jobs()
    _atomic_json(OUTPUT / "manifest.json", {
        "source_manifest": "checkpoints/multi_unseen/manifest.json", "jobs": jobs,
    })
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "jobs": len(jobs)}, indent=2))
        return
    for index, job in enumerate(jobs, start=1):
        if _complete(job):
            print(f"[{index}/{len(jobs)}] resume {job['method']} {job['dataset']} seed {job['seed']}", flush=True)
            continue
        print(f"[{index}/{len(jobs)}] run {job['method']} {job['dataset']} seed {job['seed']}", flush=True)
        _atomic_json(ROOT / job["result"], _evaluate(job))
        _atomic_json(OUTPUT / "summary.json", _summary(jobs))
    _atomic_json(OUTPUT / "summary.json", _summary(jobs))


if __name__ == "__main__":
    main()
