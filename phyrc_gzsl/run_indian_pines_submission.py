from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from phyrc_gzsl.baseline.evaluate_cada_vae import evaluate_split as evaluate_cada_vae
from phyrc_gzsl.baseline.evaluate_eszsl import evaluate_split as evaluate_eszsl
from phyrc_gzsl.baseline.evaluate_f_clswgan import evaluate_split as evaluate_f_clswgan
from phyrc_gzsl.baseline.evaluate_free import STRICT_FREE, evaluate_split as evaluate_free
from phyrc_gzsl.baseline.evaluate_sae import evaluate_split as evaluate_sae
from phyrc_gzsl.evaluate_phyrc import evaluate_stage2_split, _validate_backbone_partition
from phyrc_gzsl.run_phyrc_multi_unseen import _atomic_json
from phyrc_gzsl.utils.config import class_id_lists, load_config, set_unseen_classes


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "phyrc_gzsl/checkpoints/indian_pines_submission"
CONFIG = "phyrc_gzsl/configs/indian_pines_p1.yaml"
ATTRIBUTES = "phyrc_gzsl/data/processed/IndianPines_structured_attributes.json"
SEEDS = list(range(42, 47))
CLASSES = list(range(1, 17))
MULTI_UNSEEN = [4, 8, 12]
METHODS = ("eszsl", "sae", "f_clswgan", "cada_vae", "free", "phyrc")
METRICS = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
EVALUATORS = {
    "eszsl": evaluate_eszsl, "sae": evaluate_sae,
    "f_clswgan": evaluate_f_clswgan, "cada_vae": evaluate_cada_vae,
    "free": evaluate_free,
}
PILOT = {
    "phyrc": "phyrc_gzsl/checkpoints/indian_pines_phyrc_single_unseen.json",
    "eszsl": "phyrc_gzsl/checkpoints/indian_pines_pilot/eszsl_seed42.json",
    "sae": "phyrc_gzsl/checkpoints/indian_pines_pilot/sae_seed42.json",
    "f_clswgan": "phyrc_gzsl/checkpoints/indian_pines_pilot/f_clswgan_seed42.json",
    "cada_vae": "phyrc_gzsl/checkpoints/indian_pines_pilot/cada_vae_seed42.json",
    "free": "phyrc_gzsl/checkpoints/indian_pines_pilot/free_seed42.json",
}


def _backbone(protocol: str, seed: int, unseen: list[int]) -> str:
    if protocol == "single" and seed == 42:
        return f"phyrc_gzsl/checkpoints/indian_pines_p1_backbone_s{unseen[0]}.pt"
    suffix = f"backbone_s{unseen[0]}.pt" if protocol == "single" else "backbone.pt"
    return f"phyrc_gzsl/checkpoints/indian_pines_submission/{protocol}/seed{seed}/{suffix}"


def _evaluation(protocol: str, method: str, seed: int, unseen: list[int]) -> dict:
    tag = f"s{unseen[0]}" if protocol == "single" else "multi"
    return {
        "protocol": protocol, "method": method, "seed": seed,
        "unseen_classes": unseen, "config": CONFIG, "attributes": ATTRIBUTES,
        "backbone": _backbone(protocol, seed, unseen),
        "result": f"phyrc_gzsl/checkpoints/indian_pines_submission/{protocol}/seed{seed}/{method}_{tag}.json",
    }


def build_manifest() -> dict:
    single = [
        _evaluation("single", method, seed, [unseen])
        for method in METHODS for seed in SEEDS for unseen in CLASSES
    ]
    multi = [
        _evaluation("multi", method, seed, list(MULTI_UNSEEN))
        for method in METHODS for seed in SEEDS
    ]
    return {
        "design": "docs/superpowers/specs/2026-07-20-indian-pines-submission-evidence-design.md",
        "seeds": SEEDS, "single_unseen_classes": CLASSES,
        "multi_unseen_classes": MULTI_UNSEEN,
        "single_evaluations": single, "multi_evaluations": multi,
    }


def _partition(unseen: list[int]) -> tuple[list[int], list[int]]:
    config = load_config(ROOT / CONFIG)
    set_unseen_classes(config, unseen)
    seen, unseen, _ = class_id_lists(config)
    return seen, unseen


def _checkpoint_ok(path: Path, unseen: list[int]) -> bool:
    if not path.is_file():
        return False
    try:
        _validate_backbone_partition(path, _partition(unseen)[0])
        return True
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _train(path: Path, unseen: list[int], seed: int) -> None:
    subprocess.run([
        sys.executable, "-m", "phyrc_gzsl.train_spectral_mae", "--config", CONFIG,
        "--backbone", "spectral_morphology", "--checkpoint", str(path),
        "--unseen-classes", *map(str, unseen), "--seed", str(seed),
    ], cwd=ROOT, check=True)


def _complete(job: dict) -> bool:
    try:
        payload = json.loads((ROOT / job["result"]).read_text(encoding="utf-8"))
        metrics = payload["result"]["metrics"]
        return (
            payload["protocol"] == job["protocol"] and payload["method"] == job["method"]
            and payload["seed"] == job["seed"]
            and payload["unseen_classes"] == job["unseen_classes"]
            and all(math.isfinite(float(metrics[key])) for key in METRICS)
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _wrap(job: dict, result: dict) -> dict:
    seen, unseen = _partition(job["unseen_classes"])
    return {
        "protocol": job["protocol"], "method": job["method"], "seed": job["seed"],
        "seen_classes": seen, "unseen_classes": unseen, "result": result,
    }


def _import_seed42(jobs: list[dict]) -> None:
    for method, source in PILOT.items():
        payload = json.loads((ROOT / source).read_text(encoding="utf-8"))
        rows = {int(row["unseen"]): row for row in payload["results"]}
        for job in jobs:
            if job["method"] == method and job["seed"] == 42 and not _complete(job):
                _atomic_json(ROOT / job["result"], _wrap(job, rows[job["unseen_classes"][0]]))


def _evaluate(job: dict) -> dict:
    seen, unseen = _partition(job["unseen_classes"])
    checkpoint = ROOT / job["backbone"]
    _validate_backbone_partition(checkpoint, seen)
    if job["method"] == "phyrc":
        result = evaluate_stage2_split(
            ROOT / CONFIG, checkpoint, ROOT / ATTRIBUTES, ROOT,
            unseen[0] if len(unseen) == 1 else unseen, job["seed"], "full",
        )
    else:
        kwargs = STRICT_FREE if job["method"] == "free" else {}
        result = EVALUATORS[job["method"]](
            ROOT / CONFIG, checkpoint, ROOT / ATTRIBUTES, ROOT,
            unseen[0] if len(unseen) == 1 else unseen, job["seed"], **kwargs,
        )
    return _wrap(job, result)


def _aggregate(jobs: list[dict]) -> dict:
    payloads = [
        json.loads((ROOT / job["result"]).read_text(encoding="utf-8"))
        for job in jobs if _complete(job)
    ]
    methods = {}
    for method in METHODS:
        seed_rows = []
        for seed in SEEDS:
            rows = [row for row in payloads if row["method"] == method and row["seed"] == seed]
            if not rows:
                continue
            seed_rows.append({
                "seed": seed,
                **{key: float(np.mean([row["result"]["metrics"][key] for row in rows])) for key in METRICS},
            })
        methods[method] = {
            "seed_results": seed_rows,
            **({
                key: {
                    "mean": float(np.mean([row[key] for row in seed_rows])),
                    "std": float(np.std([row[key] for row in seed_rows], ddof=0)),
                } for key in METRICS
            } if seed_rows else {}),
        }
    return {"completed_jobs": len(payloads), "expected_jobs": len(jobs), "methods": methods}


def _run(protocol: str, jobs: list[dict]) -> None:
    unique_backbones = {(job["backbone"], tuple(job["unseen_classes"]), job["seed"]) for job in jobs}
    for index, (relative, unseen, seed) in enumerate(sorted(unique_backbones), start=1):
        path = ROOT / relative
        if not _checkpoint_ok(path, list(unseen)):
            print(f"[{protocol}] train backbone {index}/{len(unique_backbones)} seed={seed} unseen={list(unseen)}", flush=True)
            _train(path, list(unseen), seed)
    if protocol == "single":
        _import_seed42(jobs)
    for index, job in enumerate(jobs, start=1):
        if _complete(job):
            continue
        print(f"[{protocol}] evaluate {index}/{len(jobs)} {job['method']} seed={job['seed']} unseen={job['unseen_classes']}", flush=True)
        _atomic_json(ROOT / job["result"], _evaluate(job))
        _atomic_json(OUTPUT / f"{protocol}_summary.json", _aggregate(jobs))
    _atomic_json(OUTPUT / f"{protocol}_summary.json", _aggregate(jobs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Indian Pines five-seed submission evidence.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--phase", choices=("single", "multi", "all"), default="all")
    args = parser.parse_args()
    manifest = build_manifest()
    _atomic_json(OUTPUT / "manifest.json", manifest)
    if not args.execute:
        print(json.dumps({"single": 480, "multi": 30, "manifest": str(OUTPUT / "manifest.json")}, indent=2))
        return
    if args.phase in ("single", "all"):
        _run("single", manifest["single_evaluations"])
    if args.phase in ("multi", "all"):
        _run("multi", manifest["multi_evaluations"])


if __name__ == "__main__":
    main()
