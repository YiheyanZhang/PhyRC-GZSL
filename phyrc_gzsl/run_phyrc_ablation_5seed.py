from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyrc_gzsl.diagnose_rstd_ot import resolve_project_path
from phyrc_gzsl.evaluate_phyrc import evaluate_stage2_split


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "phyrc_gzsl/checkpoints/ablations/phyrc"
VARIANTS = (
    "no_relational_prototype", "no_cross_fitting",
    "no_dual_evidence", "no_risk_constraint",
)
DATASETS = {
    "paviau": ("phyrc_gzsl/configs/paviau_p1.yaml", "phyrc_gzsl/data/processed/PaviaU_structured_attributes.json", range(1, 10)),
    "houston": ("phyrc_gzsl/configs/houston_p1.yaml", "phyrc_gzsl/data/processed/Houston_structured_attributes.json", range(1, 16)),
    "longkou": ("phyrc_gzsl/configs/longkou_p1.yaml", "phyrc_gzsl/data/processed/LongKou_structured_attributes.json", range(1, 10)),
    "indian_pines": ("phyrc_gzsl/configs/indian_pines_p1.yaml", "phyrc_gzsl/data/processed/IndianPines_structured_attributes.json", range(1, 17)),
}
METRICS = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")


def checkpoint(dataset: str, seed: int, unseen: int) -> Path:
    if dataset == "indian_pines" and seed != 42:
        return ROOT / f"phyrc_gzsl/checkpoints/indian_pines_submission/single/seed{seed}/backbone_s{unseen}.pt"
    name = f"{dataset}_p1_backbone_s{unseen}.pt"
    relative = f"phyrc_gzsl/checkpoints/{name}" if seed == 42 else f"phyrc_gzsl/checkpoints/multiseed/{dataset}/seed{seed}/{name}"
    return resolve_project_path(ROOT, relative)


def complete(path: Path, count: int, variant: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["ablation"] == variant and len(payload["results"]) == count
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return False


def full_result_paths(dataset: str, seed: int) -> list[Path]:
    if dataset == "indian_pines":
        return [ROOT / f"phyrc_gzsl/checkpoints/indian_pines_submission/single/seed{seed}/phyrc_s{unseen}.json" for unseen in range(1, 17)]
    if seed == 42:
        return [ROOT / f"phyrc_gzsl/checkpoints/{dataset}_phyrc_single_unseen.json"]
    return [ROOT / f"phyrc_gzsl/checkpoints/multiseed/{dataset}/seed{seed}/{dataset}_phyrc.json"]


def _full_seed_summary(dataset: str, seed: int) -> dict:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in full_result_paths(dataset, seed)]
    if len(payloads) == 1:
        return payloads[0]["summary"]
    results = [payload["result"] for payload in payloads]
    summary = {key: float(np.mean([row["metrics"][key] for row in results])) for key in METRICS}
    summary["seen_zero"] = int(sum(
        sum(float(value) == 0.0 for index, value in row["metrics"]["per_class"].items() if int(index) < len(payload["seen_classes"]))
        for payload, row in zip(payloads, results)
    ))
    return summary


def write_summary() -> None:
    table = {}
    for variant in ("full",) + VARIANTS:
        table[variant] = {}
        for dataset in DATASETS:
            if variant == "full":
                paths = [path for seed in range(42, 47) for path in full_result_paths(dataset, seed)]
                if not all(path.is_file() for path in paths): continue
                rows = [_full_seed_summary(dataset, seed) for seed in range(42, 47)]
            else:
                paths = [OUTPUT / variant / f"{dataset}_seed{seed}.json" for seed in range(42, 47)]
                if not all(path.is_file() for path in paths): continue
                rows = [json.loads(path.read_text(encoding="utf-8"))["summary"] for path in paths]
            table[variant][dataset] = {
                key: {"mean": float(np.mean([row[key] for row in rows])), "std": float(np.std([row[key] for row in rows], ddof=0))}
                for key in METRICS
            }
            table[variant][dataset]["seen_zero"] = int(sum(row["seen_zero"] for row in rows))
    (OUTPUT / "summary.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        folder = OUTPUT / variant; folder.mkdir(exist_ok=True)
        for dataset, (config, attributes, classes) in DATASETS.items():
            class_ids = list(classes)
            for seed in range(42, 47):
                output = folder / f"{dataset}_seed{seed}.json"
                if complete(output, len(class_ids), variant): continue
                rows = [evaluate_stage2_split(
                    resolve_project_path(ROOT, config), checkpoint(dataset, seed, unseen),
                    resolve_project_path(ROOT, attributes), ROOT, unseen, seed, variant,
                ) for unseen in class_ids]
                summary = {key: float(np.mean([row["metrics"][key] for row in rows])) for key in METRICS}
                summary["seen_zero"] = int(sum(sum(value == 0.0 for index, value in row["metrics"]["per_class"].items() if index < len(row["metrics"]["per_class"]) - 1) for row in rows))
                output.write_text(json.dumps({"ablation": variant, "summary": summary, "results": rows}, indent=2), encoding="utf-8")
                print(json.dumps({"variant": variant, "dataset": dataset, "seed": seed, **summary}), flush=True)
                write_summary()
    write_summary()


if __name__ == "__main__": main()
