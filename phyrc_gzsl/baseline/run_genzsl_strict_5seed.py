from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phyrc_gzsl.baseline.evaluate_genzsl_strict import OFFICIAL_COMMIT, STRICT_GENZSL, STRICT_PROTOCOL, evaluate_split
from phyrc_gzsl.diagnose_rstd_ot import resolve_project_path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "phyrc_gzsl/checkpoints/baselines/genzsl_strict"
DATASETS = {
    "paviau": ("phyrc_gzsl/configs/paviau_p1.yaml", "phyrc_gzsl/data/processed/PaviaU_structured_attributes.json", range(1, 10)),
    "houston": ("phyrc_gzsl/configs/houston_p1.yaml", "phyrc_gzsl/data/processed/Houston_structured_attributes.json", range(1, 16)),
    "longkou": ("phyrc_gzsl/configs/longkou_p1.yaml", "phyrc_gzsl/data/processed/LongKou_structured_attributes.json", range(1, 10)),
}


def checkpoint_pattern(dataset: str, seed: int) -> str:
    name = f"{dataset}_p1_backbone_s{{unseen}}.pt"
    return f"phyrc_gzsl/checkpoints/{name}" if seed == 42 else f"phyrc_gzsl/checkpoints/multiseed/{dataset}/seed{seed}/{name}"


def complete(path: Path, count: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["protocol"]["strict_inductive"] is True and len(payload["results"]) == count
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return False


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dataset, (config, attributes, classes) in DATASETS.items():
        class_ids = list(classes)
        for seed in range(42, 47):
            output = OUTPUT / f"{dataset}_seed{seed}.json"
            if complete(output, len(class_ids)): continue
            rows = [evaluate_split(resolve_project_path(ROOT, config), resolve_project_path(ROOT, checkpoint_pattern(dataset, seed).format(unseen=cls)), resolve_project_path(ROOT, attributes), ROOT, cls, seed, **STRICT_GENZSL) for cls in class_ids]
            keys = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
            summary = {key: float(np.mean([row["metrics"][key] for row in rows])) for key in keys}
            output.write_text(json.dumps({"summary": summary, "results": rows, "protocol": STRICT_PROTOCOL, "hyperparameters": STRICT_GENZSL, "official_commit": OFFICIAL_COMMIT}, indent=2), encoding="utf-8")
            print(json.dumps({"dataset": dataset, "seed": seed, **summary}), flush=True)


if __name__ == "__main__": main()
