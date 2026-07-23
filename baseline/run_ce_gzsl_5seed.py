from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "checkpoints/baselines/ce_gzsl"
DATASETS = (
    ("paviau", "configs/paviau_p1.yaml", "data/processed/PaviaU_structured_attributes.json", "checkpoints/paviau_p1_backbone_s{unseen}.pt", range(1, 10)),
    ("houston", "configs/houston_p1.yaml", "data/processed/Houston_structured_attributes.json", "checkpoints/houston_p1_backbone_s{unseen}.pt", range(1, 16)),
    ("longkou", "configs/longkou_p1.yaml", "data/processed/LongKou_structured_attributes.json", "checkpoints/longkou_p1_backbone_s{unseen}.pt", range(1, 10)),
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, config, attributes, pattern, classes in DATASETS:
        jobs = []
        for seed in range(42, 47):
            stdout = (OUTPUT / f"{name}_seed{seed}.stdout.log").open("w", encoding="utf-8")
            stderr = (OUTPUT / f"{name}_seed{seed}.stderr.log").open("w", encoding="utf-8")
            command = [
                sys.executable, "-u", "baseline/evaluate_ce_gzsl.py",
                "--config", config, "--attributes", attributes,
                "--unseen-classes", *map(str, classes),
                "--backbone-pattern", pattern, "--seed", str(seed),
                "--output", f"checkpoints/baselines/ce_gzsl/{name}_seed{seed}.json",
                "--gan-epochs", "100", "--classifier-epochs", "25",
                "--synthetic-per-class", "100", "--batch-size", "2048",
                "--hidden-dim", "1024", "--embedding-dim", "512",
                "--projection-dim", "128", "--relation-hidden-dim", "512",
            ]
            jobs.append((seed, subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr), stdout, stderr))
        for seed, process, stdout, stderr in jobs:
            returncode = process.wait()
            stdout.close()
            stderr.close()
            if returncode:
                details = (OUTPUT / f"{name}_seed{seed}.stderr.log").read_text(encoding="utf-8")
                raise RuntimeError(f"{name} seed {seed} failed: {details}")
        print(f"completed {name} seeds 42-46", flush=True)


if __name__ == "__main__":
    main()
