from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import class_names_from_config, load_config
from utils.text_encoder import SemanticConditionEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="checkpoints/paviau_sensor_aligned_conditions.pt")
    parser.add_argument("--class-id", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(args.config)
    config["text"] = dict(config["text"])
    config["text"].update({
        "semantic_mode": "sensor_aligned",
        "wavelength_nm": [430, 860],
        "cache_file": "data/processed/PaviaU_sensor_aligned_descriptions.json",
        "refresh_cache": True,
    })
    output = root / args.output if not Path(args.output).is_absolute() else Path(args.output)
    class_names = class_names_from_config(config)
    if args.class_id is not None:
        class_names = {args.class_id: class_names[args.class_id]}
    conditions = SemanticConditionEncoder(config, root).encode_classes(class_names)
    if args.class_id is not None and output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        existing.update(conditions)
        conditions = existing
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(conditions, output)
    print("saved", output)


if __name__ == "__main__":
    main()
