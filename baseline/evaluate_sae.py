from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.eszsl import calibrate_seen_scores
from baseline.evaluate_eszsl import (
    _ATTRIBUTE_KEYS,
    _BIASES,
    _balanced_subset,
    choose_eszsl_candidate,
)
from baseline.sae import fit_sae, sae_scores
from diagnose_rstd_ot import resolve_project_path
from models.backbone import build_backbone
from utils.config import (
    class_id_lists,
    load_config,
    resolve_device,
    set_seed,
    set_unseen_classes,
)
from utils.data_loader import flatten_hsi, load_paviac_from_config, stratified_train_mask
from utils.metrics import gzsl_metrics


_REGULARIZATIONS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _semantic_targets(
    labels: torch.Tensor,
    classes: list[int],
    attributes: torch.Tensor,
) -> torch.Tensor:
    lookup = {class_id: index for index, class_id in enumerate(classes)}
    return attributes[torch.tensor([lookup[int(label)] for label in labels])]


def select_sae_parameters(
    features: torch.Tensor,
    labels: torch.Tensor,
    classes: list[int],
    attributes: torch.Tensor,
    regularizations: tuple[float, ...] = _REGULARIZATIONS,
    biases: tuple[float, ...] = _BIASES,
) -> dict:
    """Select SAE regularization and Seen bias using Seen-only LOCO."""
    candidates = []
    for regularization in regularizations:
        episodes = []
        for target_index, target_class in enumerate(classes):
            support_classes = [value for value in classes if value != target_class]
            support_indices = [index for index in range(len(classes)) if index != target_index]
            support_mask = labels != target_class
            weight = fit_sae(
                features[support_mask],
                _semantic_targets(
                    labels[support_mask], support_classes, attributes[support_indices],
                ),
                regularization,
            )
            episode_features, episode_labels = _balanced_subset(features, labels, classes)
            episode_classes = support_classes + [target_class]
            episode_attributes = torch.cat([
                attributes[support_indices], attributes[target_index : target_index + 1],
            ])
            targets = torch.tensor([
                {value: index for index, value in enumerate(episode_classes)}[int(label)]
                for label in episode_labels
            ])
            episodes.append((
                sae_scores(episode_features, weight, episode_attributes),
                targets,
                len(support_classes),
            ))
        for bias in biases:
            results = [
                gzsl_metrics(
                    calibrate_seen_scores(scores, list(range(seen_count)), bias).argmax(1).numpy(),
                    targets.numpy(), list(range(seen_count)), [seen_count],
                )
                for scores, targets, seen_count in episodes
            ]
            candidates.append({
                "regularization": regularization,
                "bias": bias,
                "seen_zero": sum(
                    sum(result["per_class"][index] == 0.0 for index in range(len(result["per_class"]) - 1))
                    for result in results
                ),
                **{
                    key: float(np.mean([result[key] for result in results]))
                    for key in ("OA", "AA", "Seen_AA", "Unseen_AA", "H")
                },
            })
    return choose_eszsl_candidate(candidates)


def evaluate_split(
    config_path: Path,
    backbone_checkpoint: Path,
    attribute_path: Path,
    root: Path,
    unseen_class: int | list[int],
    seed: int,
) -> dict:
    if not backbone_checkpoint.is_file():
        raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_checkpoint}")
    config = load_config(config_path)
    set_unseen_classes(config, [unseen_class] if isinstance(unseen_class, int) else unseen_class)
    config.setdefault("runtime", {})["seed"] = seed
    config.setdefault("data_split", {})["seed"] = seed
    config["model"].update({
        "backbone": "spectral_morphology",
        "pretrained_backbone": str(backbone_checkpoint),
        "freeze_backbone": True,
    })
    set_seed(seed)
    seen, unseen, all_classes = class_id_lists(config)
    x, gt = load_paviac_from_config(config, root)
    spectra, labels = flatten_hsi(x, gt, ignore_background=True)
    train_mask = stratified_train_mask(
        labels, seen, float(config["data_split"]["train_ratio"]), seed,
    )
    device = resolve_device(config["runtime"].get("device", "auto"))
    model = build_backbone(x.shape[-1], config).to(device).eval()
    with torch.no_grad():
        features = torch.cat([
            model(batch.to(device)).cpu()
            for batch in torch.from_numpy(spectra.astype(np.float32)).split(512)
        ])
    features = F.normalize(features.float(), dim=1)
    train_features = features[torch.from_numpy(train_mask)]
    train_labels = torch.from_numpy(labels[train_mask].astype(np.int64))
    payload = json.loads(attribute_path.read_text(encoding="utf-8"))
    attributes = F.normalize(torch.tensor([
        [payload[str(class_id)][key] for key in _ATTRIBUTE_KEYS]
        for class_id in all_classes
    ], dtype=torch.float32), dim=1)
    selected = select_sae_parameters(
        train_features, train_labels, seen, attributes[: len(seen)],
    )
    weight = fit_sae(
        train_features,
        _semantic_targets(train_labels, seen, attributes[: len(seen)]),
        selected["regularization"],
    )
    label_to_index = {class_id: index for index, class_id in enumerate(all_classes)}
    test_mask = (~train_mask) & np.isin(labels, all_classes)
    test_targets = np.array([label_to_index[int(value)] for value in labels[test_mask]])
    scores = calibrate_seen_scores(
        sae_scores(features[torch.from_numpy(test_mask)], weight, attributes),
        list(range(len(seen))), selected["bias"],
    )
    metrics = gzsl_metrics(
        scores.argmax(1).numpy(), test_targets,
        list(range(len(seen))), list(range(len(seen), len(all_classes))),
    )
    return {"unseen": unseen[0] if len(unseen) == 1 else unseen, "selection": selected, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--unseen-classes", nargs="+", type=int, required=True)
    parser.add_argument("--backbone-pattern", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [
        evaluate_split(
            resolve_project_path(root, args.config),
            resolve_project_path(root, args.backbone_pattern.format(unseen=unseen)),
            resolve_project_path(root, args.attributes),
            root, unseen, args.seed,
        )
        for unseen in args.unseen_classes
    ]
    keys = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
    summary = {key: float(np.mean([row["metrics"][key] for row in rows])) for key in keys}
    output = resolve_project_path(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
