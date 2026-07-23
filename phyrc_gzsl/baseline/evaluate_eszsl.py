from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phyrc_gzsl.baseline.eszsl import calibrate_seen_scores, eszsl_scores, fit_eszsl
from phyrc_gzsl.diagnose_rstd_ot import resolve_project_path
from phyrc_gzsl.models.backbone import build_backbone
from phyrc_gzsl.utils.config import (
    class_id_lists,
    load_config,
    resolve_device,
    set_seed,
    set_unseen_classes,
)
from phyrc_gzsl.utils.data_loader import flatten_hsi, load_paviac_from_config, stratified_train_mask
from phyrc_gzsl.utils.metrics import gzsl_metrics


_ATTRIBUTE_KEYS = (
    "brightness", "visible_slope", "green_peak", "red_absorption",
    "red_edge", "nir_level", "smoothness", "variability",
)
_REGULARIZATIONS = (0.01, 1.0, 100.0)
_BIASES = tuple(value / 4 for value in range(9))


def choose_eszsl_candidate(candidates: list[dict]) -> dict:
    """Use Seen-only mean GZSL H, as in standard calibrated stacking validation."""
    return max(candidates, key=lambda value: (
        value["H"], value["Unseen_AA"], value["OA"], -value["seen_zero"],
    ))


def _targets(labels: torch.Tensor, classes: list[int]) -> torch.Tensor:
    indices = torch.tensor([{value: index for index, value in enumerate(classes)}[int(label)] for label in labels])
    return F.one_hot(indices, len(classes)).float()


def _balanced_subset(
    features: torch.Tensor,
    labels: torch.Tensor,
    classes: list[int],
    limit: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ponytail: fixed cap keeps LOCO cheap; raise it if selection variance is material.
    indices = torch.cat([torch.where(labels == class_id)[0][:limit] for class_id in classes])
    return features[indices], labels[indices]


def select_eszsl_parameters(
    features: torch.Tensor,
    labels: torch.Tensor,
    classes: list[int],
    attributes: torch.Tensor,
    regularizations: tuple[float, ...] = _REGULARIZATIONS,
    biases: tuple[float, ...] = _BIASES,
) -> dict:
    """Select ESZSL regularization and calibrated stacking using Seen-only LOCO."""
    candidates = []
    for feature_regularization in regularizations:
        for semantic_regularization in regularizations:
            episodes = []
            for target_index, target_class in enumerate(classes):
                support_classes = [value for value in classes if value != target_class]
                support_indices = [index for index in range(len(classes)) if index != target_index]
                support_mask = labels != target_class
                weight = fit_eszsl(
                    features[support_mask],
                    _targets(labels[support_mask], support_classes),
                    attributes[support_indices],
                    feature_regularization,
                    semantic_regularization,
                )
                episode_features, episode_labels = _balanced_subset(features, labels, classes)
                episode_classes = support_classes + [target_class]
                episode_attributes = torch.cat([
                    attributes[support_indices], attributes[target_index : target_index + 1],
                ])
                scores = eszsl_scores(episode_features, weight, episode_attributes)
                scores /= scores.std().clamp_min(1e-6)
                targets = torch.tensor([
                    {value: index for index, value in enumerate(episode_classes)}[int(label)]
                    for label in episode_labels
                ])
                episodes.append((scores, targets, len(support_classes)))
            for bias in biases:
                results = []
                for scores, targets, seen_count in episodes:
                    predictions = calibrate_seen_scores(
                        scores, list(range(seen_count)), bias,
                    ).argmax(1)
                    results.append(gzsl_metrics(
                        predictions.numpy(), targets.numpy(),
                        list(range(seen_count)), [seen_count],
                    ))
                candidates.append({
                    "feature_regularization": feature_regularization,
                    "semantic_regularization": semantic_regularization,
                    "bias": bias,
                    "seen_zero": sum(
                        sum(result["per_class"][index] == 0.0 for index in range(len(result["per_class"]) - 1))
                        for result in results
                    ),
                    "worst_h": min(result["H"] for result in results),
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
    selected = select_eszsl_parameters(
        train_features, train_labels, seen, attributes[: len(seen)],
    )
    weight = fit_eszsl(
        train_features, _targets(train_labels, seen), attributes[: len(seen)],
        selected["feature_regularization"], selected["semantic_regularization"],
    )
    train_scores = eszsl_scores(train_features, weight, attributes)
    score_scale = train_scores.std().clamp_min(1e-6)
    label_to_index = {class_id: index for index, class_id in enumerate(all_classes)}
    test_mask = (~train_mask) & np.isin(labels, all_classes)
    test_features = features[torch.from_numpy(test_mask)]
    test_targets = np.array([label_to_index[int(value)] for value in labels[test_mask]])
    scores = calibrate_seen_scores(
        eszsl_scores(test_features, weight, attributes) / score_scale,
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
    root = Path(__file__).resolve().parents[2]
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
