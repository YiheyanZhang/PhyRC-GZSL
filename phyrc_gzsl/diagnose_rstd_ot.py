from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyrc_gzsl.models.backbone import build_backbone
from phyrc_gzsl.utils.config import class_id_lists, class_names_from_config, load_config, resolve_device, set_seed
from phyrc_gzsl.utils.data_loader import (
    flatten_hsi, load_mat_array, load_paviac_from_config, resolve_path, stratified_train_mask,
)
from phyrc_gzsl.utils.text_encoder import SemanticConditionEncoder


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = project_root / path
    if direct.exists() or path.parts[:1] == ("phyrc_gzsl",):
        return direct
    return project_root / "phyrc_gzsl" / path


def sinkhorn_plan(cost: torch.Tensor, epsilon: float = 0.1, iterations: int = 100) -> torch.Tensor:
    kernel = torch.exp(-cost / max(float(epsilon), 1e-6)).clamp_min(1e-30)
    rows, cols = cost.shape
    a = cost.new_full((rows,), 1.0 / rows)
    b = cost.new_full((cols,), 1.0 / cols)
    u, v = torch.ones_like(a), torch.ones_like(b)
    for _ in range(int(iterations)):
        u = a / (kernel @ v).clamp_min(1e-30)
        v = b / (kernel.T @ u).clamp_min(1e-30)
    plan = u[:, None] * kernel * v[None]
    return plan / plan.sum().clamp_min(1e-30)


def weighted_variance(delta: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
    mean = (plan[..., None] * delta).sum((0, 1))
    return (plan * (delta - mean).square().sum(-1)).sum()


def calibrate_target_attributes(
    support_llm: torch.Tensor,
    support_empirical: torch.Tensor,
    target_llm: torch.Tensor,
    ridge: float = 0.1,
) -> torch.Tensor:
    if support_llm.ndim != 2 or support_empirical.ndim != 2:
        raise ValueError("support attributes must be matrices")
    if (
        support_llm.shape != support_empirical.shape
        or target_llm.numel() != support_llm.shape[1]
    ):
        raise ValueError("incompatible attribute shapes")
    predictions = []
    for index in range(support_llm.shape[1]):
        design = torch.stack([support_llm[:, index].float(), torch.ones(len(support_llm))], dim=1)
        penalty = torch.diag(torch.tensor([float(ridge), 0.0]))
        weights = torch.linalg.solve(
            design.T @ design + penalty,
            design.T @ support_empirical[:, index].float(),
        )
        predictions.append(torch.tensor([float(target_llm[index]), 1.0]) @ weights)
    return torch.stack(predictions)


def predict_relational_center(
    target_attributes: torch.Tensor,
    support_attributes: torch.Tensor,
    support_centers: torch.Tensor,
    ridge: float = 0.1,
    tau: float = 0.5,
) -> torch.Tensor:
    if support_attributes.ndim != 2 or support_centers.ndim != 2:
        raise ValueError("support attributes and centers must be matrices")
    if len(support_attributes) != len(support_centers) or len(support_attributes) < 2:
        raise ValueError("at least two aligned support classes are required")
    classes = list(range(len(support_attributes)))
    candidates = predict_relational_hypotheses(
        target_attributes, support_attributes, support_centers, ridge,
    )
    distances = torch.cdist(target_attributes.float().reshape(1, -1), support_attributes.float()).flatten()
    weights = torch.softmax(-distances / max(float(tau), 1e-6), dim=0)
    return (weights[:, None] * candidates).sum(0)


def predict_relational_hypotheses(
    target_attributes: torch.Tensor,
    support_attributes: torch.Tensor,
    support_centers: torch.Tensor,
    ridge: float = 0.1,
) -> torch.Tensor:
    classes = list(range(len(support_attributes)))
    semantics = {index: support_attributes[index].float() for index in classes}
    centers = {index: support_centers[index].float() for index in classes}
    transform = _fit_relation(classes, centers, semantics, float(ridge))
    return support_centers.float() + (target_attributes.float() - support_attributes.float()) @ transform


def spectral_class_attributes(
    spectra: torch.Tensor,
    labels: torch.Tensor,
    classes: list[int],
    wavelengths: torch.Tensor,
) -> torch.Tensor:
    spectra = spectra.float()
    wavelengths = wavelengths.float()
    if spectra.ndim != 2 or wavelengths.numel() != spectra.shape[1] or len(labels) != len(spectra):
        raise ValueError("spectra, labels and wavelengths are not aligned")

    def region(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
        mask = (wavelengths >= low) & (wavelengths <= high)
        return values[:, mask].mean(1)

    rows = []
    for class_id in classes:
        values = spectra[labels == int(class_id)]
        if not len(values):
            raise ValueError(f"class {class_id} has no spectra")
        brightness = values.mean()
        visible_slope = (region(values, 600, 680) - region(values, 430, 500)).mean()
        green_peak = (region(values, 530, 570) - 0.5 * (region(values, 480, 520) + region(values, 600, 640))).mean()
        red_absorption = (0.5 * (region(values, 640, 660) + region(values, 700, 720)) - region(values, 665, 690)).mean()
        red_edge = (region(values, 740, 780) - region(values, 665, 690)).mean()
        nir_level = region(values, 750, 860).mean()
        curvature = torch.diff(values, n=2, dim=1).abs().mean()
        smoothness = 1.0 / (1.0 + curvature / values.std(unbiased=False).clamp_min(1e-6))
        variability = values.std(dim=0, unbiased=False).mean()
        rows.append(torch.stack([
            brightness, visible_slope, green_peak, red_absorption,
            red_edge, nir_level, smoothness, variability,
        ]))
    return torch.stack(rows)


def center_prediction_metrics(
    target_features: torch.Tensor,
    predicted_center: torch.Tensor,
    support_centers: torch.Tensor,
) -> dict[str, float]:
    real_center = target_features.float().mean(0)
    hypotheses = predicted_center.float()
    if hypotheses.ndim == 1:
        hypotheses = hypotheses[None]
    error = 1.0 - F.cosine_similarity(hypotheses[:1], real_center[None]).item()
    bank = F.normalize(torch.cat([support_centers.float(), hypotheses]), dim=1)
    predictions = F.normalize(target_features.float(), dim=1) @ bank.T
    ncm = predictions.argmax(1).ge(len(support_centers)).float().mean().item() * 100.0
    return {"center_cosine_error": float(error), "ncm": float(ncm)}


def proxy_gzsl_metrics(
    target_features: torch.Tensor,
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    predicted_center: torch.Tensor,
    support_centers: torch.Tensor,
) -> dict[str, float]:
    hypotheses = predicted_center.float()
    if hypotheses.ndim == 1:
        hypotheses = hypotheses[None]
    bank = F.normalize(torch.cat([support_centers.float(), hypotheses]), dim=1)
    seen_predictions = (F.normalize(support_features.float(), dim=1) @ bank.T).argmax(1)
    seen_aa = torch.stack([
        seen_predictions[support_labels == index].eq(index).float().mean()
        for index in range(len(support_centers))
    ]).mean().item() * 100.0
    unseen_predictions = (F.normalize(target_features.float(), dim=1) @ bank.T).argmax(1)
    unseen_aa = unseen_predictions.ge(len(support_centers)).float().mean().item() * 100.0
    h = 2.0 * seen_aa * unseen_aa / max(seen_aa + unseen_aa, 1e-12)
    return {"seen_aa": float(seen_aa), "unseen_aa": float(unseen_aa), "h": float(h)}


def rank_relation_attributes(
    support: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat([support.float(), target.float().reshape(1, -1)])
    ranks = (values[:, None] > values[None]).float().sum(1)
    ranks += 0.5 * (values[:, None] == values[None]).float().sum(1) - 0.5
    ranks /= max(len(values) - 1, 1)
    return ranks[:-1], ranks[-1]


def attribute_reliability_weights(
    attributes: torch.Tensor, centers: torch.Tensor,
) -> torch.Tensor:
    center_distances = torch.pdist(centers.float())
    center_distances = (center_distances - center_distances.mean()) / center_distances.std(unbiased=False).clamp_min(1e-6)
    scores = []
    for slot in range(attributes.shape[1]):
        distances = torch.pdist(attributes[:, slot:slot + 1].float())
        distances = (distances - distances.mean()) / distances.std(unbiased=False).clamp_min(1e-6)
        scores.append((distances * center_distances).mean().clamp_min(0.0))
    weights = torch.stack(scores)
    return weights / weights.mean().clamp_min(1e-6) if weights.sum() > 0 else torch.ones_like(weights)


def _standardize_relation_attributes(
    support: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = support.float().mean(0)
    std = support.float().std(0, unbiased=False).clamp_min(1e-4)
    return (support.float() - mean) / std, (target.float() - mean) / std


def relational_loco_error(
    attributes: torch.Tensor,
    centers: torch.Tensor,
    ridge: float,
    tau: float,
) -> float:
    errors = []
    for target in range(len(attributes)):
        keep = torch.arange(len(attributes)) != target
        predicted = predict_relational_center(
            attributes[target], attributes[keep], centers[keep], ridge=ridge, tau=tau,
        )
        errors.append(1.0 - F.cosine_similarity(predicted[None], centers[target][None]).item())
    return float(np.mean(errors))


def _select_relational_parameters(attributes: torch.Tensor, centers: torch.Tensor) -> tuple[float, float]:
    best = None
    for ridge in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        for tau in (0.1, 0.25, 0.5, 1.0, 2.0):
            score = relational_loco_error(attributes, centers, ridge, tau)
            if best is None or score < best[0]:
                best = score, ridge, tau
    return float(best[1]), float(best[2])


def _clip_weighted_center(
    target: torch.Tensor, support: torch.Tensor, centers: torch.Tensor, tau: float,
) -> torch.Tensor:
    similarities = F.normalize(target.float()[None], dim=1) @ F.normalize(support.float(), dim=1).T
    weights = torch.softmax(similarities.flatten() / max(float(tau), 1e-6), dim=0)
    return (weights[:, None] * centers.float()).sum(0)


def loco_center_diagnostic(
    features: torch.Tensor,
    labels: torch.Tensor,
    classes: list[int],
    llm_attributes: torch.Tensor,
    empirical_attributes: torch.Tensor,
    clip_conditions: torch.Tensor,
    clip_tau: float,
) -> dict:
    centers = torch.stack([features[labels == class_id].mean(0) for class_id in classes])
    rows = []
    for target in range(len(classes)):
        keep = torch.arange(len(classes)) != target
        target_features = features[labels == classes[target]]
        support_centers = centers[keep]
        support_class_ids = [classes[index] for index in torch.where(keep)[0].tolist()]
        support_features = torch.cat([features[labels == class_id] for class_id in support_class_ids])
        support_labels = torch.cat([
            torch.full((int((labels == class_id).sum()),), index, dtype=torch.long)
            for index, class_id in enumerate(support_class_ids)
        ])

        def evaluate(predicted: torch.Tensor) -> dict[str, float]:
            metrics = center_prediction_metrics(target_features, predicted, support_centers)
            metrics.update(proxy_gzsl_metrics(
                target_features, support_features, support_labels, predicted, support_centers,
            ))
            return metrics

        clip = _clip_weighted_center(
            clip_conditions[target], clip_conditions[keep], support_centers, clip_tau,
        )

        raw_support, raw_target = _standardize_relation_attributes(
            llm_attributes[keep], llm_attributes[target],
        )
        raw_ridge, raw_tau = _select_relational_parameters(raw_support, support_centers)
        uncalibrated = predict_relational_center(
            raw_target, raw_support, support_centers, raw_ridge, raw_tau,
        )
        raw_hypotheses = predict_relational_hypotheses(
            raw_target, raw_support, support_centers, raw_ridge,
        )

        ranked_support, ranked_target = rank_relation_attributes(
            llm_attributes[keep], llm_attributes[target],
        )
        reliability = attribute_reliability_weights(ranked_support, support_centers).sqrt()
        ranked_support, ranked_target = ranked_support * reliability, ranked_target * reliability
        ranked_ridge, ranked_tau = _select_relational_parameters(ranked_support, support_centers)
        ranked = predict_relational_center(
            ranked_target, ranked_support, support_centers, ranked_ridge, ranked_tau,
        )
        raw_score = relational_loco_error(raw_support, support_centers, raw_ridge, raw_tau)
        ranked_score = relational_loco_error(ranked_support, support_centers, ranked_ridge, ranked_tau)
        robust, robust_method = (
            (ranked, "rank_reliability") if ranked_score < raw_score
            else (uncalibrated, "raw_physical")
        )

        calibrated_target = calibrate_target_attributes(
            llm_attributes[keep], empirical_attributes[keep], llm_attributes[target],
        )
        calibrated_support, calibrated_target = _standardize_relation_attributes(
            empirical_attributes[keep], calibrated_target,
        )
        calibrated_ridge, calibrated_tau = _select_relational_parameters(
            calibrated_support, support_centers,
        )
        calibrated = predict_relational_center(
            calibrated_target, calibrated_support, support_centers,
            calibrated_ridge, calibrated_tau,
        )
        calibrated_hypotheses = predict_relational_hypotheses(
            calibrated_target, calibrated_support, support_centers, calibrated_ridge,
        )

        rows.append({
            "class": int(classes[target]),
            "clip": evaluate(clip),
            "physical_uncalibrated": evaluate(uncalibrated),
            "physical_ranked": evaluate(ranked),
            "physical_robust": evaluate(robust),
            "physical_calibrated": evaluate(calibrated),
            "physical_hypothesis_bank": evaluate(torch.cat([
                uncalibrated[None], calibrated[None], raw_hypotheses, calibrated_hypotheses,
            ])),
            "physical_parameters": {
                "ridge": raw_ridge,
                "tau": raw_tau,
                "ranked_ridge": ranked_ridge,
                "ranked_tau": ranked_tau,
                "selected": robust_method,
                "reliability": reliability.square().tolist(),
            },
        })

    def summarize(key: str) -> dict[str, float]:
        errors = [row[key]["center_cosine_error"] for row in rows]
        ncm = [row[key]["ncm"] for row in rows]
        seen_aa = [row[key]["seen_aa"] for row in rows]
        h = [row[key]["h"] for row in rows]
        return {
            "mean_center_cosine_error": float(np.mean(errors)),
            "mean_ncm": float(np.mean(ncm)),
            "worst_ncm": float(np.min(ncm)),
            "mean_seen_aa": float(np.mean(seen_aa)),
            "mean_h": float(np.mean(h)),
            "worst_h": float(np.min(h)),
        }

    summary = {
        key: summarize(key)
        for key in (
            "clip", "physical_uncalibrated", "physical_ranked", "physical_robust",
            "physical_calibrated", "physical_hypothesis_bank",
        )
    }
    clip_error = summary["clip"]["mean_center_cosine_error"]
    improved_error = summary["physical_robust"]["mean_center_cosine_error"]
    summary["improved_vs_clip_error_reduction_pct"] = float(
        (1.0 - improved_error / max(clip_error, 1e-12)) * 100.0
    )
    return {"summary": summary, "by_class": rows}


def _fit_relation(classes, centers, semantics, ridge):
    x, y = [], []
    for source in classes:
        for target in classes:
            if source != target:
                x.append(semantics[target] - semantics[source])
                y.append(centers[target] - centers[source])
    x, y = torch.stack(x), torch.stack(y)
    return x.T @ torch.linalg.solve(x @ x.T + ridge * torch.eye(len(x)), y)


def _predict_center(target, support, centers, semantics, transform, tau=0.07):
    candidates = torch.stack([centers[source] + (semantics[target] - semantics[source]) @ transform for source in support])
    conditions = torch.stack([semantics[source] for source in support])
    weights = torch.softmax(F.normalize(semantics[target][None], dim=1) @ F.normalize(conditions, dim=1).T / tau, dim=1).flatten()
    return (weights[:, None] * candidates).sum(0)


def _select_ridge(support, centers, semantics):
    best = None
    for ridge in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        errors = []
        for target in support:
            inner = [value for value in support if value != target]
            transform = _fit_relation(inner, centers, semantics, ridge)
            prediction = _predict_center(target, inner, centers, semantics, transform)
            errors.append(1.0 - float(F.cosine_similarity(prediction[None], centers[target][None])))
        score = float(np.mean(errors))
        if best is None or score < best[0]:
            best = score, ridge
    return best[1]


def loco_relational_ncm(features, labels, classes, semantics):
    centers = {c: features[labels == c].mean(0) for c in classes}
    scores = []
    for target in classes:
        support = [c for c in classes if c != target]
        ridge = _select_ridge(support, centers, semantics)
        transform = _fit_relation(support, centers, semantics, ridge)
        predicted = _predict_center(target, support, centers, semantics, transform)
        bank = torch.stack([centers[c] for c in support] + [predicted])
        logits = F.normalize(features[labels == target], dim=1) @ F.normalize(bank, dim=1).T
        scores.append(float((logits.argmax(1) == len(support)).float().mean() * 100.0))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--max-per-class", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--attributes", default="phyrc_gzsl/data/processed/PaviaU_structured_attributes.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    backbone_checkpoint = resolve_project_path(root, args.backbone_checkpoint)
    config["model"].update({
        "backbone": "spectral_morphology",
        "pretrained_backbone": str(backbone_checkpoint),
        "freeze_backbone": True,
    })
    set_seed(int(config["runtime"]["seed"]))
    device = resolve_device(config["runtime"].get("device", "auto"))
    x, gt = load_paviac_from_config(config, root)
    spectra, raw_labels = flatten_hsi(x, gt, ignore_background=True)
    seen, _, _ = class_id_lists(config)
    mask = stratified_train_mask(raw_labels, seen, float(config["data_split"]["train_ratio"]), int(config["data_split"]["seed"]))
    spectra = torch.from_numpy(spectra[mask].astype(np.float32))
    labels = torch.from_numpy(raw_labels[mask].astype(np.int64))

    ds = config["dataset"]
    raw_root = root / "phyrc_gzsl" / ds.get("raw_dir", "data/raw")
    raw_x = load_mat_array(
        resolve_path(raw_root, ds["data_file"], ds.get("fallback_data_file")),
        ds["data_key"], ds.get("fallback_data_key"),
    )
    raw_spectra, raw_spectral_labels = flatten_hsi(raw_x, gt, ignore_background=True)
    if not np.array_equal(raw_spectral_labels, raw_labels):
        raise RuntimeError("raw and normalized labels are not aligned")
    raw_spectra = torch.from_numpy(raw_spectra[mask].astype(np.float32))
    model = build_backbone(x.shape[-1], config).to(device).eval()
    with torch.no_grad():
        features = torch.cat([model(batch.to(device)).cpu() for batch in spectra.split(512)])
        shapes = torch.cat([model.morphology_features(batch.to(device)).cpu() for batch in spectra.split(512)])
    feature_costs, joint_costs, random_feature_costs, random_joint_costs = [], [], [], []
    feature_vars, joint_vars, random_vars = [], [], []
    for i, source in enumerate(seen):
        for target in seen[i + 1:]:
            source_indices = torch.where(labels == source)[0]
            target_indices = torch.where(labels == target)[0]
            a = source_indices[torch.randperm(len(source_indices))[:args.max_per_class]]
            b = target_indices[torch.randperm(len(target_indices))[:args.max_per_class]]
            fd = torch.cdist(features[a], features[b]).square()
            sd = torch.cdist(shapes[a], shapes[b]).square()
            feature_cost = fd / fd.median().clamp_min(1e-6)
            joint_cost = feature_cost + sd / sd.median().clamp_min(1e-6)
            feature_plan = sinkhorn_plan(feature_cost, args.epsilon)
            joint_plan = sinkhorn_plan(joint_cost, args.epsilon)
            random = torch.full_like(joint_plan, 1.0 / joint_plan.numel())
            delta = features[b][None] - features[a][:, None]
            feature_costs.append(float((feature_plan * feature_cost).sum()))
            joint_costs.append(float((joint_plan * joint_cost).sum()))
            random_feature_costs.append(float((random * feature_cost).sum()))
            random_joint_costs.append(float((random * joint_cost).sum()))
            feature_vars.append(float(weighted_variance(delta, feature_plan)))
            joint_vars.append(float(weighted_variance(delta, joint_plan)))
            random_vars.append(float(weighted_variance(delta, random)))
    encoder = SemanticConditionEncoder(config, root)
    semantic_map = encoder.encode_classes(class_names_from_config(config))
    clip_conditions = torch.stack([semantic_map[class_id].float() for class_id in seen])
    attribute_path = Path(args.attributes)
    if not attribute_path.is_absolute():
        attribute_path = root / attribute_path
    attribute_payload = json.loads(attribute_path.read_text(encoding="utf-8"))
    attribute_keys = (
        "brightness", "visible_slope", "green_peak", "red_absorption",
        "red_edge", "nir_level", "smoothness", "variability",
    )
    llm_attributes = torch.tensor([
        [attribute_payload[str(class_id)][key] for key in attribute_keys]
        for class_id in seen
    ], dtype=torch.float32)
    low, high = config.get("text", {}).get("wavelength_nm", [430, 860])
    empirical_attributes = spectral_class_attributes(
        raw_spectra, labels, seen, torch.linspace(float(low), float(high), raw_spectra.shape[1]),
    )
    center_diagnostic = loco_center_diagnostic(
        features, labels, seen, llm_attributes, empirical_attributes, clip_conditions,
        clip_tau=float(config.get("diffusion", {}).get("text_proto_tau", 0.07)),
    )
    feature_variance_reduction = float((1 - np.mean(feature_vars) / np.mean(random_vars)) * 100)
    joint_variance_reduction = float((1 - np.mean(joint_vars) / np.mean(random_vars)) * 100)
    result = {
        "split": Path(args.config).stem,
        "seen_classes": seen,
        "ot": {
            "feature_only": {
                "ot_cost": float(np.mean(feature_costs)),
                "random_cost": float(np.mean(random_feature_costs)),
                "cost_reduction_pct": float((1 - np.mean(feature_costs) / np.mean(random_feature_costs)) * 100),
                "ot_relation_variance": float(np.mean(feature_vars)),
                "random_relation_variance": float(np.mean(random_vars)),
                "variance_reduction_pct": feature_variance_reduction,
            },
            "feature_morphology": {
                "ot_cost": float(np.mean(joint_costs)),
                "random_cost": float(np.mean(random_joint_costs)),
                "cost_reduction_pct": float((1 - np.mean(joint_costs) / np.mean(random_joint_costs)) * 100),
                "ot_relation_variance": float(np.mean(joint_vars)),
                "random_relation_variance": float(np.mean(random_vars)),
                "variance_reduction_pct": joint_variance_reduction,
            },
            "best_variance_reduction_pct": max(feature_variance_reduction, joint_variance_reduction),
            "passes_20pct_gate": bool(max(feature_variance_reduction, joint_variance_reduction) >= 20.0),
        },
        "center_prediction": center_diagnostic,
        "attribute_diagnostic": {
            str(class_id): {
                "llm": llm_attributes[index].tolist(),
                "empirical": empirical_attributes[index].tolist(),
            }
            for index, class_id in enumerate(seen)
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved {output}")


if __name__ == "__main__":
    main()
