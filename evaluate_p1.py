from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_rstd_ot import (
    _select_relational_parameters,
    _standardize_relation_attributes,
    attribute_reliability_weights,
    predict_relational_center,
    predict_relational_hypotheses,
    rank_relation_attributes,
    relational_loco_error,
    resolve_project_path,
)
from models.backbone import build_backbone
from utils.config import (
    class_id_lists,
    class_names_from_config,
    load_config,
    resolve_device,
    set_seed,
    set_single_unseen_class,
)
from utils.data_loader import flatten_hsi, load_paviac_from_config, stratified_train_mask
from utils.metrics import gzsl_metrics


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate_prototype_bank(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    seen_indices: list[int],
    unseen_indices: list[int],
    unseen_bias: float = 0.0,
) -> dict:
    scores = F.normalize(features.float(), dim=1) @ F.normalize(prototypes.float(), dim=1).T
    scores[:, unseen_indices] += float(unseen_bias)
    return gzsl_metrics(
        scores.argmax(1).numpy(), labels.numpy(), seen_indices, unseen_indices,
    )


def build_router_features(
    features: torch.Tensor,
    scores: torch.Tensor,
    seen_indices: list[int],
    unseen_indices: list[int],
    seen_centers: torch.Tensor,
    seen_variances: torch.Tensor,
) -> torch.Tensor:
    seen_max = scores[:, seen_indices].max(1).values
    unseen_max = scores[:, unseen_indices].max(1).values
    density = (
        (features[:, None] - seen_centers[None]).square()
        / seen_variances.clamp_min(1e-4)[None]
    ).mean(2).min(1).values
    return torch.stack([seen_max, unseen_max, unseen_max - seen_max, density.log1p()], dim=1)


def transfer_unseen_variance(
    unseen_center: torch.Tensor,
    seen_centers: torch.Tensor,
    seen_variances: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    similarity = (
        F.normalize(unseen_center[None].float(), dim=1)
        @ F.normalize(seen_centers.float(), dim=1).T
    )
    weights = (similarity[0] / max(float(temperature), 1e-6)).softmax(0)
    return weights @ seen_variances.float()


def build_density_router_features(
    features: torch.Tensor,
    scores: torch.Tensor,
    seen_indices: list[int],
    unseen_indices: list[int],
    seen_centers: torch.Tensor,
    seen_variances: torch.Tensor,
    unseen_center: torch.Tensor,
    unseen_variance: torch.Tensor,
) -> torch.Tensor:
    base = build_router_features(
        features, scores, seen_indices, unseen_indices, seen_centers, seen_variances,
    )
    unseen_density = (
        (features - unseen_center[None]).square()
        / unseen_variance.clamp_min(1e-4)[None]
    ).mean(1).log1p()
    return torch.cat([base, unseen_density[:, None]], dim=1)


def fit_seen_unseen_router(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    ridge: float = 1e-2,
) -> dict:
    inputs, labels = inputs.float(), labels.float()
    mean, scale = inputs.mean(0), inputs.std(0).clamp_min(1e-6)
    normalized = (inputs - mean) / scale
    weight = torch.where(
        labels > 0.5,
        0.5 / (labels > 0.5).sum().clamp_min(1),
        0.5 / (labels <= 0.5).sum().clamp_min(1),
    )
    parameters = torch.zeros(normalized.shape[1] + 1, requires_grad=True)
    optimizer = torch.optim.LBFGS([parameters], max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = normalized @ parameters[:-1] + parameters[-1]
        loss = (
            F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weight
        ).sum() + float(ridge) * parameters[:-1].square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "mean": mean.detach(),
        "scale": scale.detach(),
        "weight": parameters[:-1].detach(),
        "bias": parameters[-1].detach(),
    }


def router_probability(inputs: torch.Tensor, router: dict) -> torch.Tensor:
    normalized = (inputs.float() - router["mean"]) / router["scale"]
    return torch.sigmoid(normalized @ router["weight"] + router["bias"])


def _class_nonconformity(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("Seen conformal scores require at least two classes")
    top_values, top_indices = scores.float().topk(2, dim=1)
    best_other = top_values[:, :1].expand_as(scores).clone()
    best_other.scatter_(1, top_indices[:, :1], top_values[:, 1:2])
    return best_other - scores.float()


def fit_seen_conformal(scores: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
    nonconformity = _class_nonconformity(scores)
    return [
        nonconformity[labels == class_index, class_index].sort().values
        for class_index in range(scores.shape[1])
    ]


def seen_conformal_probability(
    scores: torch.Tensor,
    calibration: list[torch.Tensor],
) -> torch.Tensor:
    nonconformity = _class_nonconformity(scores)
    probabilities = []
    for class_index, reference in enumerate(calibration):
        if reference.numel() == 0:
            raise ValueError(f"Seen class {class_index} has no calibration samples")
        rank = torch.searchsorted(
            reference, nonconformity[:, class_index].contiguous(), right=False,
        )
        probabilities.append(
            (1 + reference.numel() - rank).float() / (reference.numel() + 1)
        )
    return torch.stack(probabilities, dim=1).max(1).values


def apply_conformal_shield(
    unseen_probability: torch.Tensor,
    seen_probability: torch.Tensor,
    power: float = 1.0,
) -> torch.Tensor:
    return unseen_probability * (1.0 - seen_probability).clamp(0.0, 1.0).pow(float(power))


def apply_router_csd(
    scores: torch.Tensor,
    unseen_indices: list[int],
    bias: float,
    unseen_probability: torch.Tensor,
) -> torch.Tensor:
    adjusted = scores.clone()
    adjusted[:, unseen_indices] += float(bias) * unseen_probability[:, None]
    return adjusted


def predict_semantic_center(
    support_attributes: torch.Tensor,
    target_attributes: torch.Tensor,
    support_centers: torch.Tensor,
    ridge: float = 1e-3,
) -> torch.Tensor:
    eye = torch.eye(support_attributes.shape[1], dtype=support_attributes.dtype)
    transform = torch.linalg.solve(
        support_attributes.T @ support_attributes + float(ridge) * eye,
        support_attributes.T @ support_centers,
    )
    return target_attributes @ transform


def predict_learned_relational_center(
    target_attributes: torch.Tensor,
    support_attributes: torch.Tensor,
    support_centers: torch.Tensor,
    ridge,
    tau,
) -> torch.Tensor:
    differences = support_attributes[:, None] - support_attributes[None]
    center_differences = support_centers[:, None] - support_centers[None]
    keep = ~torch.eye(len(support_attributes), dtype=torch.bool)
    x, y = differences[keep].float(), center_differences[keep].float()
    ridge = torch.as_tensor(ridge, dtype=x.dtype)
    transform = torch.linalg.solve(
        x.T @ x + ridge * torch.eye(x.shape[1]), x.T @ y,
    )
    candidates = support_centers.float() + (
        target_attributes.float() - support_attributes.float()
    ) @ transform
    distances = torch.cdist(
        target_attributes.float()[None], support_attributes.float(),
    ).flatten()
    tau = torch.as_tensor(tau, dtype=x.dtype).clamp_min(1e-6)
    return (torch.softmax(-distances / tau, dim=0)[:, None] * candidates).sum(0)


def learn_relational_parameters(
    attributes: torch.Tensor,
    centers: torch.Tensor,
    *,
    steps: int = 100,
    learning_rate: float = 0.05,
) -> dict:
    if len(attributes) < 3:
        raise ValueError("relational parameter learning requires at least three classes")
    raw_ridge = torch.tensor(0.0, requires_grad=True)
    raw_tau = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.Adam((raw_ridge, raw_tau), lr=learning_rate)
    log_ridge_min, log_ridge_max = np.log(1e-3), np.log(10.0)
    log_tau_min, log_tau_max = np.log(0.05), np.log(2.0)
    for _ in range(steps):
        ridge = torch.exp(torch.tensor(log_ridge_min) + (log_ridge_max - log_ridge_min) * torch.sigmoid(raw_ridge))
        tau = torch.exp(torch.tensor(log_tau_min) + (log_tau_max - log_tau_min) * torch.sigmoid(raw_tau))
        errors = []
        for target in range(len(attributes)):
            keep = torch.arange(len(attributes)) != target
            predicted = predict_learned_relational_center(
                attributes[target], attributes[keep], centers[keep], ridge, tau,
            )
            errors.append(1 - F.cosine_similarity(predicted[None], centers[target][None]) + 0.1 * F.mse_loss(F.normalize(predicted, dim=0), F.normalize(centers[target], dim=0)))
        loss = torch.stack(errors).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {
        "ridge": float(torch.exp(torch.tensor(log_ridge_min) + (log_ridge_max - log_ridge_min) * torch.sigmoid(raw_ridge.detach()))),
        "tau": float(torch.exp(torch.tensor(log_tau_min) + (log_tau_max - log_tau_min) * torch.sigmoid(raw_tau.detach()))),
    }


def predict_learned_relation(
    support_attributes: torch.Tensor,
    target_attributes: torch.Tensor,
    support_centers: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    ranked_support, ranked_target = rank_relation_attributes(
        support_attributes, target_attributes,
    )
    reliability = attribute_reliability_weights(ranked_support, support_centers).sqrt()
    ranked_support, ranked_target = ranked_support * reliability, ranked_target * reliability
    parameters = learn_relational_parameters(ranked_support, support_centers)
    center = predict_learned_relational_center(
        ranked_target, ranked_support, support_centers,
        parameters["ridge"], parameters["tau"],
    )
    return center, parameters


def predict_prototype_set(
    support_attributes: torch.Tensor,
    target_attributes: torch.Tensor,
    support_centers: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    from phyrc.attention import predict_attention_center

    variants = predict_relation_variants(
        support_attributes, target_attributes, support_centers,
    )
    attention, parameters = predict_attention_center(
        support_attributes, target_attributes, support_centers,
    )
    return torch.stack([variants["semantic"], variants["ranked"], attention]), parameters


def predict_relation_variants(
    support_attributes: torch.Tensor,
    target_attributes: torch.Tensor,
    support_centers: torch.Tensor,
) -> dict:
    raw_support, raw_target = _standardize_relation_attributes(
        support_attributes, target_attributes,
    )
    raw_ridge, raw_tau = _select_relational_parameters(raw_support, support_centers)
    raw_center = predict_relational_center(
        raw_target, raw_support, support_centers, raw_ridge, raw_tau,
    )
    raw_score = relational_loco_error(raw_support, support_centers, raw_ridge, raw_tau)

    ranked_support, ranked_target = rank_relation_attributes(
        support_attributes, target_attributes,
    )
    reliability = attribute_reliability_weights(ranked_support, support_centers).sqrt()
    ranked_support, ranked_target = ranked_support * reliability, ranked_target * reliability
    ranked_ridge, ranked_tau = _select_relational_parameters(ranked_support, support_centers)
    ranked_center = predict_relational_center(
        ranked_target, ranked_support, support_centers, ranked_ridge, ranked_tau,
    )
    ranked_score = relational_loco_error(
        ranked_support, support_centers, ranked_ridge, ranked_tau,
    )
    robust_center, selected = (
        (ranked_center, "rank_reliability")
        if ranked_score < raw_score
        else (raw_center, "raw_physical")
    )
    return {
        "semantic": predict_semantic_center(
            support_attributes, target_attributes, support_centers,
        ),
        "raw": raw_center,
        "ranked": ranked_center,
        "robust": robust_center,
        "selected": selected,
        "loco_error": {"raw": raw_score, "ranked": ranked_score},
    }


def choose_best_bias(choices: tuple[float, ...], h_scores: torch.Tensor) -> float:
    return float(choices[int(h_scores.mean(1).argmax())])


def build_seen_mode_distributions(
    features: torch.Tensor, labels: torch.Tensor, classes: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    from phyrc.attention import class_mode_distribution

    distributions = [class_mode_distribution(features[labels == cls]) for cls in classes]
    return torch.stack([row[0] for row in distributions]), torch.stack([row[1] for row in distributions])


def mode_distribution_temperature(centers: torch.Tensor, modes: torch.Tensor) -> float:
    similarities = (
        F.normalize(modes.float(), dim=2)
        * F.normalize(centers.float(), dim=1)[:, None]
    ).sum(2)
    return float((1 - similarities).median().clamp(0.03, 0.20))


def predict_physical_distribution(
    support_attributes: torch.Tensor,
    target_attribute: torch.Tensor,
    support_centers: torch.Tensor,
    support_modes: torch.Tensor,
    support_mode_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from phyrc.attention import transport_mode_distribution

    ranked_support, ranked_target = rank_relation_attributes(
        support_attributes, target_attribute,
    )
    reliability = attribute_reliability_weights(ranked_support, support_centers).sqrt()
    ranked_support, ranked_target = ranked_support * reliability, ranked_target * reliability
    ridge, tau = _select_relational_parameters(ranked_support, support_centers)
    anchors = predict_relational_hypotheses(
        ranked_target, ranked_support, support_centers, ridge,
    )
    distances = torch.cdist(ranked_target[None], ranked_support).flatten()
    anchor_weights = torch.softmax(-distances / max(float(tau), 1e-6), dim=0)
    modes, weights = transport_mode_distribution(
        anchors, support_centers, support_modes, support_mode_weights, anchor_weights,
    )
    centre = F.normalize((weights[:, None] * modes).sum(0), dim=0)
    return modes, weights, centre


def physical_distribution_score_matrix(
    features: torch.Tensor,
    seen_modes: torch.Tensor,
    seen_weights: torch.Tensor,
    unseen_modes: torch.Tensor,
    unseen_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    from phyrc.attention import prototype_distribution_scores

    columns = [
        prototype_distribution_scores(features, modes, weights, temperature)
        for modes, weights in zip(seen_modes, seen_weights)
    ]
    columns.append(prototype_distribution_scores(
        features, unseen_modes, unseen_weights, temperature,
    ))
    return torch.stack(columns, dim=1)


def hybrid_distribution_score_matrix(
    features: torch.Tensor,
    seen_modes: torch.Tensor,
    seen_weights: torch.Tensor,
    unseen_candidates: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    from phyrc.attention import prototype_distribution_scores, prototype_set_scores

    columns = [
        prototype_distribution_scores(features, modes, weights, temperature)
        for modes, weights in zip(seen_modes, seen_weights)
    ]
    columns.append(prototype_set_scores(features, unseen_candidates))
    return torch.stack(columns, dim=1)


def build_loco_episodes(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    classes: list[int],
    attributes: torch.Tensor,
    centers: torch.Tensor,
    method: str = "ranked",
) -> list[dict]:
    all_modes = all_mode_weights = None
    if method in {"physical_distribution", "hybrid_distribution"}:
        all_modes, all_mode_weights = build_seen_mode_distributions(
            train_features, train_labels, classes,
        )
    episodes = []
    for target in range(len(classes)):
        keep = torch.arange(len(classes)) != target
        keep_indices = torch.where(keep)[0].tolist()
        if method in {"attention", "prototype_set"}:
            from phyrc.attention import predict_attention_center
        candidates = None
        distribution = None
        if method == "physical_distribution":
            distribution = predict_physical_distribution(
                attributes[keep], attributes[target], centers[keep],
                all_modes[keep], all_mode_weights[keep],
            )
            predicted = distribution[2]
        elif method in {"prototype_set", "hybrid_distribution"}:
            candidates, _ = predict_prototype_set(
                attributes[keep], attributes[target], centers[keep],
            )
            predicted = candidates.mean(0)
        else:
            predicted = (
            predict_learned_relation(
                attributes[keep], attributes[target], centers[keep],
            )[0]
            if method == "learned" else
            predict_attention_center(
                attributes[keep], attributes[target], centers[keep],
            )[0]
            if method == "attention" else
            predict_relation_variants(
                attributes[keep], attributes[target], centers[keep],
            )[method]
            )
        episode_features = torch.cat([
            train_features[train_labels == classes[index]] for index in keep_indices
        ] + [train_features[train_labels == classes[target]]])
        episode_labels = torch.cat([
            torch.full(
                (int((train_labels == classes[index]).sum()),), local, dtype=torch.long,
            )
            for local, index in enumerate(keep_indices)
        ] + [
            torch.full(
                (int((train_labels == classes[target]).sum()),),
                len(classes) - 1,
                dtype=torch.long,
            )
        ])
        support_centers = centers[keep]
        variances = torch.stack([
            train_features[train_labels == classes[index]].var(0, unbiased=False)
            for index in keep_indices
        ])
        bank = torch.cat([support_centers, predicted[None]])
        scores = (
            physical_distribution_score_matrix(
                episode_features, all_modes[keep], all_mode_weights[keep],
                distribution[0], distribution[1],
                mode_distribution_temperature(support_centers, all_modes[keep]),
            )
            if distribution is not None else
            hybrid_distribution_score_matrix(
                episode_features, all_modes[keep], all_mode_weights[keep], candidates,
                mode_distribution_temperature(support_centers, all_modes[keep]),
            )
            if method == "hybrid_distribution" else
            F.normalize(episode_features.float(), dim=1)
            @ F.normalize(bank.float(), dim=1).T
        )
        if candidates is not None and method == "prototype_set":
            from phyrc.attention import prototype_set_scores
            scores[:, -1] = prototype_set_scores(episode_features, candidates)
        seen_indices = list(range(len(classes) - 1))
        unseen_indices = [len(classes) - 1]
        episodes.append({
            "target_index": target,
            "features": episode_features,
            "labels": episode_labels,
            "bank": bank,
            "bank_attributes": torch.cat([attributes[keep], attributes[target][None]]),
            "variances": variances,
            "scores": scores,
            "router_inputs": build_router_features(
                episode_features, scores, seen_indices, unseen_indices,
                support_centers, variances,
            ),
            "density_inputs": build_density_router_features(
                episode_features, scores, seen_indices, unseen_indices,
                support_centers, variances, predicted,
                transfer_unseen_variance(predicted, support_centers, variances),
            ),
            "router_labels": (episode_labels == unseen_indices[0]).float(),
        })
    return episodes


def select_csd_bias(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    classes: list[int],
    attributes: torch.Tensor,
    centers: torch.Tensor,
    method: str,
) -> tuple[float, float, dict, dict]:
    choices = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    h_scores = torch.zeros(len(choices), len(classes))
    episodes = build_loco_episodes(
        train_features, train_labels, classes, attributes, centers, method,
    )
    for episode in episodes:
        for index, bias in enumerate(choices):
            result = evaluate_prototype_bank(
                episode["features"], episode["labels"], episode["bank"],
                list(range(len(classes) - 1)), [len(classes) - 1], unseen_bias=bias,
            )
            h_scores[index, episode["target_index"]] = result["H"]
    bias = choose_best_bias(choices, h_scores)
    router_labels = torch.cat([episode["router_labels"] for episode in episodes])
    router = fit_seen_unseen_router(
        torch.cat([episode["router_inputs"] for episode in episodes]), router_labels,
    )
    density_router = fit_seen_unseen_router(
        torch.cat([episode["density_inputs"] for episode in episodes]), router_labels,
    )
    return bias, float(h_scores[choices.index(bias)].mean()), router, density_router


def evaluate_split(
    config_path: Path,
    backbone_checkpoint: Path,
    attribute_path: Path,
    baseline_checkpoint: Path | None,
    root: Path,
    unseen_override: int | None = None,
) -> dict:
    config = load_config(config_path)
    if unseen_override is not None:
        set_single_unseen_class(config, unseen_override)
    seen, unseen, all_classes = class_id_lists(config)
    if len(unseen) != 1:
        raise ValueError("P1 evaluation requires one Unseen class")
    config["model"].update({
        "backbone": "spectral_morphology",
        "pretrained_backbone": str(backbone_checkpoint),
        "freeze_backbone": True,
    })
    if not backbone_checkpoint.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_checkpoint}")
    set_seed(int(config["runtime"]["seed"]))
    device = resolve_device(config["runtime"].get("device", "auto"))
    x, gt = load_paviac_from_config(config, root)
    spectra, labels = flatten_hsi(x, gt, ignore_background=True)
    train_mask = stratified_train_mask(
        labels, seen, float(config["data_split"]["train_ratio"]),
        int(config["data_split"]["seed"]),
    )
    model = build_backbone(x.shape[-1], config).to(device).eval()
    tensor_spectra = torch.from_numpy(spectra.astype(np.float32))
    with torch.no_grad():
        features = torch.cat([
            model(batch.to(device)).cpu() for batch in tensor_spectra.split(512)
        ])

    train_features = features[torch.from_numpy(train_mask)]
    train_labels = torch.from_numpy(labels[train_mask].astype(np.int64))
    seen_centers = torch.stack([
        train_features[train_labels == class_id].mean(0) for class_id in seen
    ])
    seen_variances = torch.stack([
        train_features[train_labels == class_id].var(0, unbiased=False)
        for class_id in seen
    ])
    attributes_payload = json.loads(attribute_path.read_text(encoding="utf-8"))
    attribute_keys = (
        "brightness", "visible_slope", "green_peak", "red_absorption",
        "red_edge", "nir_level", "smoothness", "variability",
    )
    support_attributes = torch.tensor([
        [attributes_payload[str(class_id)][key] for key in attribute_keys]
        for class_id in seen
    ], dtype=torch.float32)
    target_attributes = torch.tensor([
        attributes_payload[str(unseen[0])][key] for key in attribute_keys
    ], dtype=torch.float32)

    variants = predict_relation_variants(
        support_attributes, target_attributes, seen_centers,
    )
    raw_center = variants["raw"]
    ranked_center = variants["ranked"]
    robust_center = variants["robust"]
    ranked_bias, ranked_proxy_h, ranked_router, ranked_density_router = select_csd_bias(
        train_features, train_labels, seen, support_attributes, seen_centers, "ranked",
    )
    robust_bias, robust_proxy_h, _, _ = select_csd_bias(
        train_features, train_labels, seen, support_attributes, seen_centers, "robust",
    )
    seen_label_to_index = {class_id: index for index, class_id in enumerate(seen)}
    conformal_calibration = fit_seen_conformal(
        F.normalize(train_features.float(), dim=1)
        @ F.normalize(seen_centers.float(), dim=1).T,
        torch.tensor([seen_label_to_index[int(value)] for value in train_labels]),
    )

    label_to_index = {class_id: index for index, class_id in enumerate(all_classes)}
    test_mask = (~train_mask) & np.isin(labels, all_classes)
    test_features = features[torch.from_numpy(test_mask)]
    test_labels = torch.tensor([label_to_index[int(value)] for value in labels[test_mask]])
    seen_indices = [label_to_index[class_id] for class_id in seen]
    unseen_indices = [label_to_index[class_id] for class_id in unseen]

    def metrics(center: torch.Tensor, bias: float = 0.0) -> dict:
        return evaluate_prototype_bank(
            test_features, test_labels, torch.cat([seen_centers, center[None]]),
            seen_indices, unseen_indices, unseen_bias=bias,
        )

    def router_metrics(center: torch.Tensor, bias: float, router: dict) -> dict:
        bank = torch.cat([seen_centers, center[None]])
        scores = F.normalize(test_features.float(), dim=1) @ F.normalize(bank.float(), dim=1).T
        inputs = build_router_features(
            test_features, scores, seen_indices, unseen_indices,
            seen_centers, seen_variances,
        )
        probability = router_probability(inputs, router)
        predictions = apply_router_csd(
            scores, unseen_indices, bias, probability,
        ).argmax(1)
        result = gzsl_metrics(
            predictions.numpy(), test_labels.numpy(), seen_indices, unseen_indices,
        )
        result["mean_unseen_probability"] = float(probability.mean())
        return result

    def conformal_router_metrics(center: torch.Tensor, bias: float, router: dict) -> dict:
        bank = torch.cat([seen_centers, center[None]])
        scores = F.normalize(test_features.float(), dim=1) @ F.normalize(bank.float(), dim=1).T
        inputs = build_router_features(
            test_features, scores, seen_indices, unseen_indices,
            seen_centers, seen_variances,
        )
        seen_probability = seen_conformal_probability(
            scores[:, seen_indices], conformal_calibration,
        )
        probability = apply_conformal_shield(
            router_probability(inputs, router), seen_probability,
        )
        predictions = apply_router_csd(
            scores, unseen_indices, bias, probability,
        ).argmax(1)
        result = gzsl_metrics(
            predictions.numpy(), test_labels.numpy(), seen_indices, unseen_indices,
        )
        result["mean_unseen_probability"] = float(probability.mean())
        result["mean_seen_probability"] = float(seen_probability.mean())
        return result

    def density_router_metrics(center: torch.Tensor, bias: float, router: dict) -> dict:
        bank = torch.cat([seen_centers, center[None]])
        scores = F.normalize(test_features.float(), dim=1) @ F.normalize(bank.float(), dim=1).T
        inputs = build_density_router_features(
            test_features, scores, seen_indices, unseen_indices,
            seen_centers, seen_variances, center,
            transfer_unseen_variance(center, seen_centers, seen_variances),
        )
        probability = router_probability(inputs, router)
        predictions = apply_router_csd(
            scores, unseen_indices, bias, probability,
        ).argmax(1)
        result = gzsl_metrics(
            predictions.numpy(), test_labels.numpy(), seen_indices, unseen_indices,
        )
        result["mean_unseen_probability"] = float(probability.mean())
        return result

    names = class_names_from_config(config)
    methods = {
        "raw": metrics(raw_center),
        "ranked": metrics(ranked_center),
        "robust": metrics(robust_center),
        "ranked_csd": metrics(ranked_center, ranked_bias),
        "ranked_router_csd": router_metrics(ranked_center, ranked_bias, ranked_router),
        "ranked_conformal_router_csd": conformal_router_metrics(
            ranked_center, ranked_bias, ranked_router,
        ),
        "ranked_density_router_csd": density_router_metrics(
            ranked_center, ranked_bias, ranked_density_router,
        ),
        "robust_csd": metrics(robust_center, robust_bias),
    }
    for value in methods.values():
        value["per_class_named"] = {
            f"{all_classes[index]}:{names[all_classes[index]]}": score
            for index, score in value["per_class"].items()
        }
    payload = {
        "unseen": unseen[0],
        "selected": variants["selected"],
        "loco_error": variants["loco_error"],
        "csd": {
            "ranked_bias": ranked_bias,
            "ranked_proxy_h": ranked_proxy_h,
            "robust_bias": robust_bias,
            "robust_proxy_h": robust_proxy_h,
        },
        "methods": methods,
    }
    if baseline_checkpoint is not None:
        baseline = _torch_load(baseline_checkpoint)["metrics"]
        payload["baseline"] = {
            key: baseline[key]
            for key in ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H", "per_class_named")
        }
        payload["baseline_no_csd"] = {
            key: baseline["no_csd"][key]
            for key in ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H", "per_class_named")
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument(
        "--attributes", default="data/processed/PaviaU_structured_attributes.json",
    )
    parser.add_argument("--output", default="checkpoints/p1_final_gzsl.json")
    parser.add_argument("--unseen-classes", nargs="+", type=int)
    parser.add_argument(
        "--backbone-pattern", default="checkpoints/paviau_p1_backbone_s{unseen}.pt",
    )
    parser.add_argument("--baseline-pattern")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    attribute_path = resolve_project_path(root, args.attributes)
    jobs = (
        [(args.configs[0], unseen) for unseen in args.unseen_classes]
        if args.unseen_classes
        else [(config_name, None) for config_name in args.configs]
    )
    rows = []
    for config_name, unseen_override in jobs:
        config_path = resolve_project_path(root, config_name)
        config = load_config(config_path)
        if unseen_override is not None:
            set_single_unseen_class(config, unseen_override)
        unseen = class_id_lists(config)[1][0]
        rows.append(evaluate_split(
            config_path,
            resolve_project_path(root, args.backbone_pattern.format(unseen=unseen)),
            attribute_path,
            None if args.no_baseline or not args.baseline_pattern else resolve_project_path(
                root, args.baseline_pattern.format(unseen=unseen),
            ),
            root,
            unseen_override=unseen_override,
        ))
    keys = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
    methods = (
        "raw", "ranked", "robust", "ranked_csd", "ranked_router_csd",
        "ranked_conformal_router_csd", "ranked_density_router_csd", "robust_csd",
    )
    summary = {
        method: {
            key: float(np.mean([row["methods"][method][key] for row in rows]))
            for key in keys
        }
        for method in methods
    }
    for method in ("baseline", "baseline_no_csd"):
        if all(method in row for row in rows):
            summary[method] = {
                key: float(np.mean([row[method][key] for row in rows])) for key in keys
            }
    output = resolve_project_path(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
