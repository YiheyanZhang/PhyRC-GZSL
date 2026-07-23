from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_rstd_ot import resolve_project_path
from evaluate_p1 import (
    apply_conformal_shield,
    apply_router_csd,
    build_loco_episodes,
    build_router_features,
    choose_best_bias,
    fit_seen_conformal,
    fit_seen_unseen_router,
    predict_relation_variants,
    predict_prototype_set,
    predict_learned_relation,
    router_probability,
    seen_conformal_probability,
    build_seen_mode_distributions,
    mode_distribution_temperature,
    physical_distribution_score_matrix,
    hybrid_distribution_score_matrix,
    predict_physical_distribution,
)
from models.backbone import build_backbone
from phyrc.calibration import (
    calibrated_domain_probability,
    cross_fit_domain_scores,
    dual_domain_pvalues,
    fit_dual_calibration,
    select_dual_candidate,
)
from phyrc.attention import (
    fit_gzsl_attention_decoder, predict_attention_center, prototype_set_scores,
)
from phyrc.decoder import (
    adaptive_joint_scores,
    joint_risk_scores,
    learn_nested_adaptive_parameters,
    learned_joint_scores,
    learn_nested_risk_parameters,
    select_decoder_candidate,
)
from utils.config import (
    class_id_lists,
    load_config,
    resolve_device,
    set_seed,
    set_unseen_classes,
)
from utils.data_loader import flatten_hsi, load_paviac_from_config, stratified_train_mask
from utils.metrics import gzsl_metrics


_BIASES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
_BETAS = (0.25, 0.5, 1.0, 2.0)
_TEMPERATURES = (0.7, 1.0, 1.3)
_RISK_WEIGHTS = (0.05, 0.1, 0.2, 0.4)
_UNSEEN_PRIORS = (0.0, 0.05, 0.1, 0.2, 0.3)
_ABLATIONS = (
    "full", "no_relational_prototype", "no_cross_fitting",
    "no_dual_evidence", "no_risk_constraint", "learned_rcjd", "learned_all", "adaptive_rcjd",
    "primal_dual_rcjd",
    "relation_cvar",
    "prototype_attention", "decoder_attention", "prototype_set", "physical_distribution",
    "hybrid_distribution",
)


def _mean_metrics(results: list[dict]) -> dict:
    keys = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
    return {key: float(np.mean([result[key] for result in results])) for key in keys}


def _relation_episode_weights(attributes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = attributes.float().std(0, unbiased=False).clamp_min(1e-6)
    distances = ((attributes.float() - target.float()) / scale).square().sum(1).sqrt()
    temperature = distances[distances > 0].median().clamp_min(1e-6)
    return torch.softmax(-distances / temperature, dim=0)


def _predict_ranked_centers(
    attributes: torch.Tensor,
    target_attributes: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    targets = target_attributes[None] if target_attributes.ndim == 1 else target_attributes
    return torch.stack([
        predict_relation_variants(attributes, target, centers)["ranked"]
        for target in targets
    ])


def _validate_backbone_partition(backbone_checkpoint: str | Path, seen: list[int]) -> None:
    try:
        payload = torch.load(Path(backbone_checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(backbone_checkpoint), map_location="cpu")
    stored = payload.get("seen_classes") if isinstance(payload, dict) else None
    if stored is None or [int(value) for value in stored] != [int(value) for value in seen]:
        raise ValueError("backbone checkpoint partition does not match requested Seen classes")


def _relation_metrics(results: list[dict], weights: torch.Tensor) -> dict:
    keys = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
    weights = weights / weights.sum()
    metrics = {
        f"relation_{key}": float(sum(weights[i] * results[i][key] for i in range(len(results))))
        for key in keys
    }
    order = sorted(range(len(results)), key=lambda i: results[i]["H"])
    remaining, tail = 0.3, 0.0
    for index in order:
        mass = min(float(weights[index]), remaining)
        tail += mass * results[index]["H"]
        remaining -= mass
        if remaining <= 1e-9:
            break
    metrics["relation_cvar"] = tail / (0.3 - remaining)
    return metrics


def _evaluate_episode_probability(episode: dict, probability: torch.Tensor, bias: float) -> dict:
    unseen_index = episode["scores"].shape[1] - 1
    predictions = apply_router_csd(
        episode["scores"], [unseen_index], bias, probability,
    ).argmax(1)
    return gzsl_metrics(
        predictions.numpy(), episode["labels"].numpy(),
        list(range(unseen_index)), [unseen_index],
    )


def select_cfdc_parameters(
    episodes: list[dict], *, cross_fitting: bool = True,
    dual_evidence: bool = True, risk_constraint: bool = True,
    learnable: bool = False, adaptive: bool = False, primal_dual: bool = False,
    episode_weights: torch.Tensor | None = None,
) -> dict:
    inputs = torch.cat([episode["router_inputs"] for episode in episodes])
    labels = torch.cat([episode["router_labels"] for episode in episodes])
    episode_ids = torch.cat([
        torch.full((len(episode["labels"]),), index, dtype=torch.long)
        for index, episode in enumerate(episodes)
    ])
    if cross_fitting:
        cross_fitted = cross_fit_domain_scores(inputs, labels, episode_ids)
    else:
        router = fit_seen_unseen_router(inputs, labels)
        cross_fitted = router_probability(inputs, router)
    calibration = fit_dual_calibration(cross_fitted, labels)
    p_seen, p_unseen = dual_domain_pvalues(cross_fitted, calibration)
    if not dual_evidence:
        p_seen = torch.ones_like(p_seen)
        p_unseen = torch.ones_like(p_unseen)
    offsets = torch.tensor([0] + list(np.cumsum([len(episode["labels"]) for episode in episodes])))

    h_scores = torch.zeros(len(_BIASES), len(episodes))
    for bias_index, bias in enumerate(_BIASES):
        for episode_index, episode in enumerate(episodes):
            result = _evaluate_episode_probability(
                episode,
                torch.ones(len(episode["labels"])),
                bias,
            )
            h_scores[bias_index, episode_index] = result["H"]
    p1_bias = choose_best_bias(_BIASES, h_scores)

    p1_results = []
    for index, episode in enumerate(episodes):
        start, end = int(offsets[index]), int(offsets[index + 1])
        unseen_index = episode["scores"].shape[1] - 1
        seen_mask = episode["labels"] != unseen_index
        seen_calibration = fit_seen_conformal(
            episode["scores"][seen_mask, :-1], episode["labels"][seen_mask],
        )
        shielded = apply_conformal_shield(
            cross_fitted[start:end],
            seen_conformal_probability(episode["scores"][:, :-1], seen_calibration),
        )
        p1_results.append(_evaluate_episode_probability(episode, shielded, p1_bias))
    p1 = {
        "mode": "p1", "bias": p1_bias, "beta": 0.0,
        "seen_zero": int(sum(
            sum(result["per_class"][i] == 0.0 for i in range(len(result["per_class"]) - 1))
            for result in p1_results
        )),
        "worst_h": float(min(result["H"] for result in p1_results)),
        **_mean_metrics(p1_results),
        **(_relation_metrics(p1_results, episode_weights) if episode_weights is not None else {}),
    }

    candidates = []
    for beta in _BETAS:
        calibrated = calibrated_domain_probability(cross_fitted, p_seen, p_unseen, beta)
        for bias in _BIASES:
            results = [
                _evaluate_episode_probability(
                    episode,
                    calibrated[int(offsets[index]) : int(offsets[index + 1])],
                    bias,
                )
                for index, episode in enumerate(episodes)
            ]
            candidates.append({
                "mode": "dual", "bias": bias, "beta": beta,
                "seen_zero": int(sum(
                    sum(result["per_class"][i] == 0.0 for i in range(len(result["per_class"]) - 1))
                    for result in results
                )),
                **_mean_metrics(results),
            })
    rcjd_candidates = []
    for temperature in _TEMPERATURES:
        for risk_weight in _RISK_WEIGHTS:
            for unseen_prior in _UNSEEN_PRIORS:
                results = []
                for index, episode in enumerate(episodes):
                    start, end = int(offsets[index]), int(offsets[index + 1])
                    unseen_index = episode["scores"].shape[1] - 1
                    adjusted = joint_risk_scores(
                        episode["scores"], list(range(unseen_index)), [unseen_index],
                        p_seen[start:end], p_unseen[start:end], temperature,
                        risk_weight, unseen_prior,
                    )
                    results.append(gzsl_metrics(
                        adjusted.argmax(1).numpy(), episode["labels"].numpy(),
                        list(range(unseen_index)), [unseen_index],
                    ))
                rcjd_candidates.append({
                    "mode": "rcjd",
                    "temperature": temperature,
                    "risk_weight": risk_weight,
                    "unseen_prior": unseen_prior,
                    "seen_zero": int(sum(
                        sum(result["per_class"][i] == 0.0 for i in range(len(result["per_class"]) - 1))
                        for result in results
                    )),
                    "worst_h": float(min(result["H"] for result in results)),
                    **_mean_metrics(results),
                    **(_relation_metrics(results, episode_weights) if episode_weights is not None else {}),
                })
    reference = select_decoder_candidate(rcjd_candidates, p1)
    if episode_weights is not None:
        keys = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
        feasible = [row for row in rcjd_candidates if (
            row["seen_zero"] <= reference["seen_zero"] and
            all(row[f"relation_{key}"] >= reference[f"relation_{key}"] for key in keys)
        )]
        selected = max(feasible + [reference], key=lambda row: (
            row["relation_cvar"], row["relation_H"], row["relation_Unseen_AA"], row["relation_OA"],
        ))
    elif adaptive:
        selected = {
            "mode": "adaptive_rcjd",
            **learn_nested_adaptive_parameters(episodes, p_seen, p_unseen, reference),
        }
    elif primal_dual:
        learned = learn_nested_risk_parameters(
            episodes, p_seen, p_unseen, reference=reference,
        )
        learned_results = []
        for index, episode in enumerate(episodes):
            start, end = int(offsets[index]), int(offsets[index + 1])
            unseen_index = episode["scores"].shape[1] - 1
            adjusted = learned_joint_scores(
                episode["scores"], list(range(unseen_index)), [unseen_index],
                p_seen[start:end], p_unseen[start:end],
                learned["alpha"], learned["delta"],
            )
            learned_results.append(gzsl_metrics(
                adjusted.argmax(1).numpy(), episode["labels"].numpy(),
                list(range(unseen_index)), [unseen_index],
            ))
        learned_candidate = {
            "mode": "learned_rcjd", **learned,
            "seen_zero": int(sum(
                sum(result["per_class"][i] == 0.0 for i in range(len(result["per_class"]) - 1))
                for result in learned_results
            )),
            "worst_h": float(min(result["H"] for result in learned_results)),
            **_mean_metrics(learned_results),
        }
        selected = select_decoder_candidate([learned_candidate], reference)
    elif learnable:
        learned = learn_nested_risk_parameters(episodes, p_seen, p_unseen)
        selected = {"mode": "learned_rcjd", **learned}
    elif risk_constraint:
        selected = reference
    else:
        selected = max(rcjd_candidates, key=lambda row: (
            row.get("worst_h", row["H"]), row["H"], row["Unseen_AA"], row["OA"],
        ))
    return {
        "selected": selected,
        "dual_selected": select_dual_candidate(candidates, p1),
        "p1_proxy": p1,
        "candidates": candidates,
        "rcjd_candidates": rcjd_candidates,
        "calibration": calibration,
        "cross_fitted_scores": cross_fitted,
        "cross_fitted_labels": labels,
    }


def _binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    order = scores.float().argsort()
    sorted_scores = scores.float()[order]
    sorted_labels = (labels[order] > 0.5).float()
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    cumulative_positive = sorted_labels.cumsum(0)
    positive = cumulative_positive[counts.cumsum(0) - 1]
    positive = torch.diff(torch.cat([positive.new_zeros(1), positive]))
    negative = counts.to(positive) - positive
    negative_before = negative.cumsum(0) - negative
    total_positive, total_negative = positive.sum(), negative.sum()
    if not bool(total_positive > 0 and total_negative > 0):
        raise ValueError("AUROC requires both domains")
    return float((positive * (negative_before + 0.5 * negative)).sum() / (total_positive * total_negative))


def _domain_diagnostics(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    auroc = _binary_auroc(scores, labels)
    brier = float((scores - labels.float()).square().mean())
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        mask = (scores >= lower) & (scores < lower + 0.1)
        if mask.any():
            ece += float(mask.float().mean() * (scores[mask].mean() - labels[mask].float().mean()).abs())
    return {"AUROC": auroc, "Brier": brier, "ECE": ece}


def evaluate_stage2_split(
    config_path: Path,
    backbone_checkpoint: Path,
    attribute_path: Path,
    root: Path,
    unseen_override: int | list[int],
    seed_override: int | None = None,
    ablation: str = "full",
) -> dict:
    if ablation not in _ABLATIONS:
        raise ValueError(f"unknown ablation: {ablation}")
    config = load_config(config_path)
    unseen_requested = (
        [int(unseen_override)] if isinstance(unseen_override, (int, np.integer))
        else [int(value) for value in unseen_override]
    )
    set_unseen_classes(config, unseen_requested)
    if len(unseen_requested) > 1 and ablation != "full":
        raise ValueError("multi-Unseen evaluation is supported only for the frozen Full method")
    if seed_override is not None:
        config.setdefault("runtime", {})["seed"] = seed_override
        config.setdefault("data_split", {})["seed"] = seed_override
    seen, unseen, all_classes = class_id_lists(config)
    _validate_backbone_partition(backbone_checkpoint, seen)
    config["model"].update({
        "backbone": "spectral_morphology",
        "pretrained_backbone": str(backbone_checkpoint),
        "freeze_backbone": True,
    })
    set_seed(int(config["runtime"]["seed"]))
    device = resolve_device(config["runtime"].get("device", "auto"))
    x, gt = load_paviac_from_config(config, root)
    spectra, labels = flatten_hsi(x, gt, ignore_background=True)
    train_mask = stratified_train_mask(
        labels, seen, float(config["data_split"]["train_ratio"]),
        int(config["data_split"]["seed"]),
    )
    model = build_backbone(x.shape[-1], config).to(device).eval()
    with torch.no_grad():
        features = torch.cat([
            model(batch.to(device)).cpu()
            for batch in torch.from_numpy(spectra.astype(np.float32)).split(512)
        ])
    train_features = features[torch.from_numpy(train_mask)]
    train_labels = torch.from_numpy(labels[train_mask].astype(np.int64))
    centers = torch.stack([
        train_features[train_labels == class_id].mean(0) for class_id in seen
    ])
    variances = torch.stack([
        train_features[train_labels == class_id].var(0, unbiased=False) for class_id in seen
    ])
    payload = json.loads(attribute_path.read_text(encoding="utf-8"))
    keys = (
        "brightness", "visible_slope", "green_peak", "red_absorption",
        "red_edge", "nir_level", "smoothness", "variability",
    )
    attributes = torch.tensor([
        [payload[str(class_id)][key] for key in keys] for class_id in seen
    ], dtype=torch.float32)
    target_attributes = torch.tensor([
        [payload[str(class_id)][key] for key in keys] for class_id in unseen
    ], dtype=torch.float32)
    if len(unseen) == 1:
        target_attributes = target_attributes[0]
    prototype_method = (
        "semantic" if ablation == "no_relational_prototype" else
        "learned" if ablation == "learned_all" else "ranked"
    )
    prototype_candidates = None
    physical_distribution = None
    hybrid_distribution = None
    if ablation == "physical_distribution":
        prototype_method = "physical_distribution"
        seen_modes, seen_mode_weights = build_seen_mode_distributions(
            train_features, train_labels, seen,
        )
        unseen_modes, unseen_mode_weights, center = predict_physical_distribution(
            attributes, target_attributes, centers, seen_modes, seen_mode_weights,
        )
        distribution_temperature = mode_distribution_temperature(centers, seen_modes)
        physical_distribution = (
            seen_modes, seen_mode_weights, unseen_modes, unseen_mode_weights,
            distribution_temperature,
        )
        prototype_parameters = {"temperature": distribution_temperature}
    elif ablation == "hybrid_distribution":
        prototype_method = "hybrid_distribution"
        seen_modes, seen_mode_weights = build_seen_mode_distributions(
            train_features, train_labels, seen,
        )
        prototype_candidates, prototype_parameters = predict_prototype_set(
            attributes, target_attributes, centers,
        )
        center = prototype_candidates.mean(0)
        distribution_temperature = mode_distribution_temperature(centers, seen_modes)
        hybrid_distribution = (
            seen_modes, seen_mode_weights, prototype_candidates, distribution_temperature,
        )
        prototype_parameters["temperature"] = distribution_temperature
    elif ablation == "prototype_set":
        prototype_method = "prototype_set"
        prototype_candidates, prototype_parameters = predict_prototype_set(
            attributes, target_attributes, centers,
        )
        center = prototype_candidates.mean(0)
    elif ablation == "prototype_attention":
        prototype_method = "attention"
        center, prototype_parameters = predict_attention_center(
            attributes, target_attributes, centers,
        )
    elif prototype_method == "learned":
        center, prototype_parameters = predict_learned_relation(
            attributes, target_attributes, centers,
        )
    elif len(unseen) > 1:
        center = _predict_ranked_centers(attributes, target_attributes, centers)
        prototype_parameters = None
    else:
        center = predict_relation_variants(attributes, target_attributes, centers)[prototype_method]
        prototype_parameters = None

    episodes = build_loco_episodes(
        train_features, train_labels, seen, attributes, centers, prototype_method,
    )
    if ablation == "decoder_attention":
        for episode in episodes:
            unseen_index = episode["scores"].shape[1] - 1
            support = episode["labels"] != unseen_index
            decoder, mean, scale = fit_gzsl_attention_decoder(
                episode["features"][support], episode["labels"][support],
                episode["bank"][:-1], episode["bank_attributes"][:-1],
            )
            with torch.no_grad():
                episode["scores"] = decoder(
                    episode["features"], episode["bank"],
                    (episode["bank_attributes"] - mean) / scale,
                )
            episode["router_inputs"] = build_router_features(
                episode["features"], episode["scores"], list(range(unseen_index)),
                [unseen_index], episode["bank"][:-1], episode["variances"],
            )
    selection = select_cfdc_parameters(
        episodes,
        cross_fitting=ablation != "no_cross_fitting",
        dual_evidence=ablation != "no_dual_evidence",
        risk_constraint=ablation != "no_risk_constraint",
        learnable=ablation in {"learned_rcjd", "learned_all"},
        adaptive=ablation == "adaptive_rcjd",
        primal_dual=ablation == "primal_dual_rcjd",
        episode_weights=(
            _relation_episode_weights(attributes, target_attributes)
            if ablation == "relation_cvar" else None
        ),
    )
    router = fit_seen_unseen_router(
        torch.cat([episode["router_inputs"] for episode in episodes]),
        torch.cat([episode["router_labels"] for episode in episodes]),
    )

    label_to_index = {class_id: index for index, class_id in enumerate(all_classes)}
    test_mask = (~train_mask) & np.isin(labels, all_classes)
    test_features = features[torch.from_numpy(test_mask)]
    test_labels = torch.tensor([label_to_index[int(value)] for value in labels[test_mask]])
    seen_indices = [label_to_index[class_id] for class_id in seen]
    unseen_indices = [label_to_index[class_id] for class_id in unseen]
    unseen_centers = center if center.ndim == 2 else center[None]
    bank = torch.cat([centers, unseen_centers])
    if ablation == "decoder_attention":
        local_labels = torch.tensor([seen.index(int(value)) for value in train_labels])
        decoder, mean, scale = fit_gzsl_attention_decoder(
            train_features, local_labels, centers, attributes,
        )
        bank_attributes = torch.cat([attributes, target_attributes[None]])
        with torch.no_grad():
            scores = decoder(test_features, bank, (bank_attributes - mean) / scale)
    else:
        scores = F.normalize(test_features.float(), dim=1) @ F.normalize(bank.float(), dim=1).T
        if prototype_candidates is not None:
            scores[:, -1] = prototype_set_scores(test_features, prototype_candidates)
    if physical_distribution is not None:
        scores = physical_distribution_score_matrix(
            test_features, *physical_distribution,
        )
    if hybrid_distribution is not None:
        scores = hybrid_distribution_score_matrix(
            test_features, *hybrid_distribution,
        )
    raw_probability = router_probability(
        build_router_features(
            test_features, scores, seen_indices, unseen_indices, centers, variances,
        ),
        router,
    )
    selected = selection["selected"]
    if selected["mode"] in {"rcjd", "learned_rcjd", "adaptive_rcjd"}:
        if ablation == "no_dual_evidence":
            p_seen = p_unseen = torch.ones_like(raw_probability)
        else:
            p_seen, p_unseen = dual_domain_pvalues(raw_probability, selection["calibration"])
        final_scores = (
            adaptive_joint_scores(
                scores, seen_indices, unseen_indices, p_seen, p_unseen,
                a=selected["a"], b=selected["b"], c=selected["c"], d=selected["d"],
            )
            if selected["mode"] == "adaptive_rcjd" else
            learned_joint_scores(
                scores, seen_indices, unseen_indices, p_seen, p_unseen,
                selected["alpha"], selected["delta"],
            )
            if selected["mode"] == "learned_rcjd" else
            joint_risk_scores(
                scores, seen_indices, unseen_indices, p_seen, p_unseen,
                selected["temperature"], selected["risk_weight"], selected["unseen_prior"],
            )
        )
        predictions = final_scores.argmax(1)
    else:
        seen_label_to_index = {class_id: index for index, class_id in enumerate(seen)}
        calibration = fit_seen_conformal(
            F.normalize(train_features.float(), dim=1) @ F.normalize(centers.float(), dim=1).T,
            torch.tensor([seen_label_to_index[int(value)] for value in train_labels]),
        )
        probability = apply_conformal_shield(
            raw_probability,
            seen_conformal_probability(scores[:, seen_indices], calibration),
        )
        predictions = apply_router_csd(
            scores, unseen_indices, selected["bias"], probability,
        ).argmax(1)
    result = gzsl_metrics(
        predictions.numpy(), test_labels.numpy(), seen_indices, unseen_indices,
    )
    diagnostics = _domain_diagnostics(
        selection["cross_fitted_scores"], selection["cross_fitted_labels"],
    )
    serializable_selection = {
        "selected": selected,
        "dual_selected": selection["dual_selected"],
        "p1_proxy": selection["p1_proxy"],
        "candidates": selection["candidates"],
        "rcjd_candidates": selection["rcjd_candidates"],
    }
    return {
        "ablation": ablation,
        "unseen": unseen[0] if len(unseen) == 1 else unseen,
        "unseen_classes": unseen,
        "prototype_parameters": prototype_parameters,
        "selection": serializable_selection,
        "diagnostics": diagnostics,
        "metrics": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--unseen-classes", nargs="+", type=int, required=True)
    parser.add_argument("--backbone-pattern", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ablation", choices=_ABLATIONS, default="full")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = resolve_project_path(root, args.config)
    attribute_path = resolve_project_path(root, args.attributes)
    rows = [
        evaluate_stage2_split(
            config_path,
            resolve_project_path(root, args.backbone_pattern.format(unseen=unseen)),
            attribute_path,
            root,
            unseen,
            args.seed,
            args.ablation,
        )
        for unseen in args.unseen_classes
    ]
    metric_keys = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
    summary = {
        key: float(np.mean([row["metrics"][key] for row in rows])) for key in metric_keys
    }
    summary["seen_zero"] = int(sum(
        sum(value == 0.0 for index, value in row["metrics"]["per_class"].items()
            if index < len(row["metrics"]["per_class"]) - 1)
        for row in rows
    ))
    summary["rcjd_selected"] = int(sum(
        row["selection"]["selected"]["mode"] == "rcjd" for row in rows
    ))
    summary["AUROC"] = float(np.mean([row["diagnostics"]["AUROC"] for row in rows]))
    summary["ECE"] = float(np.mean([row["diagnostics"]["ECE"] for row in rows]))
    output = resolve_project_path(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"ablation": args.ablation, "summary": summary, "results": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
