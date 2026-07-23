from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.backbone import (
    PhysicsInvariantBackbone,
    SpectralMAEBackbone,
    SpectralMorphologyBackbone,
    WavelengthTransformerBackbone,
    load_backbone_checkpoint,
)
from utils.config import (
    class_id_lists, load_config, resolve_device, set_seed,
    set_single_unseen_class, set_unseen_classes,
)
from utils.data_loader import flatten_hsi, load_paviac_from_config, stratified_train_mask
from utils.text_encoder import SemanticConditionEncoder


def make_band_mask(spectra: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(spectra) < float(mask_ratio)
    if not bool(mask.any()):
        mask[:, 0] = True
    masked = spectra.clone()
    masked[mask] = 0.0
    return masked, mask


def make_band_span_mask(spectra: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    span = max(1, min(spectra.shape[1], int(round(spectra.shape[1] * float(mask_ratio)))))
    starts = torch.randint(0, spectra.shape[1] - span + 1, (spectra.shape[0],), device=spectra.device)
    positions = torch.arange(spectra.shape[1], device=spectra.device)[None]
    mask = (positions >= starts[:, None]) & (positions < starts[:, None] + span)
    return spectra.masked_fill(mask, 0.0), mask


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    return diff[mask].mean()


def spectral_angle_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - torch.nn.functional.cosine_similarity(prediction, target, dim=1, eps=1e-8)).mean()


def physics_augment(spectra: torch.Tensor) -> torch.Tensor:
    scale = torch.empty((spectra.shape[0], 1), device=spectra.device).uniform_(0.8, 1.2)
    baseline = torch.empty((spectra.shape[0], 1), device=spectra.device).uniform_(-0.05, 0.05)
    noise = 0.01 * spectra.std(dim=1, keepdim=True, unbiased=False) * torch.randn_like(spectra)
    return scale * spectra + baseline + noise


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    features = torch.nn.functional.normalize(features, dim=1)
    labels = labels.view(-1, 1)
    positive = torch.eq(labels, labels.T).float().to(features.device)
    positive.fill_diagonal_(0.0)
    if not bool(positive.sum() > 0):
        return features.new_tensor(0.0)
    logits = features @ features.T / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(positive)
    logits_mask.fill_diagonal_(0.0)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (positive * log_prob).sum(dim=1) / positive.sum(dim=1).clamp_min(1.0)
    valid = positive.sum(dim=1) > 0
    return -mean_log_prob_pos[valid].mean()


def semantic_geometry_loss(features: torch.Tensor, semantics: torch.Tensor, labels: torch.Tensor):
    classes = torch.unique(labels)
    if classes.numel() < 2:
        zero = features.new_tensor(0.0)
        return zero, zero
    feature_centers = torch.stack([features[labels == value].mean(0) for value in classes])
    semantic_centers = torch.stack([semantics[labels == value].mean(0) for value in classes])
    pairs = torch.triu_indices(classes.numel(), classes.numel(), offset=1, device=features.device)
    feature_delta = feature_centers[pairs[1]] - feature_centers[pairs[0]]
    semantic_delta = semantic_centers[pairs[1]] - semantic_centers[pairs[0]]
    direction = (1.0 - torch.nn.functional.cosine_similarity(feature_delta, semantic_delta, dim=1)).mean()
    feature_distance = feature_delta.norm(dim=1)
    semantic_distance = semantic_delta.norm(dim=1)
    topology = torch.nn.functional.smooth_l1_loss(
        feature_distance / feature_distance.mean().clamp_min(1e-6),
        semantic_distance / semantic_distance.mean().clamp_min(1e-6),
    )
    return direction, topology


def main() -> None:
    parser = argparse.ArgumentParser(description="Seen-only unlabeled spectral MAE pretraining.")
    parser.add_argument("--config", default="configs/paviau_spectral_mae.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--backbone", choices=("spectral_mae", "spectral_morphology", "physics_invariant", "wavelength_transformer"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--distill-weight", type=float)
    parser.add_argument("--semantic-direction-weight", type=float)
    parser.add_argument("--semantic-topology-weight", type=float)
    unseen_group = parser.add_mutually_exclusive_group()
    unseen_group.add_argument("--unseen-class", type=int)
    unseen_group.add_argument("--unseen-classes", nargs="+", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = load_config(args.config)
    if args.unseen_class is not None:
        set_single_unseen_class(config, args.unseen_class)
    elif args.unseen_classes is not None:
        set_unseen_classes(config, args.unseen_classes)
    if args.backbone:
        config.setdefault("model", {})["backbone"] = args.backbone
    if args.checkpoint:
        config.setdefault("spectral_mae", {})["checkpoint"] = args.checkpoint
    if args.seed is not None:
        config.setdefault("runtime", {})["seed"] = args.seed
        config.setdefault("data_split", {})["seed"] = args.seed
    set_seed(int(config.get("runtime", {}).get("seed", 42)))
    device = resolve_device(config.get("runtime", {}).get("device", "auto"))

    seen, _, _ = class_id_lists(config)
    x, gt = load_paviac_from_config(config, project_root)
    cfg = config.get("spectral_mae", {})
    model_cfg = config.get("model", {})
    split_cfg = config.get("data_split", {})
    train_ratio = float(split_cfg.get("train_ratio", 1.0))
    split_seed = int(split_cfg.get("seed", config.get("runtime", {}).get("seed", 42)))
    spectra, labels = flatten_hsi(x, gt, ignore_background=True)
    seen_train_mask = stratified_train_mask(labels, seen, train_ratio=train_ratio, seed=split_seed)
    spectra = spectra[seen_train_mask].astype(np.float32)
    labels = labels[seen_train_mask].astype(np.int64)
    loader = DataLoader(TensorDataset(torch.from_numpy(spectra).float(), torch.from_numpy(labels).long()), batch_size=int(cfg.get("batch_size", 512)), shuffle=True, num_workers=int(config.get("runtime", {}).get("num_workers", 0)))
    backbone_name = str(model_cfg.get("backbone", "spectral_mae")).lower()
    backbone_class = {
        "spectral_mae": SpectralMAEBackbone,
        "spectral_morphology": SpectralMorphologyBackbone,
        "physics_invariant": PhysicsInvariantBackbone,
        "wavelength_transformer": WavelengthTransformerBackbone,
    }[backbone_name]
    extra = ({"num_layers": int(cfg.get("num_layers", 2)), "num_heads": int(cfg.get("num_heads", 4))}
             if backbone_name == "wavelength_transformer" else {})
    model = backbone_class(
        input_bands=x.shape[-1], feature_dim=int(model_cfg.get("feature_dim", cfg.get("feature_dim", 64))),
        hidden_dim=int(cfg.get("hidden_dim", 256)), **extra,
    ).to(device)
    teacher = None
    semantic_mapper = None
    semantic_by_class = None
    teacher_checkpoint = args.teacher_checkpoint or cfg.get("teacher_checkpoint")
    if backbone_name in ("spectral_morphology", "physics_invariant") and teacher_checkpoint:
        teacher_path = Path(teacher_checkpoint)
        if not teacher_path.is_absolute():
            teacher_path = project_root / teacher_path
        teacher_class = SpectralMorphologyBackbone if backbone_name == "physics_invariant" else SpectralMAEBackbone
        teacher = teacher_class(input_bands=x.shape[-1], feature_dim=model.feature_dim, hidden_dim=int(cfg.get("hidden_dim", 256))).to(device)
        load_backbone_checkpoint(teacher, teacher_path)
        model.base.load_state_dict(teacher.state_dict())
        teacher.freeze()
        teacher.eval()
    direction_weight = float(args.semantic_direction_weight if args.semantic_direction_weight is not None else cfg.get("semantic_direction_weight", 0.0))
    topology_weight = float(args.semantic_topology_weight if args.semantic_topology_weight is not None else cfg.get("semantic_topology_weight", 0.0))
    if direction_weight or topology_weight:
        semantic_by_class = SemanticConditionEncoder(config, project_root).encode_classes(
            config["classes"]["names"]
        )
        condition_dim = int(next(iter(semantic_by_class.values())).numel())
        semantic_mapper = torch.nn.Sequential(
            torch.nn.Linear(condition_dim, 32, bias=False),
            torch.nn.Linear(32, model.feature_dim, bias=False),
        ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + (list(semantic_mapper.parameters()) if semantic_mapper else []),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    mask_ratio = float(cfg.get("mask_ratio", 0.3))
    mae_weight = float(cfg.get("mae_weight", 1.0))
    supcon_weight = float(cfg.get("supcon_weight", 0.0))
    temperature = float(cfg.get("temperature", 0.1))
    distill_weight = float(args.distill_weight if args.distill_weight is not None else cfg.get("distill_weight", 1.0))
    sam_weight = float(cfg.get("sam_weight", 0.1 if backbone_name == "physics_invariant" else 0.0))
    invariant_weight = float(cfg.get("invariant_weight", 0.1 if backbone_name == "physics_invariant" else 0.0))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("epochs", 100))
    checkpoint = Path(cfg.get("checkpoint", "checkpoints/spectral_mae_seen.pt"))
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in tqdm(range(epochs), desc="train_spectral_mae"):
        losses = []
        mae_losses = []
        con_losses = []
        model.train()
        for batch, batch_labels in loader:
            batch = batch.to(device)
            batch_labels = batch_labels.to(device)
            masked, mask = (make_band_span_mask(batch, mask_ratio)
                            if backbone_name == "wavelength_transformer" else make_band_mask(batch, mask_ratio))
            pred = model.reconstruct(masked)
            mae = masked_mse(pred, batch, mask)
            contrastive = supcon_loss(model(batch), batch_labels, temperature=temperature)
            distill = (torch.nn.functional.mse_loss(model(batch), teacher(batch).detach())
                       if teacher is not None else batch.new_tensor(0.0))
            sam = spectral_angle_loss(pred, batch)
            invariant = (1.0 - torch.nn.functional.cosine_similarity(
                model.physics_features(batch), model.physics_features(physics_augment(batch)), dim=1,
            )).mean() if backbone_name == "physics_invariant" else batch.new_tensor(0.0)
            direction = topology = batch.new_tensor(0.0)
            if semantic_mapper is not None:
                semantic = torch.stack([semantic_by_class[int(value)] for value in batch_labels]).to(device)
                direction, topology = semantic_geometry_loss(model(batch), semantic_mapper(semantic), batch_labels)
            loss = (mae_weight * mae + supcon_weight * contrastive + distill_weight * distill
                    + sam_weight * sam + invariant_weight * invariant
                    + direction_weight * direction + topology_weight * topology)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mae_losses.append(float(mae.detach().cpu()))
            con_losses.append(float(contrastive.detach().cpu()))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"epoch={epoch + 1} loss={np.mean(losses):.6f} "
                f"mae={np.mean(mae_losses):.6f} supcon={np.mean(con_losses):.6f}"
            )
        if args.save_every_epoch:
            epoch_checkpoint = checkpoint.with_name(
                f"{checkpoint.stem}_epoch{epoch + 1}{checkpoint.suffix}"
            )
            torch.save(
                {
                    "backbone": model.state_dict(),
                    "input_bands": int(x.shape[-1]),
                    "feature_dim": int(model.feature_dim),
                    "seen_classes": seen,
                    "config": config,
                },
                epoch_checkpoint,
            )

    torch.save(
        {
            "backbone": model.state_dict(),
            "input_bands": int(x.shape[-1]),
            "feature_dim": int(model.feature_dim),
            "seen_classes": seen,
            "config": config,
        },
        checkpoint,
    )
    print(f"saved {checkpoint}")


if __name__ == "__main__":
    main()
