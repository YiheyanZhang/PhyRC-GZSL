from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def class_mode_distribution(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must be a non-empty matrix")
    normalized = F.normalize(features.float(), dim=1)
    centre = F.normalize(normalized.mean(0), dim=0)
    if len(normalized) < 2:
        return torch.stack([centre, centre]), torch.tensor([0.5, 0.5])
    centred = normalized - normalized.mean(0)
    direction = torch.linalg.eigh(centred.T @ centred).eigenvectors[:, -1]
    order = (centred @ direction).argsort()
    split = len(order) // 2
    groups = (order[:split], order[split:])
    if any(len(group) == 0 for group in groups):
        return torch.stack([centre, centre]), torch.tensor([0.5, 0.5])
    modes = torch.stack([F.normalize(normalized[group].mean(0), dim=0) for group in groups])
    weights = torch.tensor([len(group) / len(normalized) for group in groups])
    return modes, weights


def transport_mode_distribution(
    anchor_centres: torch.Tensor,
    source_centres: torch.Tensor,
    source_modes: torch.Tensor,
    source_weights: torch.Tensor,
    anchor_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchors = F.normalize(anchor_centres.float(), dim=1)
    centres = F.normalize(source_centres.float(), dim=1)
    modes = F.normalize(source_modes.float(), dim=2)
    transported = F.normalize(anchors[:, None] + modes - centres[:, None], dim=2)
    weights = anchor_weights.float()[:, None] * source_weights.float()
    weights = weights.flatten()
    return transported.flatten(0, 1), weights / weights.sum().clamp_min(1e-12)


def prototype_distribution_scores(
    features: torch.Tensor,
    modes: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if modes.ndim != 2 or len(modes) != len(weights):
        raise ValueError("modes and weights must align")
    weights = weights.float() / weights.float().sum().clamp_min(1e-12)
    similarities = F.normalize(features.float(), dim=1) @ F.normalize(modes.float(), dim=1).T
    tau = max(float(temperature), 1e-6)
    return tau * torch.logsumexp(similarities / tau + weights.clamp_min(1e-12).log(), dim=1)


def prototype_set_scores(features: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return (
        F.normalize(features.float(), dim=1)
        @ F.normalize(candidates.float(), dim=1).T
    ).max(1).values


class RelationalPrototypeAttention(nn.Module):
    """Small attribute-query attention over seen visual prototypes."""

    def __init__(self, attribute_dim: int, feature_dim: int, hidden_dim: int = 32, heads: int = 2):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.query = nn.Linear(attribute_dim, hidden_dim, bias=False)
        self.attribute_key = nn.Linear(attribute_dim, hidden_dim, bias=False)
        self.center_key = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.heads = heads
        self.head_dim = hidden_dim // heads

    def forward(
        self, support_attributes: torch.Tensor, target_attribute: torch.Tensor,
        support_centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(target_attribute.float()).reshape(self.heads, self.head_dim)
        keys = (
            self.attribute_key(support_attributes.float())
            + self.center_key(F.normalize(support_centers.float(), dim=1))
        ).reshape(len(support_attributes), self.heads, self.head_dim)
        weights = torch.softmax(torch.einsum("hd,shd->hs", query, keys) / math.sqrt(self.head_dim), dim=1)
        center = torch.einsum("hs,sf->hf", weights, support_centers.float()).mean(0)
        return center, weights


class GZSLAttentionDecoder(nn.Module):
    """Feature-query attention over visual-semantic class tokens."""

    def __init__(self, attribute_dim: int, feature_dim: int, heads: int = 4):
        super().__init__()
        if feature_dim % heads:
            raise ValueError("feature_dim must be divisible by heads")
        self.query = nn.Linear(feature_dim, feature_dim, bias=False)
        self.center_key = nn.Linear(feature_dim, feature_dim, bias=False)
        self.attribute_key = nn.Linear(attribute_dim, feature_dim, bias=False)
        self.heads, self.head_dim = heads, feature_dim // heads
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.center_key.weight)
        nn.init.zeros_(self.attribute_key.weight)

    def forward(
        self, features: torch.Tensor, centers: torch.Tensor, attributes: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(F.normalize(features.float(), dim=1)).reshape(-1, self.heads, self.head_dim)
        keys = (
            self.center_key(F.normalize(centers.float(), dim=1))
            + self.attribute_key(attributes.float())
        ).reshape(len(centers), self.heads, self.head_dim)
        head_logits = torch.einsum("nhd,chd->nhc", query, keys) / math.sqrt(self.head_dim)
        return head_logits.sum(1) * math.sqrt(self.head_dim)


def fit_gzsl_attention_decoder(
    features: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor,
    attributes: torch.Tensor, *, steps: int = 80, seed: int = 0,
) -> tuple[GZSLAttentionDecoder, torch.Tensor, torch.Tensor]:
    mean = attributes.float().mean(0)
    scale = attributes.float().std(0, unbiased=False).clamp_min(1e-4)
    scaled = (attributes.float() - mean) / scale
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = GZSLAttentionDecoder(attributes.shape[1], centers.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        counts = torch.bincount(labels, minlength=len(centers)).float().clamp_min(1)
        class_weights = counts.mean() / counts
        for _ in range(steps):
            loss = F.cross_entropy(model(features, centers, scaled), labels, weight=class_weights)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model.eval(), mean, scale


def _scaled_attributes(support: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = support.float().mean(0)
    scale = support.float().std(0, unbiased=False).clamp_min(1e-4)
    return (support.float() - mean) / scale, (target.float() - mean) / scale


def predict_attention_center(
    attributes: torch.Tensor, target: torch.Tensor, centers: torch.Tensor,
    *, steps: int = 150, seed: int = 0,
) -> tuple[torch.Tensor, dict]:
    if len(attributes) < 3:
        raise ValueError("attention prototype requires at least three support classes")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = RelationalPrototypeAttention(attributes.shape[1], centers.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-3)
        for _ in range(steps):
            losses = []
            for held_out in range(len(attributes)):
                keep = torch.arange(len(attributes)) != held_out
                support, query = _scaled_attributes(attributes[keep], attributes[held_out])
                predicted, _ = model(support, query, centers[keep])
                losses.append(1 - F.cosine_similarity(predicted[None], centers[held_out][None]).mean())
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        support, query = _scaled_attributes(attributes, target)
        with torch.no_grad():
            center, weights = model(support, query, centers)
    return center, {"attention": weights.tolist(), "loco_loss": float(loss.detach())}
