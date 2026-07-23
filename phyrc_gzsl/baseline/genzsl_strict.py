"""Strict-inductive adapter of GenZSL (Chen et al., ICML 2025)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def validate_seen_only(labels: torch.Tensor, seen_count: int) -> None:
    if labels.numel() and (labels.min() < 0 or labels.max() >= seen_count):
        raise ValueError("unseen labels are forbidden during GenZSL training")


def fit_semantic_projection(
    attributes: torch.Tensor, seen_centres: torch.Tensor, ridge: float = 1e-3,
) -> torch.Tensor:
    """Fit attribute->visual projection using Seen class centres only."""
    seen_attributes = attributes[: seen_centres.shape[0]].float()
    eye = torch.eye(seen_attributes.shape[1], device=seen_attributes.device)
    projection = torch.linalg.solve(
        seen_attributes.T @ seen_attributes + ridge * eye,
        seen_attributes.T @ seen_centres.float(),
    )
    return F.normalize(attributes.float() @ projection, dim=1)


def topk_seen_classes(semantics: torch.Tensor, seen_count: int, k: int = 2) -> torch.Tensor:
    if seen_count <= k:
        raise ValueError("top-k requires more Seen classes than k")
    scores = F.normalize(semantics, dim=1) @ F.normalize(semantics[:seen_count], dim=1).T
    rows = torch.arange(seen_count, device=scores.device)
    scores[rows, rows] = -torch.inf
    return scores.topk(k, dim=1).indices


class _Encoder(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU())
        self.mean, self.log_var = nn.Linear(hidden, dim), nn.Linear(hidden, dim)

    def forward(self, feature: torch.Tensor, semantic: torch.Tensor):
        hidden = self.body(torch.cat((feature, semantic), dim=1))
        mean, log_var = self.mean(hidden), self.log_var(hidden)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
        return z, mean, log_var


class _Decoder(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, z: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat((z, semantic), dim=1))


class _Classifier(nn.Module):
    def __init__(self, dim: int, classes: int):
        super().__init__()
        middle = max(1, dim // 2)
        self.body = nn.Sequential(
            nn.Linear(dim, middle), nn.ReLU(), nn.Dropout(.5),
            nn.Linear(middle, middle), nn.ReLU(), nn.Dropout(.5),
            nn.Linear(middle, classes),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.body(feature)


def _contrastive(image: torch.Tensor, text: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = torch.arange(image.shape[0], device=image.device)
    logits = image @ text.T / temperature
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def _mixed_features(
    labels: torch.Tensor, neighbours: torch.Tensor, pools: list[torch.Tensor],
    weights: tuple[float, ...], generator: torch.Generator,
) -> torch.Tensor:
    parts = []
    for rank, weight in enumerate(weights):
        samples = []
        for label in labels.tolist():
            pool = pools[int(neighbours[label, rank])]
            index = torch.randint(len(pool), (1,), generator=generator).item()
            samples.append(pool[index])
        parts.append(weight * torch.stack(samples))
    return sum(parts)


def train_genzsl(
    features: torch.Tensor, labels: torch.Tensor, attributes: torch.Tensor,
    seen_count: int, *, seed: int, epochs: int = 150, loops: int = 3,
    classifier_epochs: int = 20, batch_size: int = 64,
    synthetic_per_class: int = 800, hidden_dim: int = 2048,
    top_k: int = 2, weights: tuple[float, ...] = (.8, .2), ridge: float = 1e-3,
    alpha_contrast: float = .1, alpha_reconstruction: float = 1.,
    temperature: float = .07, use_svd: bool = True,
    device: torch.device | str = "cpu",
) -> nn.Module:
    validate_seen_only(labels, seen_count)
    if len(weights) != top_k:
        raise ValueError("one mixing weight is required per neighbour")
    torch.manual_seed(seed)
    device = torch.device(device)
    features, labels, attributes = features.float(), labels.long(), attributes.float()
    centres = torch.stack([features[labels == cls].mean(0) for cls in range(seen_count)])
    semantics = fit_semantic_projection(attributes, centres, ridge)
    if use_svd:
        u = torch.linalg.svd(semantics, full_matrices=True).U[:, 1:]
        semantics = F.normalize(u @ u.T @ semantics, dim=1)
    neighbours = topk_seen_classes(semantics, seen_count, top_k)
    pools = [features[labels == cls] for cls in range(seen_count)]
    dim, classes = features.shape[1], attributes.shape[0]
    encoder, decoder = _Encoder(dim, hidden_dim).to(device), _Decoder(dim, hidden_dim).to(device)
    optimiser = torch.optim.Adam((*encoder.parameters(), *decoder.parameters()), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for _ in range(loops):
            for start in range(0, len(features), batch_size):
                batch_labels = labels[start:start + batch_size]
                mixed = _mixed_features(batch_labels, neighbours, pools, weights, generator).to(device)
                semantic = semantics[batch_labels].to(device)
                z, mean, log_var = encoder(mixed, semantic)
                reconstruction = F.normalize(decoder(z, semantic), dim=1)
                recon = F.mse_loss(reconstruction, mixed, reduction="sum") / len(mixed)
                kl = -.5 * torch.mean(1 + log_var - mean.square() - log_var.exp())
                loss = alpha_reconstruction * (recon + kl) + alpha_contrast * _contrastive(
                    reconstruction, semantic, temperature,
                )
                optimiser.zero_grad(); loss.backward(); optimiser.step()
    decoder.eval()
    synth_features, synth_labels = [features], [labels]
    with torch.no_grad():
        for cls in range(seen_count, classes):
            semantic = semantics[cls].to(device).repeat(synthetic_per_class, 1)
            generated = F.normalize(decoder(torch.randn_like(semantic), semantic), dim=1).cpu()
            synth_features.append(generated)
            synth_labels.append(torch.full((synthetic_per_class,), cls))
    train_x, train_y = torch.cat(synth_features).to(device), torch.cat(synth_labels).to(device)
    classifier = _Classifier(dim, classes).to(device)
    optimiser = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    for _ in range(classifier_epochs):
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(train_x), batch_size):
            idx = order[start:start + batch_size]
            loss = F.cross_entropy(classifier(train_x[idx]), train_y[idx])
            optimiser.zero_grad(); loss.backward(); optimiser.step()
    return classifier.eval()
