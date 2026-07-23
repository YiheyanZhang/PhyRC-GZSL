"""Strict inductive CADA-VAE for frozen PhyRC-GZSL features.

Schonfeld et al., CVPR 2019. Architecture/loss reference:
https://github.com/edgarschnfld/CADA-VAE-PyTorch at commit 26f0085.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.mu(h), self.logvar(h)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AlignedVAE(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        attribute_dim: int,
        latent_dim: int,
        visual_hidden: tuple[int, int],
        attribute_hidden: tuple[int, int],
    ):
        super().__init__()
        self.visual_encoder = _Encoder(visual_dim, visual_hidden[0], latent_dim)
        self.attribute_encoder = _Encoder(attribute_dim, attribute_hidden[0], latent_dim)
        self.visual_decoder = _Decoder(latent_dim, visual_hidden[1], visual_dim)
        self.attribute_decoder = _Decoder(latent_dim, attribute_hidden[1], attribute_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode_visual(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual_encoder(x.float().to(next(self.parameters()).device))[0]


def _warmup(epoch: int, start: int, end: int, factor: float) -> float:
    return factor * min(max((epoch - start) / max(end - start, 1), 0.0), 1.0)


def _sample_rows_per_class(
    features: torch.Tensor, labels: torch.Tensor, class_count: int, per_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = []
    for class_id in range(class_count):
        available = torch.where(labels == class_id)[0]
        indices.append(available[torch.randint(len(available), (per_class,), device=labels.device)])
    selected = torch.cat(indices)
    return features[selected], labels[selected]


def synthesize_latents(
    model: AlignedVAE,
    attributes: torch.Tensor,
    per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    repeated = F.normalize(attributes.float().to(device), dim=1).repeat_interleave(per_class, 0)
    labels = torch.arange(len(attributes), device=device).repeat_interleave(per_class)
    model.eval()
    with torch.no_grad():
        mu, logvar = model.attribute_encoder(repeated)
        return model._sample(mu, logvar), labels


def _fit_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
    epochs: int,
    batch_size: int,
) -> nn.Linear:
    classifier = nn.Linear(features.shape[1], class_count).to(features.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    for _ in range(epochs):
        for indices in torch.randperm(len(features), device=features.device).split(batch_size):
            loss = F.cross_entropy(classifier(features[indices]), labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return classifier


def train_cada_vae(
    seen_features: torch.Tensor,
    seen_labels: torch.Tensor,
    seen_attributes: torch.Tensor,
    all_attributes: torch.Tensor,
    *,
    seed: int,
    vae_epochs: int = 100,
    classifier_epochs: int = 30,
    batch_size: int = 50,
    seen_per_class: int = 200,
    synthetic_per_class: int = 400,
    latent_dim: int = 64,
    hidden_dim: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[AlignedVAE, nn.Linear]:
    """Train with Seen visual samples and class attributes only."""
    if seen_features.ndim != 2 or len(seen_features) != len(seen_labels):
        raise ValueError("seen_features and seen_labels must contain matching samples")
    if len(seen_attributes) < 2 or len(all_attributes) <= len(seen_attributes):
        raise ValueError("all_attributes must contain Seen followed by Unseen classes")
    if seen_labels.min() < 0 or seen_labels.max() >= len(seen_attributes):
        raise ValueError("seen_labels must be contiguous indices into seen_attributes")

    torch.manual_seed(seed)
    device = torch.device(device)
    visual = F.normalize(seen_features.float().to(device), dim=1)
    labels = seen_labels.long().to(device)
    seen_attributes = F.normalize(seen_attributes.float().to(device), dim=1)
    all_attributes = F.normalize(all_attributes.float().to(device), dim=1)
    paired_attributes = seen_attributes[labels]
    visual_hidden = (hidden_dim, hidden_dim) if hidden_dim else (1560, 1660)
    attribute_hidden = (hidden_dim, hidden_dim) if hidden_dim else (1450, 665)
    model = AlignedVAE(
        visual.shape[1], seen_attributes.shape[1], latent_dim,
        visual_hidden, attribute_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4, amsgrad=True)

    model.train()
    for epoch in range(vae_epochs):
        order = torch.randperm(len(visual), device=device)
        for indices in order.split(batch_size):
            x, a = visual[indices], paired_attributes[indices]
            mu_x, lv_x = model.visual_encoder(x)
            mu_a, lv_a = model.attribute_encoder(a)
            z_x, z_a = model._sample(mu_x, lv_x), model._sample(mu_a, lv_a)
            reconstruction = F.l1_loss(model.visual_decoder(z_x), x, reduction="sum")
            reconstruction += F.l1_loss(model.attribute_decoder(z_a), a, reduction="sum")
            cross = F.l1_loss(model.visual_decoder(z_a), x, reduction="sum")
            cross += F.l1_loss(model.attribute_decoder(z_x), a, reduction="sum")
            kl = -0.5 * (
                (1 + lv_x - mu_x.square() - lv_x.exp()).sum()
                + (1 + lv_a - mu_a.square() - lv_a.exp()).sum()
            )
            distance = torch.sqrt(
                (mu_x - mu_a).square().sum(1)
                + (torch.exp(0.5 * lv_x) - torch.exp(0.5 * lv_a)).square().sum(1)
                + 1e-8
            ).sum()
            loss = reconstruction
            loss += _warmup(epoch, 0, 93, 0.25) * kl
            loss += _warmup(epoch, 21, 75, 2.37) * cross
            loss += _warmup(epoch, 6, 22, 8.13) * distance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    sampled_visual, sampled_labels = _sample_rows_per_class(
        visual, labels, len(seen_attributes), seen_per_class,
    )
    with torch.no_grad():
        mu, logvar = model.visual_encoder(sampled_visual)
        seen_latents = model._sample(mu, logvar)
    unseen_latents, unseen_labels = synthesize_latents(
        model, all_attributes[len(seen_attributes):], synthetic_per_class, seed + 1,
    )
    classifier = _fit_classifier(
        torch.cat((seen_latents, unseen_latents)),
        torch.cat((sampled_labels, unseen_labels + len(seen_attributes))),
        len(all_attributes), classifier_epochs, 32,
    )
    return model, classifier
