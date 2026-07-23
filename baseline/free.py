"""Strict inductive FREE for frozen PhyRC-GZSL features.

Chen et al., ICCV 2021. Network/loss reference:
https://github.com/shiming-chen/FREE at commit 3578a6a.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    def __init__(self, feature_dim: int, attribute_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(feature_dim + attribute_dim, hidden_dim), nn.LeakyReLU(0.2),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, features: torch.Tensor, attributes: torch.Tensor):
        hidden = self.body(torch.cat((features, attributes), dim=1))
        return self.mu(hidden), self.logvar(hidden)


class _Generator(nn.Module):
    def __init__(self, feature_dim: int, attribute_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + attribute_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feature_dim), nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor, attributes: torch.Tensor):
        return self.net(torch.cat((latent, attributes), dim=1))


class _Critic(nn.Module):
    def __init__(self, feature_dim: int, attribute_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + attribute_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, attributes: torch.Tensor):
        return self.net(torch.cat((features, attributes), dim=1)).squeeze(1)


class _FeatureRefiner(nn.Module):
    def __init__(self, feature_dim: int, attribute_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU(0.2))
        self.statistics = nn.Linear(hidden_dim, attribute_dim * 2)
        self.attribute_dim = attribute_dim

    def forward(self, features: torch.Tensor):
        hidden = self.hidden(features)
        mu, raw_std = self.statistics(hidden).split(self.attribute_dim, dim=1)
        std = torch.sigmoid(raw_std)
        reconstruction = torch.sigmoid(mu + torch.randn_like(mu) * std)
        return hidden, mu, std, reconstruction

    def deterministic(self, features: torch.Tensor):
        hidden = self.hidden(features)
        mu = self.statistics(hidden)[:, :self.attribute_dim]
        return torch.cat((features, hidden, torch.sigmoid(mu)), dim=1)


class FREEModel(nn.Module):
    def __init__(
        self, encoder: _Encoder, generator: _Generator, refiner: _FeatureRefiner,
        feature_min: torch.Tensor, feature_range: torch.Tensor,
    ):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.refiner = refiner
        self.register_buffer("feature_min", feature_min)
        self.register_buffer("feature_range", feature_range)

    def scale(self, features: torch.Tensor):
        return ((features.float().to(self.feature_min.device) - self.feature_min) / self.feature_range).clamp(0, 1)

    def unscale(self, features: torch.Tensor):
        return features * self.feature_range + self.feature_min

    def transform(self, features: torch.Tensor):
        self.eval()
        with torch.no_grad():
            return self.refiner.deterministic(self.scale(features))


def _weighted_l1(prediction: torch.Tensor, target: torch.Tensor):
    weights = (prediction - target).square()
    weights = weights / weights.sum(1, keepdim=True).sqrt().clamp_min(1e-8)
    return (weights * (prediction - target).abs()).sum(1).mean()


def _triplet_center_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor,
    margin: float, incenter_weight: float,
):
    distances = torch.cdist(embeddings, centers).square()
    positive = distances.gather(1, labels[:, None]).squeeze(1)
    mask = F.one_hot(labels, len(centers)).bool()
    negative = distances.masked_fill(mask, float("inf")).min(1).values
    return F.relu(margin + incenter_weight * positive - (1 - incenter_weight) * negative).mean()


def _gradient_penalty(
    critic: _Critic, real: torch.Tensor, fake: torch.Tensor, attributes: torch.Tensor,
):
    alpha = torch.rand(len(real), 1, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    gradients = torch.autograd.grad(
        critic(mixed, attributes).sum(), mixed, create_graph=True,
    )[0]
    return (gradients.norm(2, dim=1).sub(1).square()).mean()


def _fit_classifier(features, labels, class_count, epochs, batch_size):
    classifier = nn.Linear(features.shape[1], class_count).to(features.device)
    counts = torch.bincount(labels, minlength=class_count).float().clamp_min(1)
    weights = counts.sum() / (class_count * counts)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    for _ in range(epochs):
        for indices in torch.randperm(len(features), device=features.device).split(batch_size):
            loss = F.cross_entropy(classifier(features[indices]), labels[indices], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return classifier


def synthesize_features(
    model: FREEModel, attributes: torch.Tensor, per_class: int, seed: int,
):
    device = model.feature_min.device
    attributes = F.normalize(attributes.float().to(device), dim=1)
    labels = torch.arange(len(attributes), device=device).repeat_interleave(per_class)
    repeated = attributes.repeat_interleave(per_class, dim=0)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        len(labels), model.generator.latent_dim, generator=generator, device=device,
    )
    model.eval()
    with torch.no_grad():
        scaled = model.generator(noise, repeated)
        return model.unscale(scaled), labels


def train_free(
    seen_features: torch.Tensor,
    seen_labels: torch.Tensor,
    seen_attributes: torch.Tensor,
    all_attributes: torch.Tensor,
    *,
    seed: int,
    epochs: int = 100,
    classifier_epochs: int = 25,
    batch_size: int = 64,
    synthetic_per_class: int = 100,
    hidden_dim: int = 512,
    latent_dim: int | None = None,
    critic_steps: int = 5,
    gradient_weight: float = 10.0,
    reconstruction_weight: float = 0.01,
    center_weight: float = 0.5,
    center_margin: float = 1.0,
    incenter_weight: float = 0.5,
    device: str | torch.device = "cpu",
):
    """Train FREE using Seen visual samples and all-class attributes only."""
    if seen_features.ndim != 2 or len(seen_features) != len(seen_labels):
        raise ValueError("seen_features and seen_labels must contain matching samples")
    if len(seen_attributes) < 2 or len(all_attributes) <= len(seen_attributes):
        raise ValueError("all_attributes must contain Seen followed by Unseen classes")
    if seen_labels.min() < 0 or seen_labels.max() >= len(seen_attributes):
        raise ValueError("seen_labels must be contiguous indices into seen_attributes")

    torch.manual_seed(seed)
    device = torch.device(device)
    raw_features = seen_features.float().to(device)
    feature_min = raw_features.min(0).values
    feature_range = (raw_features.max(0).values - feature_min).clamp_min(1e-6)
    features = ((raw_features - feature_min) / feature_range).clamp(0, 1)
    labels = seen_labels.long().to(device)
    seen_attributes = F.normalize(seen_attributes.float().to(device), dim=1)
    all_attributes = F.normalize(all_attributes.float().to(device), dim=1)
    paired_attributes = seen_attributes[labels]
    feature_dim, attribute_dim = features.shape[1], seen_attributes.shape[1]
    latent_dim = latent_dim or attribute_dim

    encoder = _Encoder(feature_dim, attribute_dim, hidden_dim, latent_dim).to(device)
    generator = _Generator(feature_dim, attribute_dim, hidden_dim, latent_dim).to(device)
    critic = _Critic(feature_dim, attribute_dim, hidden_dim).to(device)
    refiner = _FeatureRefiner(feature_dim, attribute_dim, hidden_dim).to(device)
    centers = nn.Parameter(torch.randn(len(seen_attributes), attribute_dim, device=device))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))
    generator_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()), lr=1e-4, betas=(0.5, 0.9),
    )
    refiner_optimizer = torch.optim.Adam(
        list(refiner.parameters()) + [centers], lr=1e-4, betas=(0.5, 0.9),
    )

    for _ in range(epochs):
        for indices in torch.randperm(len(features), device=device).split(batch_size):
            real, attributes, batch_labels = features[indices], paired_attributes[indices], labels[indices]
            for _ in range(critic_steps):
                fake = generator(torch.randn(len(indices), latent_dim, device=device), attributes)
                critic_loss = (
                    critic(fake.detach(), attributes).mean() - critic(real, attributes).mean()
                    + gradient_weight * _gradient_penalty(critic, real, fake.detach(), attributes)
                )
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            with torch.no_grad():
                detached_fake = generator(torch.randn(len(indices), latent_dim, device=device), attributes)
            _, real_mu, _, real_reconstruction = refiner(real)
            _, _, _, fake_reconstruction = refiner(detached_fake)
            refiner_loss = reconstruction_weight * (
                _weighted_l1(real_reconstruction, attributes)
                + _weighted_l1(fake_reconstruction, attributes)
            )
            refiner_loss += center_weight * _triplet_center_loss(
                real_mu, batch_labels, centers, center_margin, incenter_weight,
            )
            refiner_optimizer.zero_grad()
            refiner_loss.backward()
            refiner_optimizer.step()

            critic.requires_grad_(False)
            refiner.requires_grad_(False)
            mu, logvar = encoder(real, attributes)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            reconstruction = generator(latent, attributes)
            vae_loss = F.binary_cross_entropy(reconstruction, real, reduction="sum") / len(real)
            vae_loss += -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / len(real)
            fake = generator(torch.randn(len(indices), latent_dim, device=device), attributes)
            _, _, _, fake_attributes = refiner(fake)
            generator_loss = (
                vae_loss - critic(fake, attributes).mean()
                + reconstruction_weight * _weighted_l1(fake_attributes, attributes)
            )
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()
            critic.requires_grad_(True)
            refiner.requires_grad_(True)

    model = FREEModel(encoder, generator, refiner, feature_min, feature_range).to(device)
    unseen, unseen_labels = synthesize_features(
        model, all_attributes[len(seen_attributes):], synthetic_per_class, seed + 1,
    )
    joint_features = torch.cat((model.transform(raw_features), model.transform(unseen)))
    joint_labels = torch.cat((labels, unseen_labels + len(seen_attributes)))
    classifier = _fit_classifier(
        joint_features, joint_labels, len(all_attributes), classifier_epochs, batch_size,
    )
    return model, classifier
