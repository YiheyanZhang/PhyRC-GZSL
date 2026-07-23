"""Strict inductive f-CLSWGAN for frozen phyrc_gzsl features.

Method: Xian et al., CVPR 2018. Network/loss reference:
https://github.com/mkara44/f-clswgan_pytorch at commit 641cd008.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, feature_dim: int, attribute_dim: int, noise_dim: int, hidden_dim: int):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(attribute_dim + noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, noise: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(torch.cat((noise, attributes), dim=1)), dim=1)


class Critic(nn.Module):
    def __init__(self, feature_dim: int, attribute_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + attribute_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((features, attributes), dim=1)).squeeze(1)


def synthesize_features(
    generator: Generator,
    attributes: torch.Tensor,
    per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate features without accepting any Unseen visual samples."""
    device = next(generator.parameters()).device
    attributes = attributes.float().to(device)
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    labels = torch.arange(len(attributes), device=device).repeat_interleave(per_class)
    repeated_attributes = attributes.repeat_interleave(per_class, dim=0)
    noise = torch.randn(
        len(labels), generator.noise_dim, generator=noise_generator, device=device,
    )
    generator.eval()
    with torch.no_grad():
        return generator(noise, repeated_attributes), labels


def _fit_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
    epochs: int,
    batch_size: int,
) -> nn.Linear:
    classifier = nn.Linear(features.shape[1], class_count).to(features.device)
    counts = torch.bincount(labels, minlength=class_count).float().clamp_min(1)
    weights = counts.sum() / (class_count * counts)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(features), device=features.device)
        for indices in order.split(batch_size):
            loss = F.cross_entropy(
                classifier(features[indices]), labels[indices], weight=weights,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return classifier


def _gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    attributes: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(len(real), 1, device=real.device)
    mixed = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    gradients = torch.autograd.grad(
        critic(mixed, attributes).sum(), mixed, create_graph=True,
    )[0]
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def train_f_clswgan(
    seen_features: torch.Tensor,
    seen_labels: torch.Tensor,
    seen_attributes: torch.Tensor,
    all_attributes: torch.Tensor,
    *,
    seed: int,
    gan_epochs: int = 100,
    classifier_epochs: int = 50,
    pretrain_epochs: int = 30,
    batch_size: int = 64,
    synthetic_per_class: int = 300,
    noise_dim: int = 128,
    hidden_dim: int = 4096,
    n_critic: int = 5,
    classification_weight: float = 0.01,
    gradient_weight: float = 10.0,
    device: str | torch.device = "cpu",
) -> tuple[Generator, nn.Linear]:
    """Train from Seen visual samples and class attributes only."""
    if seen_features.ndim != 2 or len(seen_features) != len(seen_labels):
        raise ValueError("seen_features and seen_labels must contain matching samples")
    if len(seen_attributes) < 2 or len(all_attributes) <= len(seen_attributes):
        raise ValueError("all_attributes must contain Seen followed by Unseen classes")
    if seen_labels.min() < 0 or seen_labels.max() >= len(seen_attributes):
        raise ValueError("seen_labels must be contiguous indices into seen_attributes")

    torch.manual_seed(seed)
    device = torch.device(device)
    features = F.normalize(seen_features.float().to(device), dim=1)
    labels = seen_labels.long().to(device)
    seen_attributes = F.normalize(seen_attributes.float().to(device), dim=1)
    all_attributes = F.normalize(all_attributes.float().to(device), dim=1)
    sample_attributes = seen_attributes[labels]

    seen_classifier = _fit_classifier(
        features, labels, len(seen_attributes), pretrain_epochs, batch_size,
    ).eval()
    seen_classifier.requires_grad_(False)
    generator = Generator(
        features.shape[1], seen_attributes.shape[1], noise_dim, hidden_dim,
    ).to(device)
    critic = Critic(features.shape[1], seen_attributes.shape[1], hidden_dim).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    step = 0
    for _ in range(gan_epochs):
        for indices in torch.randperm(len(features), device=device).split(batch_size):
            real = features[indices]
            attributes = sample_attributes[indices]
            batch_labels = labels[indices]
            noise = torch.randn(len(indices), noise_dim, device=device)
            fake = generator(noise, attributes)
            critic_loss = (
                critic(fake.detach(), attributes).mean()
                - critic(real, attributes).mean()
                + gradient_weight * _gradient_penalty(critic, real, fake.detach(), attributes)
            )
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if step % n_critic == 0:
                fake = generator(torch.randn(len(indices), noise_dim, device=device), attributes)
                generator_loss = (
                    -critic(fake, attributes).mean()
                    + classification_weight * F.cross_entropy(seen_classifier(fake), batch_labels)
                )
                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()
            step += 1

    unseen_features, unseen_labels = synthesize_features(
        generator, all_attributes[len(seen_attributes):], synthetic_per_class, seed + 1,
    )
    joint_features = torch.cat((features, unseen_features))
    joint_labels = torch.cat((labels, unseen_labels + len(seen_attributes)))
    classifier = _fit_classifier(
        joint_features, joint_labels, len(all_attributes), classifier_epochs, batch_size,
    )
    return generator, classifier
