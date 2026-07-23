"""Compute-matched strict CE-GZSL for frozen PhyRC-GZSL features.

Han et al., CVPR 2021. Reference implementation commit: 7bf5358.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.02)
        nn.init.zeros_(module.bias)


class _Generator(nn.Module):
    def __init__(self, feature_dim: int, attribute_dim: int, noise_dim: int, hidden_dim: int):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(attribute_dim + noise_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feature_dim), nn.ReLU(),
        )
        self.apply(_init)

    def forward(self, noise: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((noise, attributes), 1))


class _Critic(nn.Module):
    def __init__(self, feature_dim: int, attribute_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + attribute_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init)

    def forward(self, features: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((features, attributes), 1)).squeeze(1)


class _Mapper(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int, projection_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, projection_dim)
        self.apply(_init)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = F.relu(self.fc1(features))
        return embedding, F.normalize(self.fc2(embedding), dim=1)


class _Relation(nn.Module):
    def __init__(self, embedding_dim: int, attribute_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + attribute_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(1)


class CEModel(nn.Module):
    def __init__(self, generator: _Generator, mapper: _Mapper, minimum: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.generator = generator
        self.mapper = mapper
        self.register_buffer("minimum", minimum)
        self.register_buffer("scale", scale)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features.float().to(self.minimum.device) - self.minimum) / self.scale

    def embed(self, scaled_features: torch.Tensor) -> torch.Tensor:
        return self.mapper(scaled_features.float().to(self.minimum.device))[0]


def _supervised_contrastive(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = features @ features.T / temperature
    logits = logits - logits.max(1, keepdim=True).values.detach()
    eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positives.any(1)
    if not valid.any():
        return features.sum() * 0.0
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~positives, 0).sum(1)[valid] / positives.sum(1)[valid]).mean()


def _relation_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    attributes: torch.Tensor,
    relation: _Relation,
    temperature: float,
) -> torch.Tensor:
    batch, classes = len(embedding), len(attributes)
    pairs = torch.cat((
        embedding[:, None, :].expand(batch, classes, -1),
        attributes[None, :, :].expand(batch, classes, -1),
    ), 2)
    return F.cross_entropy(relation(pairs.reshape(batch * classes, -1)).reshape(batch, classes) / temperature, labels)


def _gradient_penalty(
    critic: _Critic, real: torch.Tensor, fake: torch.Tensor, attributes: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(len(real), 1, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    gradients = torch.autograd.grad(critic(mixed, attributes).sum(), mixed, create_graph=True)[0]
    return 10.0 * (gradients.norm(2, dim=1) - 1).square().mean()


def synthesize_features(
    model: CEModel, attributes: torch.Tensor, per_class: int, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = model.minimum.device
    torch.manual_seed(seed)
    attributes = F.normalize(attributes.float().to(device), dim=1)
    repeated = attributes.repeat_interleave(per_class, 0)
    labels = torch.arange(len(attributes), device=device).repeat_interleave(per_class)
    noise = torch.randn(len(repeated), model.generator.noise_dim, device=device)
    model.eval()
    with torch.no_grad():
        return model.generator(noise, repeated), labels


def _fit_classifier(features: torch.Tensor, labels: torch.Tensor, classes: int, epochs: int) -> nn.Linear:
    classifier = nn.Linear(features.shape[1], classes).to(features.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    for _ in range(epochs):
        for indices in torch.randperm(len(features), device=features.device).split(256):
            loss = F.cross_entropy(classifier(features[indices]), labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return classifier


def train_ce_gzsl(
    seen_features: torch.Tensor,
    seen_labels: torch.Tensor,
    seen_attributes: torch.Tensor,
    all_attributes: torch.Tensor,
    *,
    seed: int,
    gan_epochs: int = 100,
    classifier_epochs: int = 25,
    batch_size: int = 2048,
    synthetic_per_class: int = 100,
    noise_dim: int | None = None,
    hidden_dim: int = 4096,
    embedding_dim: int = 2048,
    projection_dim: int = 512,
    relation_hidden_dim: int = 2048,
    critic_steps: int = 5,
    device: str | torch.device = "cpu",
) -> tuple[CEModel, nn.Linear]:
    if seen_features.ndim != 2 or len(seen_features) != len(seen_labels):
        raise ValueError("seen_features and seen_labels must contain matching samples")
    if len(seen_attributes) < 2 or len(all_attributes) <= len(seen_attributes):
        raise ValueError("all_attributes must contain Seen followed by Unseen classes")
    if seen_labels.min() < 0 or seen_labels.max() >= len(seen_attributes):
        raise ValueError("seen_labels must be contiguous indices into seen_attributes")

    torch.manual_seed(seed)
    device = torch.device(device)
    raw = seen_features.float().to(device)
    minimum = raw.min(0).values
    scale = (raw.max(0).values - minimum).clamp_min(1e-6)
    features = (raw - minimum) / scale
    labels = seen_labels.long().to(device)
    seen_attributes = F.normalize(seen_attributes.float().to(device), dim=1)
    all_attributes = F.normalize(all_attributes.float().to(device), dim=1)
    paired_attributes = seen_attributes[labels]
    noise_dim = noise_dim or seen_attributes.shape[1]
    generator = _Generator(features.shape[1], seen_attributes.shape[1], noise_dim, hidden_dim).to(device)
    critic = _Critic(features.shape[1], seen_attributes.shape[1], hidden_dim).to(device)
    mapper = _Mapper(features.shape[1], embedding_dim, projection_dim).to(device)
    relation = _Relation(embedding_dim, seen_attributes.shape[1], relation_hidden_dim).to(device)
    optimizer_d = torch.optim.Adam(
        list(critic.parameters()) + list(mapper.parameters()) + list(relation.parameters()),
        lr=1e-4, betas=(0.5, 0.999),
    )
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    for _ in range(gan_epochs):
        for _ in range(max(1, (len(features) + batch_size - 1) // batch_size)):
            for _ in range(critic_steps):
                indices = torch.randint(len(features), (min(batch_size, len(features)),), device=device)
                real, batch_labels = features[indices], labels[indices]
                attributes = paired_attributes[indices]
                fake = generator(torch.randn(len(indices), noise_dim, device=device), attributes)
                embedding, projection = mapper(real)
                loss_d = critic(fake.detach(), attributes).mean() - critic(real, attributes).mean()
                loss_d += _gradient_penalty(critic, real, fake.detach(), attributes)
                loss_d += _supervised_contrastive(projection, batch_labels, 0.1)
                loss_d += _relation_loss(embedding, batch_labels, seen_attributes, relation, 0.1)
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()

            for module in (critic, mapper, relation):
                module.requires_grad_(False)
            fake = generator(torch.randn(len(indices), noise_dim, device=device), attributes)
            fake_embedding, fake_projection = mapper(fake)
            with torch.no_grad():
                _, real_projection = mapper(real)
            loss_g = -critic(fake, attributes).mean()
            loss_g += 0.001 * _supervised_contrastive(
                torch.cat((fake_projection, real_projection)),
                torch.cat((batch_labels, batch_labels)), 0.1,
            )
            loss_g += 0.001 * _relation_loss(
                fake_embedding, batch_labels, seen_attributes, relation, 0.1,
            )
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            for module in (critic, mapper, relation):
                module.requires_grad_(True)

    model = CEModel(generator, mapper, minimum, scale).to(device).eval()
    unseen, unseen_labels = synthesize_features(
        model, all_attributes[len(seen_attributes):], synthetic_per_class, seed + 1,
    )
    with torch.no_grad():
        joint_embeddings = model.embed(torch.cat((features, unseen)))
    classifier = _fit_classifier(
        joint_embeddings,
        torch.cat((labels, unseen_labels + len(seen_attributes))),
        len(all_attributes), classifier_epochs,
    )
    return model, classifier
