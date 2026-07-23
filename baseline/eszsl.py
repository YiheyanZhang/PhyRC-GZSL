"""Linear ESZSL baseline.

Reference: https://github.com/foxtrotmike/ESZSL
Commit: 6d17f8a75f532aa98136cd7e09385fc95d6922f6
"""

from __future__ import annotations

import torch


def fit_eszsl(
    features: torch.Tensor,
    targets: torch.Tensor,
    seen_attributes: torch.Tensor,
    feature_regularization: float,
    semantic_regularization: float,
) -> torch.Tensor:
    """Return the feature-to-semantic compatibility matrix."""
    features, targets, seen_attributes = (
        value.float() for value in (features, targets, seen_attributes)
    )
    feature_system = features.T @ features + feature_regularization * torch.eye(
        features.shape[1], device=features.device,
    )
    semantic_system = seen_attributes.T @ seen_attributes + semantic_regularization * torch.eye(
        seen_attributes.shape[1], device=seen_attributes.device,
    )
    left = torch.linalg.solve(feature_system, features.T @ targets @ seen_attributes)
    return torch.linalg.solve(semantic_system, left.T).T


def eszsl_scores(
    features: torch.Tensor,
    weight: torch.Tensor,
    class_attributes: torch.Tensor,
) -> torch.Tensor:
    """Score every feature against every class attribute vector."""
    return features.float() @ weight.float() @ class_attributes.float().T


def calibrate_seen_scores(
    scores: torch.Tensor,
    seen_indices: list[int],
    bias: float,
) -> torch.Tensor:
    """Apply calibrated stacking without mutating the input scores."""
    calibrated = scores.clone()
    calibrated[:, seen_indices] -= bias
    return calibrated
