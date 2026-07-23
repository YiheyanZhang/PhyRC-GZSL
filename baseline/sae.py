"""Semantic Autoencoder baseline (Kodirov et al., CVPR 2017)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import solve_sylvester


def fit_sae(
    features: torch.Tensor,
    semantic_targets: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    """Solve the SAE objective and return the visual-to-semantic encoder."""
    x = features.detach().float().cpu().numpy().astype(np.float64)
    s = semantic_targets.detach().float().cpu().numpy().astype(np.float64)
    weight = solve_sylvester(
        s.T @ s,
        regularization * (x.T @ x),
        (1.0 + regularization) * (s.T @ x),
    )
    return torch.from_numpy(weight).to(features.device, dtype=torch.float32)


def sae_scores(
    features: torch.Tensor,
    weight: torch.Tensor,
    class_attributes: torch.Tensor,
) -> torch.Tensor:
    """Cosine scores between encoded features and class attributes."""
    projected = F.normalize(features.float() @ weight.float().T, dim=1)
    return projected @ F.normalize(class_attributes.float(), dim=1).T
