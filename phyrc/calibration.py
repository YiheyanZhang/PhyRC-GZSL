from __future__ import annotations

import torch

from evaluate_p1 import fit_seen_unseen_router, router_probability


def cross_fit_domain_scores(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    episode_ids: torch.Tensor,
) -> torch.Tensor:
    if len(inputs) != len(labels) or len(labels) != len(episode_ids):
        raise ValueError("inputs, labels, and episode ids must align")
    episodes = episode_ids.unique(sorted=True)
    if len(episodes) < 2:
        raise ValueError("cross-fitting requires at least two episodes")
    scores = torch.empty(len(labels), dtype=torch.float32)
    for episode in episodes:
        held_out = episode_ids == episode
        fit = ~held_out
        if labels[fit].unique().numel() != 2:
            raise ValueError("every cross-fit training fold must contain both domains")
        router = fit_seen_unseen_router(inputs[fit], labels[fit])
        scores[held_out] = router_probability(inputs[held_out], router)
    return scores


def fit_dual_calibration(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("domain scores and labels must be aligned vectors")
    seen = scores[labels <= 0.5].float().sort().values
    unseen = scores[labels > 0.5].float().sort().values
    if seen.numel() == 0 or unseen.numel() == 0:
        raise ValueError("dual calibration requires both domains")
    return {"seen": seen, "unseen": unseen}


def dual_domain_pvalues(
    scores: torch.Tensor,
    calibration: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = scores.float().contiguous()
    seen = calibration["seen"].float().sort().values
    unseen = calibration["unseen"].float().sort().values
    seen_rank = torch.searchsorted(seen, values, right=False)
    unseen_rank = torch.searchsorted(unseen, values, right=True)
    p_seen = (1 + seen.numel() - seen_rank).float() / (seen.numel() + 1)
    p_unseen = (1 + unseen_rank).float() / (unseen.numel() + 1)
    return p_seen, p_unseen


def calibrated_domain_probability(
    raw_probability: torch.Tensor,
    p_seen: torch.Tensor,
    p_unseen: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    eps = 1e-6
    raw = raw_probability.float().clamp(eps, 1.0 - eps)
    evidence = torch.log(p_unseen.float().clamp_min(eps)) - torch.log(
        p_seen.float().clamp_min(eps)
    )
    return torch.sigmoid(torch.logit(raw) + float(beta) * evidence)


def select_dual_candidate(
    candidates: list[dict],
    p1: dict,
    seen_tolerance: float = 1.0,
    oa_tolerance: float = 0.5,
) -> dict:
    feasible = [p1] + [
        candidate for candidate in candidates
        if candidate["seen_zero"] <= p1["seen_zero"]
        and candidate["Seen_AA"] >= p1["Seen_AA"] - float(seen_tolerance)
        and candidate["OA"] >= p1["OA"] - float(oa_tolerance)
        and candidate["H"] > p1["H"]
        and candidate["Unseen_AA"] > p1["Unseen_AA"]
    ]
    return max(feasible, key=lambda row: (row["H"], row["Unseen_AA"], row["OA"]))
