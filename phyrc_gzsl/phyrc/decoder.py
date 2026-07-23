from __future__ import annotations

import torch
import torch.nn.functional as F


def joint_risk_scores(
    cosine_scores: torch.Tensor,
    seen_indices: list[int],
    unseen_indices: list[int],
    p_seen: torch.Tensor,
    p_unseen: torch.Tensor,
    temperature: float,
    risk_weight: float,
    unseen_prior: float,
) -> torch.Tensor:
    if cosine_scores.ndim != 2 or len(p_seen) != len(cosine_scores) or len(p_unseen) != len(cosine_scores):
        raise ValueError("scores and domain evidence must align")
    adjusted = cosine_scores.float() / max(float(temperature), 1e-6)
    if float(risk_weight) == 0.0 and float(unseen_prior) == 0.0 and float(temperature) == 1.0:
        return adjusted
    eps = 1e-6
    adjusted = adjusted.clone()
    adjusted[:, seen_indices] += float(risk_weight) * torch.log(
        p_seen.float().clamp_min(eps)
    )[:, None]
    adjusted[:, unseen_indices] += (
        float(risk_weight) * torch.log(p_unseen.float().clamp_min(eps))[:, None]
        + float(unseen_prior)
    )
    return adjusted


def learned_joint_scores(
    cosine_scores: torch.Tensor,
    seen_indices: list[int],
    unseen_indices: list[int],
    p_seen: torch.Tensor,
    p_unseen: torch.Tensor,
    alpha,
    delta,
) -> torch.Tensor:
    adjusted = cosine_scores.float().clone()
    alpha = torch.as_tensor(alpha, dtype=adjusted.dtype, device=adjusted.device)
    delta = torch.as_tensor(delta, dtype=adjusted.dtype, device=adjusted.device)
    adjusted[:, seen_indices] += alpha * torch.log(p_seen.float().clamp_min(1e-6))[:, None]
    adjusted[:, unseen_indices] += alpha * torch.log(p_unseen.float().clamp_min(1e-6))[:, None] + delta
    return adjusted


def adaptive_joint_scores(
    cosine_scores: torch.Tensor,
    seen_indices: list[int],
    unseen_indices: list[int],
    p_seen: torch.Tensor,
    p_unseen: torch.Tensor,
    *,
    a,
    b,
    c,
    d,
) -> torch.Tensor:
    scores = cosine_scores.float()
    evidence = torch.log(p_unseen.float().clamp_min(1e-6)) - torch.log(p_seen.float().clamp_min(1e-6))
    margin = scores[:, unseen_indices].max(1).values - scores[:, seen_indices].max(1).values
    top = scores.topk(min(2, scores.shape[1]), dim=1).values
    uncertainty = 1 - (top[:, 0] - top[:, -1]).clamp(0, 1)
    offset = (
        torch.as_tensor(a) * evidence + torch.as_tensor(b) * margin
        + torch.as_tensor(c) * uncertainty + torch.as_tensor(d)
    )
    adjusted = scores.clone()
    adjusted[:, seen_indices] -= 0.5 * offset[:, None]
    adjusted[:, unseen_indices] += 0.5 * offset[:, None]
    return adjusted


def learn_nested_adaptive_parameters(
    episodes: list[dict], p_seen: torch.Tensor, p_unseen: torch.Tensor,
    reference: dict, *, steps: int = 100, learning_rate: float = 0.05,
) -> dict:
    offsets = torch.tensor([0] + [len(e["labels"]) for e in episodes]).cumsum(0)
    alpha = float(reference.get("temperature", 1.0) * reference.get("risk_weight", 0.1))
    prior = float(reference.get("temperature", 1.0) * reference.get("unseen_prior", 0.0))
    folds = []
    for held_out in range(len(episodes)):
        raw_a = torch.logit(torch.tensor(min(max(alpha / 0.8, 1e-4), 1 - 1e-4))).requires_grad_()
        raw_b = torch.tensor(-2.9444, requires_grad=True)
        raw_c = torch.tensor(0.0, requires_grad=True)
        raw_d = torch.tensor(float(torch.atanh(torch.tensor(min(max(prior / 0.5, -0.99), 0.99)))), requires_grad=True)
        optimizer = torch.optim.Adam((raw_a, raw_b, raw_c, raw_d), lr=learning_rate)
        dual = torch.full((len(episodes) - 1, 3), 10.0)
        for step in range(steps):
            a, b = 0.8 * torch.sigmoid(raw_a), torch.sigmoid(raw_b)
            c, d = 0.5 * torch.tanh(raw_c), 0.5 * torch.tanh(raw_d)
            objectives, constraints = [], []
            for index, episode in enumerate(episodes):
                if index == held_out: continue
                start, end = int(offsets[index]), int(offsets[index + 1])
                unseen = episode["scores"].shape[1] - 1
                adjusted = adaptive_joint_scores(
                    episode["scores"], list(range(unseen)), [unseen],
                    p_seen[start:end], p_unseen[start:end], a=a, b=b, c=c, d=d,
                )
                reference_scores = learned_joint_scores(
                    episode["scores"], list(range(unseen)), [unseen],
                    p_seen[start:end], p_unseen[start:end], alpha, prior,
                )
                temperature = 0.2 * (0.25 ** (step / max(steps - 1, 1)))
                values = []
                for candidate in (adjusted, reference_scores):
                    probability = F.softmax(candidate / temperature, dim=1)
                    correct = probability.gather(1, episode["labels"][:, None]).flatten()
                    per_class = torch.stack([correct[episode["labels"] == cls].mean() for cls in range(unseen + 1)])
                    seen, unseen_value = per_class[:-1].mean(), per_class[-1]
                    values.append((correct.mean(), per_class.mean(), seen, unseen_value, 2 * seen * unseen_value / (seen + unseen_value + 1e-6)))
                current, baseline = values
                objectives.append(current[4] + 0.1 * current[3])
                constraints.append(torch.stack([
                    baseline[0] - 0.001 - current[0], baseline[1] - 0.001 - current[1], baseline[2] - 0.001 - current[2],
                ]))
            objectives, constraints = torch.stack(objectives), torch.stack(constraints)
            loss = -objectives.mean() + (dual * F.relu(constraints)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            dual = (dual + 0.5 * constraints.detach()).clamp(0, 100)
        folds.append(tuple(float(value.detach()) for value in (
            0.8 * torch.sigmoid(raw_a), torch.sigmoid(raw_b),
            0.5 * torch.tanh(raw_c), 0.5 * torch.tanh(raw_d),
        )))
    means = [sum(row[i] for row in folds) / len(folds) for i in range(4)]
    return {"a": means[0], "b": means[1], "c": means[2], "d": means[3], "fold_parameters": folds}


def learn_nested_risk_parameters(
    episodes: list[dict],
    p_seen: torch.Tensor,
    p_unseen: torch.Tensor,
    *,
    reference: dict | None = None,
    steps: int = 100,
    learning_rate: float = 0.05,
    alpha_max: float = 0.6,
    delta_max: float = 0.5,
) -> dict:
    if len(episodes) < 2:
        raise ValueError("nested risk learning requires at least two episodes")
    offsets = torch.tensor([0] + [len(e["labels"]) for e in episodes]).cumsum(0)
    fold_parameters = []
    for held_out in range(len(episodes)):
        raw_alpha = torch.tensor(-1.6094, requires_grad=True)
        raw_delta = torch.tensor(0.2027, requires_grad=True)
        optimizer = torch.optim.Adam((raw_alpha, raw_delta), lr=learning_rate)
        dual_oa = torch.tensor(10.0)
        dual_seen = torch.tensor(10.0)
        dual_episode = torch.full((len(episodes) - 1,), 5.0)
        for step in range(steps):
            alpha = float(alpha_max) * torch.sigmoid(raw_alpha)
            delta = float(delta_max) * torch.tanh(raw_delta)
            harmonic, soft_oa, soft_seen = [], [], []
            baseline_oa, baseline_seen = [], []
            for index, episode in enumerate(episodes):
                if index == held_out:
                    continue
                start, end = int(offsets[index]), int(offsets[index + 1])
                unseen_index = episode["scores"].shape[1] - 1
                adjusted = learned_joint_scores(
                    episode["scores"], list(range(unseen_index)), [unseen_index],
                    p_seen[start:end], p_unseen[start:end], alpha, delta,
                )
                temperature = 0.2 * (0.25 ** (step / max(steps - 1, 1)))
                probabilities = F.softmax(adjusted / temperature, dim=1)
                correct = probabilities.gather(1, episode["labels"][:, None]).flatten()
                unseen_mask = episode["labels"] == unseen_index
                seen_value = torch.stack([
                    correct[episode["labels"] == cls].mean() for cls in range(unseen_index)
                ]).mean()
                unseen_value = correct[unseen_mask].mean()
                harmonic.append(2 * seen_value * unseen_value / (seen_value + unseen_value + 1e-6))
                soft_oa.append(correct.mean()); soft_seen.append(seen_value)
                baseline_scores = episode["scores"]
                if reference is not None and reference.get("mode") == "rcjd":
                    reference_temperature = float(reference["temperature"])
                    baseline_scores = reference_temperature * joint_risk_scores(
                        episode["scores"], list(range(unseen_index)), [unseen_index],
                        p_seen[start:end], p_unseen[start:end], reference_temperature,
                        reference["risk_weight"], reference["unseen_prior"],
                    )
                baseline = F.softmax(baseline_scores / temperature, dim=1).gather(
                    1, episode["labels"][:, None],
                ).flatten()
                baseline_oa.append(baseline.mean())
                baseline_seen.append(torch.stack([
                    baseline[episode["labels"] == cls].mean() for cls in range(unseen_index)
                ]).mean())
            harmonic = torch.stack(harmonic)
            tail = torch.topk(1 - harmonic, max(1, (len(harmonic) + 2) // 3)).values.mean()
            soft_oa, soft_seen = torch.stack(soft_oa), torch.stack(soft_seen)
            baseline_oa, baseline_seen = torch.stack(baseline_oa), torch.stack(baseline_seen)
            g_oa = baseline_oa.mean() - 0.005 - soft_oa.mean()
            g_seen = baseline_seen.mean() - 0.01 - soft_seen.mean()
            g_episode = baseline_seen - 0.01 - soft_seen
            loss = (
                -harmonic.mean() - 0.25 * soft_oa.mean() + 0.25 * tail
                + dual_oa * F.relu(g_oa) + dual_seen * F.relu(g_seen)
                + (dual_episode * F.relu(g_episode)).mean()
            )
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            dual_oa = (dual_oa + 0.5 * g_oa.detach()).clamp(0, 100)
            dual_seen = (dual_seen + 0.5 * g_seen.detach()).clamp(0, 100)
            dual_episode = (dual_episode + 0.25 * g_episode.detach()).clamp(0, 100)
        fold_parameters.append((
            float(float(alpha_max) * torch.sigmoid(raw_alpha).detach()),
            float(float(delta_max) * torch.tanh(raw_delta).detach()),
        ))
    return {
        "alpha": float(sum(row[0] for row in fold_parameters) / len(fold_parameters)),
        "delta": float(sum(row[1] for row in fold_parameters) / len(fold_parameters)),
        "fold_parameters": fold_parameters,
    }


def select_decoder_candidate(
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
    return max(feasible, key=lambda row: (
        row.get("worst_h", row["H"]), row["H"], row["Unseen_AA"], row["OA"],
    ))
