from __future__ import annotations

import numpy as np


def per_class_accuracy(preds, targets, class_indices):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    result = {}
    for class_index in class_indices:
        mask = targets == class_index
        result[int(class_index)] = float((preds[mask] == targets[mask]).mean() * 100.0) if mask.any() else 0.0
    return result


def h_score(acc_seen: float, acc_unseen: float) -> float:
    if acc_seen <= 0.0 or acc_unseen <= 0.0:
        return 0.0
    return 2.0 * acc_seen * acc_unseen / (acc_seen + acc_unseen)


def overall_accuracy(preds, targets) -> float:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return float((preds == targets).mean() * 100.0) if targets.size else 0.0


def cohen_kappa(preds, targets, class_indices) -> float:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    total = int(targets.size)
    if total == 0:
        return 0.0
    observed = float((preds == targets).sum() / total)
    expected = 0.0
    for class_index in class_indices:
        pred_count = float((preds == class_index).sum())
        target_count = float((targets == class_index).sum())
        expected += pred_count * target_count
    expected /= float(total * total)
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def gzsl_metrics(preds, targets, seen_indices, unseen_indices):
    class_indices = list(seen_indices) + list(unseen_indices)
    per_class = per_class_accuracy(preds, targets, class_indices)
    seen_values = [per_class[int(i)] for i in seen_indices]
    unseen_values = [per_class[int(i)] for i in unseen_indices]
    seen_aa = float(np.mean(seen_values)) if seen_values else 0.0
    unseen_aa = float(np.mean(unseen_values)) if unseen_values else 0.0
    aa = float(np.mean([per_class[int(i)] for i in class_indices])) if class_indices else 0.0
    h = h_score(seen_aa, unseen_aa)
    return {
        "OA": overall_accuracy(preds, targets),
        "AA": aa,
        "Kappa": cohen_kappa(preds, targets, class_indices),
        "Seen_AA": seen_aa,
        "Unseen_AA": unseen_aa,
        "H": h,
        "Acc_Seen": seen_aa,
        "Acc_Unseen": unseen_aa,
        "H-score": h,
        "per_class": per_class,
    }


def format_per_class(per_class: dict, index_to_label: dict[int, int], class_names: dict[int, str]) -> dict:
    formatted = {}
    for internal_index, accuracy in per_class.items():
        original_label = int(index_to_label[int(internal_index)])
        class_name = class_names.get(original_label, str(original_label))
        formatted[f"{original_label}:{class_name}"] = float(accuracy)
    return formatted
