from __future__ import annotations

import random
import ast
from pathlib import Path

import numpy as np
import torch


def _parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def _load_simple_yaml(path: Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, sep, value = raw_line.strip().partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        parsed_key = _parse_scalar(key)
        if value.strip() == "":
            child: dict = {}
            current[parsed_key] = child
            stack.append((indent, child))
        else:
            current[parsed_key] = _parse_scalar(value)
    return root


def project_root_from_file(file_path: str) -> Path:
    return Path(file_path).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except ModuleNotFoundError:
        return _load_simple_yaml(config_path)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_names_from_config(config: dict) -> dict[int, str]:
    return {int(key): str(value) for key, value in config["classes"]["names"].items()}


def class_id_lists(config: dict) -> tuple[list[int], list[int], list[int]]:
    seen = [int(value) for value in config["classes"]["seen_classes"]]
    unseen = [int(value) for value in config["classes"]["unseen_classes"]]
    all_classes = seen + unseen
    if len(all_classes) != len(set(all_classes)):
        raise ValueError("seen_classes and unseen_classes must be disjoint")
    return seen, unseen, all_classes


def set_unseen_classes(config: dict, unseen_classes) -> dict:
    class_ids = sorted(int(value) for value in config["classes"]["names"])
    unseen = [int(value) for value in unseen_classes]
    if not unseen:
        raise ValueError("unseen_classes must be non-empty")
    if len(unseen) != len(set(unseen)):
        raise ValueError("unseen_classes must be unique")
    unknown = [value for value in unseen if value not in class_ids]
    if unknown:
        raise ValueError(f"Unknown class id: {unknown[0]}")
    unseen = sorted(unseen)
    config["classes"]["seen_classes"] = [value for value in class_ids if value not in unseen]
    config["classes"]["unseen_classes"] = unseen
    return config


def set_single_unseen_class(config: dict, unseen_class: int) -> dict:
    return set_unseen_classes(config, [unseen_class])
