from __future__ import annotations

import argparse, contextlib, io, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.evaluate_eszsl import _ATTRIBUTE_KEYS
from baseline.genzsl_strict import train_genzsl
from diagnose_rstd_ot import resolve_project_path
from models.backbone import build_backbone
from utils.config import class_id_lists, load_config, resolve_device, set_seed, set_single_unseen_class
from utils.data_loader import flatten_hsi, load_paviac_from_config, stratified_train_mask
from utils.metrics import gzsl_metrics


OFFICIAL_COMMIT = "5968afc81b69da08432add85e834b717525c7d8e"
STRICT_GENZSL = {
    "epochs": 150, "loops": 3, "classifier_epochs": 20, "batch_size": 64,
    "synthetic_per_class": 800, "hidden_dim": 2048, "top_k": 2,
    "weights": (.8, .2), "ridge": 1e-3, "alpha_contrast": .1,
    "alpha_reconstruction": 1., "temperature": .07, "use_svd": True,
}
STRICT_PROTOCOL = {
    "strict_inductive": True,
    "model_selection": "fixed_preregistered_budget",
    "test_evaluations_per_split": 1,
    "unseen_visual_data_used_for_training": False,
}


def evaluate_split(config_path: Path, checkpoint: Path, attribute_path: Path, root: Path,
                   unseen_class: int, seed: int, **settings) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint}")
    config = load_config(config_path)
    set_single_unseen_class(config, unseen_class)
    config.setdefault("runtime", {})["seed"] = seed
    config.setdefault("data_split", {})["seed"] = seed
    config["model"].update({"backbone": "spectral_morphology", "pretrained_backbone": str(checkpoint), "freeze_backbone": True})
    set_seed(seed)
    seen, unseen, all_classes = class_id_lists(config)
    x, gt = load_paviac_from_config(config, root)
    spectra, labels = flatten_hsi(x, gt, ignore_background=True)
    train_mask = stratified_train_mask(labels, seen, float(config["data_split"]["train_ratio"]), seed)
    device = resolve_device(config["runtime"].get("device", "auto"))
    with contextlib.redirect_stdout(io.StringIO()):
        backbone = build_backbone(x.shape[-1], config).to(device).eval()
    with torch.no_grad():
        features = torch.cat([backbone(batch.to(device)).cpu() for batch in torch.from_numpy(spectra.astype(np.float32)).split(512)])
    features = F.normalize(features.float(), dim=1)
    seen_lookup = {class_id: index for index, class_id in enumerate(seen)}
    train_labels = torch.tensor([seen_lookup[int(label)] for label in labels[train_mask]])
    payload = json.loads(attribute_path.read_text(encoding="utf-8"))
    attributes = F.normalize(torch.tensor([[payload[str(cls)][key] for key in _ATTRIBUTE_KEYS] for cls in all_classes], dtype=torch.float32), dim=1)
    classifier = train_genzsl(features[torch.from_numpy(train_mask)], train_labels, attributes, len(seen), seed=seed, device=device, **settings)
    test_mask = (~train_mask) & np.isin(labels, all_classes)
    lookup = {class_id: index for index, class_id in enumerate(all_classes)}
    targets = np.array([lookup[int(label)] for label in labels[test_mask]])
    with torch.no_grad():
        predictions = classifier(features[torch.from_numpy(test_mask)].to(device)).argmax(1).cpu().numpy()
    return {"unseen": unseen[0], "metrics": gzsl_metrics(predictions, targets, list(range(len(seen))), [len(seen)])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--attributes", required=True)
    parser.add_argument("--backbone-pattern", required=True); parser.add_argument("--unseen-classes", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    rows = [evaluate_split(resolve_project_path(root, args.config), resolve_project_path(root, args.backbone_pattern.format(unseen=cls)), resolve_project_path(root, args.attributes), root, cls, args.seed, **STRICT_GENZSL) for cls in args.unseen_classes]
    keys = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
    result = {"summary": {key: float(np.mean([row["metrics"][key] for row in rows])) for key in keys}, "results": rows, "protocol": STRICT_PROTOCOL, "hyperparameters": STRICT_GENZSL, "official_commit": OFFICIAL_COMMIT}
    output = resolve_project_path(root, args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result["summary"]))


if __name__ == "__main__": main()
