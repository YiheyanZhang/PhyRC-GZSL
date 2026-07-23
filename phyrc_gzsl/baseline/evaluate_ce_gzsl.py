from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phyrc_gzsl.baseline.ce_gzsl import train_ce_gzsl
from phyrc_gzsl.baseline.evaluate_eszsl import _ATTRIBUTE_KEYS
from phyrc_gzsl.diagnose_rstd_ot import resolve_project_path
from phyrc_gzsl.models.backbone import build_backbone
from phyrc_gzsl.utils.config import class_id_lists, load_config, resolve_device, set_seed, set_single_unseen_class
from phyrc_gzsl.utils.data_loader import flatten_hsi, load_paviac_from_config, stratified_train_mask
from phyrc_gzsl.utils.metrics import gzsl_metrics


def evaluate_split(
    config_path: Path, backbone_checkpoint: Path, attribute_path: Path, root: Path,
    unseen_class: int, seed: int, *, gan_epochs: int = 100,
    classifier_epochs: int = 25, synthetic_per_class: int = 100,
    batch_size: int = 2048, hidden_dim: int = 4096,
    embedding_dim: int = 2048, projection_dim: int = 512,
    relation_hidden_dim: int = 2048,
) -> dict:
    if not backbone_checkpoint.is_file():
        raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_checkpoint}")
    config = load_config(config_path)
    set_single_unseen_class(config, unseen_class)
    config.setdefault("runtime", {})["seed"] = seed
    config.setdefault("data_split", {})["seed"] = seed
    config["model"].update({
        "backbone": "spectral_morphology", "pretrained_backbone": str(backbone_checkpoint),
        "freeze_backbone": True,
    })
    set_seed(seed)
    seen, unseen, all_classes = class_id_lists(config)
    x, gt = load_paviac_from_config(config, root)
    spectra, labels = flatten_hsi(x, gt, ignore_background=True)
    train_mask = stratified_train_mask(labels, seen, float(config["data_split"]["train_ratio"]), seed)
    device = resolve_device(config["runtime"].get("device", "auto"))
    with contextlib.redirect_stdout(io.StringIO()):
        backbone = build_backbone(x.shape[-1], config).to(device).eval()
    with torch.no_grad():
        features = torch.cat([
            backbone(batch.to(device)).cpu()
            for batch in torch.from_numpy(spectra.astype(np.float32)).split(512)
        ])
    seen_lookup = {class_id: index for index, class_id in enumerate(seen)}
    train_labels = torch.tensor([seen_lookup[int(label)] for label in labels[train_mask]])
    payload = json.loads(attribute_path.read_text(encoding="utf-8"))
    attributes = F.normalize(torch.tensor([
        [payload[str(class_id)][key] for key in _ATTRIBUTE_KEYS] for class_id in all_classes
    ], dtype=torch.float32), dim=1)
    model, classifier = train_ce_gzsl(
        features[torch.from_numpy(train_mask)], train_labels, attributes[:len(seen)], attributes,
        seed=seed, gan_epochs=gan_epochs, classifier_epochs=classifier_epochs,
        synthetic_per_class=synthetic_per_class, batch_size=batch_size,
        hidden_dim=hidden_dim, embedding_dim=embedding_dim,
        projection_dim=projection_dim, relation_hidden_dim=relation_hidden_dim,
        device=device,
    )
    test_mask = (~train_mask) & np.isin(labels, all_classes)
    lookup = {class_id: index for index, class_id in enumerate(all_classes)}
    targets = np.array([lookup[int(label)] for label in labels[test_mask]])
    classifier.eval()
    with torch.no_grad():
        predictions = classifier(
            model.embed(model.transform(features[torch.from_numpy(test_mask)])),
        ).argmax(1).cpu().numpy()
    return {"unseen": unseen[0], "metrics": gzsl_metrics(
        predictions, targets, list(range(len(seen))), [len(seen)],
    )}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--unseen-classes", nargs="+", type=int, required=True)
    parser.add_argument("--backbone-pattern", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gan-epochs", type=int, default=100)
    parser.add_argument("--classifier-epochs", type=int, default=25)
    parser.add_argument("--synthetic-per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--relation-hidden-dim", type=int, default=2048)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    rows = [evaluate_split(
        resolve_project_path(root, args.config),
        resolve_project_path(root, args.backbone_pattern.format(unseen=unseen)),
        resolve_project_path(root, args.attributes), root, unseen, args.seed,
        gan_epochs=args.gan_epochs, classifier_epochs=args.classifier_epochs,
        synthetic_per_class=args.synthetic_per_class, batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        relation_hidden_dim=args.relation_hidden_dim,
    ) for unseen in args.unseen_classes]
    keys = ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
    summary = {key: float(np.mean([row["metrics"][key] for row in rows])) for key in keys}
    output = resolve_project_path(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
