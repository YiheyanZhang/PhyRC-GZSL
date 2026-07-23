import unittest
from pathlib import Path

import numpy as np
import torch

from baseline.free import synthesize_features, train_free
from baseline.evaluate_free import (
    STRICT_FREE, evaluate_split, validate_strict_training_labels,
)
from baseline.run_free_strict_5seed import checkpoint_pattern


class FREETests(unittest.TestCase):
    def test_strict_protocol_rejects_unseen_training_labels(self):
        with self.assertRaisesRegex(ValueError, "Unseen labels"):
            validate_strict_training_labels(np.array([1, 2, 9]), [1, 2], [9])

    def test_strict_preset_is_preregistered(self):
        self.assertEqual(STRICT_FREE, {
            "epochs": 100, "classifier_epochs": 25, "batch_size": 64,
            "synthetic_per_class": 100, "hidden_dim": 512, "critic_steps": 1,
        })

    def test_strict_checkpoint_pattern_matches_seed_layout(self):
        self.assertEqual(
            checkpoint_pattern("paviau", 42),
            "checkpoints/paviau_p1_backbone_s{unseen}.pt",
        )
        self.assertEqual(
            checkpoint_pattern("paviau", 43),
            "checkpoints/multiseed/paviau/seed43/"
            "paviau_p1_backbone_s{unseen}.pt",
        )

    def test_evaluation_rejects_missing_backbone(self):
        with self.assertRaisesRegex(FileNotFoundError, "Backbone checkpoint"):
            evaluate_split(
                Path("missing-config"), Path("missing-backbone"),
                Path("missing-attributes"), Path("."), 1, 42,
            )

    def test_seen_only_training_generates_joint_refined_classifier(self):
        features = torch.tensor([
            [2.0, 0.1, 0.0, 0.0], [1.8, -0.1, 0.1, 0.0],
            [0.0, 2.0, 0.1, 0.0], [0.1, 1.8, -0.1, 0.0],
        ])
        labels = torch.tensor([0, 0, 1, 1])
        attributes = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [0.7, 0.7],
        ])

        model, classifier = train_free(
            features, labels, attributes[:2], attributes,
            seed=7, epochs=1, classifier_epochs=2, batch_size=4,
            synthetic_per_class=3, hidden_dim=8, latent_dim=2,
            critic_steps=1, device="cpu",
        )
        synthetic, synthetic_labels = synthesize_features(
            model, attributes[2:], 3, seed=9,
        )
        refined = model.transform(synthetic)

        self.assertEqual(tuple(synthetic.shape), (3, 4))
        self.assertTrue(torch.equal(synthetic_labels, torch.zeros(3, dtype=torch.long)))
        self.assertEqual(tuple(refined.shape), (3, 14))
        self.assertEqual(tuple(classifier(refined).shape), (3, 3))
        self.assertTrue(torch.isfinite(refined).all())


if __name__ == "__main__":
    unittest.main()
