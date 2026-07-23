import unittest
from pathlib import Path

import torch

from phyrc_gzsl.baseline.ce_gzsl import synthesize_features, train_ce_gzsl
from phyrc_gzsl.baseline.evaluate_ce_gzsl import evaluate_split


class CEGZSLTests(unittest.TestCase):
    def test_evaluation_rejects_missing_backbone(self):
        with self.assertRaisesRegex(FileNotFoundError, "Backbone checkpoint"):
            evaluate_split(
                Path("missing-config"), Path("missing-backbone"),
                Path("missing-attributes"), Path("."), 1, 42,
            )

    def test_seen_only_training_generates_joint_classifier(self):
        features = torch.tensor([
            [2.0, 0.1, 0.0, 0.0], [1.8, -0.1, 0.1, 0.0],
            [0.0, 2.0, 0.1, 0.0], [0.1, 1.8, -0.1, 0.0],
        ])
        labels = torch.tensor([0, 0, 1, 1])
        attributes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])

        model, classifier = train_ce_gzsl(
            features, labels, attributes[:2], attributes,
            seed=7, gan_epochs=1, classifier_epochs=2, batch_size=4,
            synthetic_per_class=3, noise_dim=2, hidden_dim=8,
            embedding_dim=6, projection_dim=3, critic_steps=1, device="cpu",
        )
        synthetic, synthetic_labels = synthesize_features(
            model, attributes[2:], 3, seed=9,
        )

        self.assertEqual(tuple(synthetic.shape), (3, 4))
        self.assertTrue(torch.equal(synthetic_labels, torch.zeros(3, dtype=torch.long)))
        self.assertEqual(tuple(classifier(model.embed(synthetic)).shape), (3, 3))
        self.assertTrue(torch.isfinite(synthetic).all())


if __name__ == "__main__":
    unittest.main()
