import unittest
from pathlib import Path

import torch

from phyrc_gzsl.baseline.f_clswgan import synthesize_features, train_f_clswgan
from phyrc_gzsl.baseline.evaluate_f_clswgan import evaluate_split


class FCLSWGANTests(unittest.TestCase):
    def test_evaluation_rejects_missing_backbone(self):
        with self.assertRaisesRegex(FileNotFoundError, "Backbone checkpoint"):
            evaluate_split(
                Path("missing-config"), Path("missing-backbone"),
                Path("missing-attributes"), Path("."), 1, 42,
            )

    def test_seen_only_training_generates_joint_classifier(self):
        torch.manual_seed(7)
        features = torch.nn.functional.normalize(torch.tensor([
            [2.0, 0.1, 0.0, 0.0], [1.8, -0.1, 0.1, 0.0],
            [0.0, 2.0, 0.1, 0.0], [0.1, 1.8, -0.1, 0.0],
        ]), dim=1)
        labels = torch.tensor([0, 0, 1, 1])
        attributes = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [0.7, 0.7],
        ])

        generator, classifier = train_f_clswgan(
            features, labels, attributes[:2], attributes,
            seed=7, gan_epochs=1, classifier_epochs=2,
            pretrain_epochs=2, batch_size=4, synthetic_per_class=3,
            noise_dim=2, hidden_dim=8, n_critic=1, device="cpu",
        )
        synthetic, synthetic_labels = synthesize_features(
            generator, attributes[2:], 3, seed=9,
        )

        self.assertEqual(tuple(synthetic.shape), (3, 4))
        self.assertTrue(torch.equal(synthetic_labels, torch.zeros(3, dtype=torch.long)))
        self.assertEqual(tuple(classifier(synthetic).shape), (3, 3))
        self.assertTrue(torch.isfinite(synthetic).all())


if __name__ == "__main__":
    unittest.main()
