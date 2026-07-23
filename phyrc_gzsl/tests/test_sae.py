import unittest
from pathlib import Path

import torch

from phyrc_gzsl.baseline.sae import fit_sae, sae_scores
from phyrc_gzsl.baseline.evaluate_sae import evaluate_split, select_sae_parameters


class SAETests(unittest.TestCase):
    def test_sylvester_solution_and_classification(self):
        features = torch.tensor([
            [2.0, 0.0], [1.0, 0.0],
            [0.0, 2.0], [0.0, 1.0],
        ])
        semantics = torch.tensor([
            [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0],
        ])
        regularization = 0.1

        weight = fit_sae(features, semantics, regularization)
        residual = (
            semantics.T @ semantics @ weight
            + regularization * weight @ (features.T @ features)
            - (1.0 + regularization) * semantics.T @ features
        )

        self.assertLess(float(residual.norm()), 1e-5)
        self.assertTrue(torch.equal(
            sae_scores(features, weight, torch.eye(2)).argmax(1),
            torch.tensor([0, 0, 1, 1]),
        ))

    def test_loco_selection_uses_declared_grid(self):
        features = torch.tensor([
            [2.0, 0.0], [1.0, 0.0],
            [0.0, 2.0], [0.0, 1.0],
            [-2.0, -2.0], [-1.0, -1.0],
        ])
        labels = torch.tensor([1, 1, 2, 2, 3, 3])

        selected = select_sae_parameters(
            features, labels, [1, 2, 3], torch.eye(3, 2),
            regularizations=(0.1, 1.0), biases=(0.0, 0.5),
        )

        self.assertIn(selected["regularization"], (0.1, 1.0))
        self.assertIn(selected["bias"], (0.0, 0.5))

    def test_evaluation_rejects_missing_backbone(self):
        with self.assertRaisesRegex(FileNotFoundError, "Backbone checkpoint"):
            evaluate_split(
                Path("missing-config"), Path("missing-backbone"),
                Path("missing-attributes"), Path("."), 1, 42,
            )


if __name__ == "__main__":
    unittest.main()
