import unittest

import torch

from phyrc_gzsl.baseline.eszsl import calibrate_seen_scores, eszsl_scores, fit_eszsl
from phyrc_gzsl.baseline.evaluate_eszsl import choose_eszsl_candidate, select_eszsl_parameters


class ESZSLTests(unittest.TestCase):
    def test_closed_form_fits_separable_classes(self):
        features = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        attributes = torch.eye(2)

        weight = fit_eszsl(features, targets, attributes, 1e-3, 1e-3)
        predictions = eszsl_scores(features, weight, attributes).argmax(1)

        self.assertTrue(torch.equal(predictions, torch.tensor([0, 0, 1, 1])))

    def test_seen_bias_only_changes_seen_columns(self):
        scores = torch.tensor([[2.0, 3.0, 4.0]])

        calibrated = calibrate_seen_scores(scores, [0, 2], 0.5)

        self.assertTrue(torch.equal(calibrated, torch.tensor([[1.5, 3.0, 3.5]])))
        self.assertTrue(torch.equal(scores, torch.tensor([[2.0, 3.0, 4.0]])))

    def test_loco_selection_uses_declared_grid(self):
        features = torch.tensor([
            [2.0, 0.0], [1.0, 0.0],
            [0.0, 2.0], [0.0, 1.0],
            [-2.0, -2.0], [-1.0, -1.0],
        ])
        labels = torch.tensor([1, 1, 2, 2, 3, 3])

        selected = select_eszsl_parameters(
            features, labels, [1, 2, 3], torch.eye(3, 2),
            regularizations=(0.1, 1.0), biases=(0.0, 0.5),
        )

        self.assertIn(selected["feature_regularization"], (0.1, 1.0))
        self.assertIn(selected["semantic_regularization"], (0.1, 1.0))
        self.assertIn(selected["bias"], (0.0, 0.5))

    def test_loco_selection_prioritizes_h_over_seen_zero_count(self):
        conservative = {"H": 20.0, "Unseen_AA": 15.0, "OA": 80.0, "seen_zero": 0}
        balanced = {"H": 40.0, "Unseen_AA": 35.0, "OA": 70.0, "seen_zero": 1}

        self.assertIs(choose_eszsl_candidate([conservative, balanced]), balanced)


if __name__ == "__main__":
    unittest.main()
