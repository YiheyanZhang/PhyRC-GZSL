import unittest
from pathlib import Path

import torch


class PhyOTP0Tests(unittest.TestCase):
    def test_project_path_does_not_duplicate_pema(self):
        from phyrc_gzsl.diagnose_rstd_ot import resolve_project_path

        root = Path("D:/project")
        checkpoint = root / "phyrc_gzsl" / "checkpoints" / "model.pt"
        self.assertEqual(resolve_project_path(root, "phyrc_gzsl/checkpoints/model.pt"), checkpoint)
        self.assertEqual(resolve_project_path(root, "checkpoints/model.pt"), checkpoint)

    def test_calibration_predicts_target_without_target_measurement(self):
        from phyrc_gzsl.diagnose_rstd_ot import calibrate_target_attributes

        support_llm = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        support_empirical = 2.0 * support_llm + torch.tensor([1.0, -1.0])
        predicted = calibrate_target_attributes(
            support_llm, support_empirical, torch.tensor([0.25, 0.75]), ridge=1e-6,
        )

        self.assertTrue(torch.allclose(predicted, torch.tensor([1.5, 0.5]), atol=1e-3))

    def test_calibration_keeps_physical_slots_independent(self):
        from phyrc_gzsl.diagnose_rstd_ot import calibrate_target_attributes

        support_llm = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        support_empirical = 2.0 * support_llm
        predicted = calibrate_target_attributes(
            support_llm, support_empirical, torch.tensor([1.0, 10.0]), ridge=1e-6,
        )

        self.assertTrue(torch.allclose(predicted, torch.tensor([2.0, 20.0]), atol=1e-3))

    def test_relational_center_recovers_linear_attribute_shift(self):
        from phyrc_gzsl.diagnose_rstd_ot import predict_relational_center

        attributes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        transform = torch.tensor([[2.0, -1.0, 0.5], [0.5, 1.5, -2.0]])
        offset = torch.tensor([0.2, -0.3, 1.0])
        centers = attributes @ transform + offset
        target = torch.tensor([0.25, 0.75])

        predicted = predict_relational_center(target, attributes, centers, ridge=1e-6, tau=1.0)

        self.assertTrue(torch.allclose(predicted, target @ transform + offset, atol=1e-3))

    def test_relational_hypotheses_keep_each_source_transport(self):
        from phyrc_gzsl.diagnose_rstd_ot import predict_relational_hypotheses

        attributes = torch.tensor([[0.0], [1.0], [2.0]])
        centers = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        hypotheses = predict_relational_hypotheses(
            torch.tensor([3.0]), attributes, centers, ridge=1e-6,
        )

        self.assertEqual(tuple(hypotheses.shape), (3, 2))
        self.assertTrue(torch.allclose(hypotheses, torch.tensor([[3.0, 6.0]]).repeat(3, 1), atol=1e-3))

    def test_spectral_attributes_capture_brightness_slope_and_red_edge(self):
        from phyrc_gzsl.diagnose_rstd_ot import spectral_class_attributes

        wavelengths = torch.linspace(430.0, 860.0, 103)
        flat = torch.full((103,), 0.2)
        rising = torch.linspace(0.1, 0.8, 103)
        spectra = torch.stack([flat, rising])
        attributes = spectral_class_attributes(
            spectra, torch.tensor([1, 2]), [1, 2], wavelengths,
        )

        self.assertEqual(tuple(attributes.shape), (2, 8))
        self.assertGreater(attributes[1, 0].item(), attributes[0, 0].item())
        self.assertGreater(attributes[1, 1].item(), attributes[0, 1].item())
        self.assertGreater(attributes[1, 4].item(), attributes[0, 4].item())
        self.assertGreater(attributes[1, 5].item(), attributes[0, 5].item())

    def test_center_metrics_reward_a_correct_proxy_unseen_center(self):
        from phyrc_gzsl.diagnose_rstd_ot import center_prediction_metrics

        target = torch.tensor([[0.0, 1.0], [0.1, 0.9]])
        support = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        metrics = center_prediction_metrics(target, target.mean(0), support)

        self.assertAlmostEqual(metrics["center_cosine_error"], 0.0, places=6)
        self.assertEqual(metrics["ncm"], 100.0)

    def test_center_metrics_accept_multiple_unseen_hypotheses(self):
        from phyrc_gzsl.diagnose_rstd_ot import center_prediction_metrics

        target = torch.tensor([[0.0, 1.0], [0.1, 0.9]])
        support = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        hypotheses = torch.tensor([[1.0, 0.1], [0.0, 1.0]])
        metrics = center_prediction_metrics(target, hypotheses, support)

        self.assertEqual(metrics["ncm"], 100.0)

    def test_rank_attributes_are_invariant_to_monotonic_scale(self):
        from phyrc_gzsl.diagnose_rstd_ot import rank_relation_attributes

        support = torch.tensor([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0]])
        target = torch.tensor([2.5, 25.0])
        ranked_support, ranked_target = rank_relation_attributes(support, target)
        scaled_support, scaled_target = rank_relation_attributes(7.0 * support + 4.0, 7.0 * target + 4.0)

        self.assertTrue(torch.equal(ranked_support, scaled_support))
        self.assertTrue(torch.equal(ranked_target, scaled_target))

    def test_proxy_gzsl_metrics_report_seen_unseen_and_h(self):
        from phyrc_gzsl.diagnose_rstd_ot import proxy_gzsl_metrics

        support_centers = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        support_features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]])
        support_labels = torch.tensor([0, 0, 1, 1])
        target_features = torch.tensor([[0.0, 1.0], [0.1, 0.9]])
        metrics = proxy_gzsl_metrics(
            target_features, support_features, support_labels,
            torch.tensor([0.0, 1.0]), support_centers,
        )

        self.assertEqual(metrics["seen_aa"], 100.0)
        self.assertEqual(metrics["unseen_aa"], 100.0)
        self.assertEqual(metrics["h"], 100.0)

    def test_reliability_weights_downweight_unhelpful_attribute(self):
        from phyrc_gzsl.diagnose_rstd_ot import attribute_reliability_weights

        useful = torch.arange(6, dtype=torch.float32)
        attributes = torch.stack([useful, torch.tensor([0.0, 5.0, 1.0, 4.0, 2.0, 3.0])], dim=1)
        centers = torch.stack([useful, 2.0 * useful], dim=1)
        weights = attribute_reliability_weights(attributes, centers)

        self.assertGreater(weights[0].item(), weights[1].item())

    def test_loco_error_prefers_aligned_relation(self):
        from phyrc_gzsl.diagnose_rstd_ot import relational_loco_error

        centers = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        aligned = centers.clone()
        shuffled = aligned[torch.tensor([0, 2, 1, 3])]

        self.assertLess(
            relational_loco_error(aligned, centers, ridge=1e-3, tau=0.5),
            relational_loco_error(shuffled, centers, ridge=1e-3, tau=0.5),
        )


if __name__ == "__main__":
    unittest.main()
