import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


class CFDCTests(unittest.TestCase):
    def test_ranked_centers_support_multiple_unseen_targets(self):
        from evaluate_p1 import predict_relation_variants
        from evaluate_phyrc import _predict_ranked_centers

        attributes = torch.tensor([[0.0, 0.0], [0.5, 1.0], [1.0, 0.2]])
        targets = torch.tensor([[0.2, 0.8], [0.9, 0.1]])
        centers = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        predicted = _predict_ranked_centers(attributes, targets, centers)

        self.assertEqual(tuple(predicted.shape), (2, 2))
        expected = predict_relation_variants(attributes, targets[0], centers)["ranked"]
        self.assertTrue(torch.allclose(predicted[0], expected))

    def test_backbone_partition_rejects_wrong_seen_classes(self):
        from evaluate_phyrc import _validate_backbone_partition

        with patch("evaluate_phyrc.torch.load", return_value={"seen_classes": [1, 2]}):
            with self.assertRaisesRegex(ValueError, "partition"):
                _validate_backbone_partition("backbone.pt", [1, 3])

    def test_physical_distribution_loco_scores_are_finite(self):
        from evaluate_p1 import build_loco_episodes

        features = torch.tensor([
            [1.0, 0.0], [0.9, 0.1],
            [0.0, 1.0], [0.1, 0.9],
            [-1.0, 0.0], [-0.9, 0.1],
            [0.0, -1.0], [0.1, -0.9],
        ])
        labels = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
        attributes = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        centers = torch.stack([features[labels == cls].mean(0) for cls in range(1, 5)])
        for method in ("physical_distribution", "hybrid_distribution"):
            with self.subTest(method=method):
                episodes = build_loco_episodes(
                    features, labels, [1, 2, 3, 4], attributes, centers,
                    method=method,
                )
                self.assertEqual(len(episodes), 4)
                self.assertEqual(tuple(episodes[0]["scores"].shape), (8, 4))
                self.assertTrue(torch.isfinite(episodes[0]["scores"]).all())

    def test_relational_prototype_attention_normalizes_each_head(self):
        from phyrc.attention import RelationalPrototypeAttention

        model = RelationalPrototypeAttention(attribute_dim=2, feature_dim=4, hidden_dim=8, heads=2)
        center, weights = model(
            torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            torch.tensor([0.8, 0.2]),
            torch.eye(3, 4),
        )

        self.assertEqual(center.shape, (4,))
        self.assertEqual(weights.shape, (2, 3))
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(2), atol=1e-6))

    def test_gzsl_attention_decoder_starts_from_prototype_matching(self):
        from phyrc.attention import GZSLAttentionDecoder

        decoder = GZSLAttentionDecoder(attribute_dim=2, feature_dim=4, heads=2)
        prototypes = torch.eye(3, 4)
        logits = decoder(prototypes, prototypes, torch.zeros(3, 2))

        self.assertEqual(logits.shape, (3, 3))
        self.assertTrue(torch.equal(logits.argmax(1), torch.arange(3)))

    def test_prototype_set_scores_use_best_candidate_per_sample(self):
        from phyrc.attention import prototype_set_scores

        scores = prototype_set_scores(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

        self.assertTrue(torch.allclose(scores, torch.ones(2), atol=1e-6))

    def test_relation_episode_weights_favor_matching_attribute(self):
        from evaluate_phyrc import _relation_episode_weights

        attributes = torch.tensor([[0.0], [1.0], [3.0]])
        weights = _relation_episode_weights(attributes, torch.tensor([1.1]))
        self.assertEqual(int(weights.argmax()), 1)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_binary_auroc_handles_ties_without_pairwise_matrix(self):
        from evaluate_phyrc import _binary_auroc

        score = _binary_auroc(
            torch.tensor([0.1, 0.2, 0.2, 0.9]),
            torch.tensor([0, 0, 1, 1]),
        )

        self.assertAlmostEqual(score, 0.875)

    def test_dual_pvalues_favor_the_matching_domain(self):
        from phyrc.calibration import dual_domain_pvalues, fit_dual_calibration

        calibration = fit_dual_calibration(
            torch.tensor([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
            torch.tensor([0, 0, 0, 1, 1, 1]),
        )
        p_seen, p_unseen = dual_domain_pvalues(torch.tensor([0.15, 0.85]), calibration)

        self.assertGreater(p_seen[0], p_unseen[0])
        self.assertGreater(p_unseen[1], p_seen[1])
        self.assertTrue(((p_seen > 0) & (p_seen <= 1)).all())
        self.assertTrue(((p_unseen > 0) & (p_unseen <= 1)).all())

    def test_cross_fitted_scores_are_deterministic(self):
        from phyrc.calibration import cross_fit_domain_scores

        inputs = torch.tensor([
            [1.0, 0.1], [0.9, 0.2], [0.1, 0.9], [0.2, 1.0],
            [1.1, 0.0], [0.8, 0.3], [0.0, 1.1], [0.3, 0.8],
        ])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0] * 2)
        episodes = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        first = cross_fit_domain_scores(inputs, labels, episodes)
        second = cross_fit_domain_scores(inputs, labels, episodes)

        self.assertEqual(first.shape, labels.shape)
        self.assertTrue(torch.equal(first, second))

    def test_calibration_strength_zero_recovers_raw_router(self):
        from phyrc.calibration import calibrated_domain_probability

        raw = torch.tensor([0.2, 0.8])
        result = calibrated_domain_probability(
            raw, torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9]), beta=0.0,
        )

        self.assertTrue(torch.allclose(result, raw, atol=1e-6))

    def test_calibration_moves_probability_toward_dual_evidence(self):
        from phyrc.calibration import calibrated_domain_probability

        raw = torch.tensor([0.5, 0.5])
        result = calibrated_domain_probability(
            raw, torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9]), beta=1.0,
        )

        self.assertLess(result[0], raw[0])
        self.assertGreater(result[1], raw[1])

    def test_candidate_selection_keeps_p1_when_no_feasible_gain(self):
        from phyrc.calibration import select_dual_candidate

        p1 = {
            "mode": "p1", "seen_zero": 0, "OA": 80.0,
            "Seen_AA": 85.0, "H": 60.0, "Unseen_AA": 50.0,
        }
        selected = select_dual_candidate([
            {
                "mode": "dual", "seen_zero": 1, "OA": 82.0,
                "Seen_AA": 86.0, "H": 70.0, "Unseen_AA": 65.0,
            },
            {
                "mode": "dual", "seen_zero": 0, "OA": 75.0,
                "Seen_AA": 80.0, "H": 65.0, "Unseen_AA": 60.0,
            },
        ], p1)

        self.assertEqual(selected["mode"], "p1")


class RCJDTests(unittest.TestCase):
    def test_class_mode_distribution_is_deterministic_and_normalized(self):
        from phyrc.attention import class_mode_distribution

        features = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9],
        ])
        first_modes, first_weights = class_mode_distribution(features)
        second_modes, second_weights = class_mode_distribution(features)

        self.assertTrue(torch.allclose(first_modes, second_modes))
        self.assertTrue(torch.allclose(first_weights, second_weights))
        self.assertEqual(tuple(first_modes.shape), (2, 2))
        self.assertAlmostEqual(float(first_weights.sum()), 1.0, places=6)

    def test_transport_mode_distribution_normalizes_joint_weights(self):
        from phyrc.attention import transport_mode_distribution

        modes, weights = transport_mode_distribution(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
            torch.tensor([[[1.0, 0.0], [0.8, 0.2]], [[0.0, 1.0], [0.2, 0.8]]]),
            torch.tensor([[0.6, 0.4], [0.5, 0.5]]),
            torch.tensor([0.7, 0.3]),
        )

        self.assertEqual(tuple(modes.shape), (4, 2))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_collapsed_distribution_recovers_cosine_score(self):
        from phyrc.attention import prototype_distribution_scores

        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        centre = torch.tensor([1.0, 0.0])
        scores = prototype_distribution_scores(
            features, torch.stack([centre, centre]), torch.tensor([0.3, 0.7]), 0.1,
        )

        expected = F.normalize(features, dim=1) @ F.normalize(centre, dim=0)
        self.assertTrue(torch.allclose(scores, expected, atol=1e-6))

    def test_adaptive_decoder_preserves_clear_seen_and_rescues_unseen(self):
        from phyrc.decoder import adaptive_joint_scores

        scores = torch.tensor([[1.0, 0.0], [0.8, 0.7]])
        adjusted = adaptive_joint_scores(
            scores, [0], [1], torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9]),
            a=0.4, b=0.2, c=0.0, d=0.0,
        )
        self.assertTrue(torch.equal(adjusted.argmax(1), torch.tensor([0, 1])))

    def test_nested_learned_decoder_uses_domain_evidence(self):
        from phyrc.decoder import learn_nested_risk_parameters, learned_joint_scores

        episodes = []
        for _ in range(3):
            episodes.append({
                "scores": torch.tensor([[1.0, 0.0], [0.8, 0.7]]),
                "labels": torch.tensor([0, 1]),
            })
        p_seen = torch.tensor([0.9, 0.1] * 3)
        p_unseen = torch.tensor([0.1, 0.9] * 3)
        learned = learn_nested_risk_parameters(episodes, p_seen, p_unseen, steps=80)
        adjusted = learned_joint_scores(
            episodes[0]["scores"], [0], [1], p_seen[:2], p_unseen[:2],
            learned["alpha"], learned["delta"],
        )

        self.assertGreater(learned["alpha"], 0.0)
        self.assertTrue(torch.equal(adjusted.argmax(1), episodes[0]["labels"]))

    def test_nested_learned_decoder_accepts_rcjd_reference(self):
        from phyrc.decoder import learn_nested_risk_parameters

        episodes = [
            {
                "scores": torch.tensor([[1.0, 0.0], [0.8, 0.7]]),
                "labels": torch.tensor([0, 1]),
            }
            for _ in range(3)
        ]
        learned = learn_nested_risk_parameters(
            episodes,
            torch.tensor([0.9, 0.1] * 3),
            torch.tensor([0.1, 0.9] * 3),
            reference={
                "mode": "rcjd", "temperature": 1.0,
                "risk_weight": 0.2, "unseen_prior": 0.0,
            },
            steps=10,
        )

        self.assertTrue(0.0 <= learned["alpha"] <= 0.6)
        self.assertTrue(-0.5 <= learned["delta"] <= 0.5)

    def test_zero_risk_weight_recovers_cosine_scores(self):
        from phyrc.decoder import joint_risk_scores

        scores = torch.tensor([[0.8, 0.2, 0.6]])
        adjusted = joint_risk_scores(
            scores, [0, 1], [2], torch.tensor([0.9]), torch.tensor([0.1]),
            temperature=1.0, risk_weight=0.0, unseen_prior=0.0,
        )

        self.assertTrue(torch.equal(adjusted, scores))

    def test_domain_evidence_changes_only_the_matching_domain(self):
        from phyrc.decoder import joint_risk_scores

        scores = torch.zeros(2, 2)
        adjusted = joint_risk_scores(
            scores, [0], [1],
            torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9]),
            temperature=1.0, risk_weight=1.0, unseen_prior=0.0,
        )

        self.assertGreater(adjusted[0, 0], adjusted[0, 1])
        self.assertGreater(adjusted[1, 1], adjusted[1, 0])

    def test_decoder_selection_enforces_stability_before_h(self):
        from phyrc.decoder import select_decoder_candidate

        p1 = {
            "mode": "p1", "seen_zero": 0, "OA": 80.0,
            "Seen_AA": 85.0, "H": 60.0, "Unseen_AA": 50.0, "worst_h": 20.0,
        }
        selected = select_decoder_candidate([
            {
                "mode": "rcjd", "seen_zero": 0, "OA": 79.0,
                "Seen_AA": 85.0, "H": 75.0, "Unseen_AA": 70.0, "worst_h": 40.0,
            },
            {
                "mode": "rcjd", "seen_zero": 0, "OA": 80.0,
                "Seen_AA": 84.5, "H": 65.0, "Unseen_AA": 60.0, "worst_h": 30.0,
            },
        ], p1)

        self.assertEqual(selected["OA"], 80.0)

    def test_decoder_selection_rejects_unsafe_learned_candidate(self):
        from phyrc.decoder import select_decoder_candidate

        reference = {
            "mode": "rcjd", "seen_zero": 0, "OA": 80.0,
            "Seen_AA": 85.0, "H": 60.0, "Unseen_AA": 50.0, "worst_h": 20.0,
        }
        learned = {
            "mode": "learned_rcjd", "seen_zero": 0, "OA": 79.4,
            "Seen_AA": 85.0, "H": 70.0, "Unseen_AA": 65.0, "worst_h": 30.0,
        }

        self.assertIs(select_decoder_candidate([learned], reference), reference)


if __name__ == "__main__":
    unittest.main()
