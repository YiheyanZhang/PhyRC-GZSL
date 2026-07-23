import unittest

import torch

from phyrc_gzsl.evaluate_p1 import (
    apply_conformal_shield,
    apply_router_csd,
    build_density_router_features,
    build_router_features,
    choose_best_bias,
    evaluate_prototype_bank,
    fit_seen_conformal,
    fit_seen_unseen_router,
    learn_relational_parameters,
    predict_semantic_center,
    predict_learned_relational_center,
    router_probability,
    seen_conformal_probability,
    transfer_unseen_variance,
)


class P1EvaluationTests(unittest.TestCase):
    def test_learned_relational_parameters_recover_linear_center(self):
        attributes = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        centers = torch.cat((attributes, 2 * attributes), dim=1)
        learned = learn_relational_parameters(attributes, centers, steps=40)
        predicted = predict_learned_relational_center(
            torch.tensor([4.0]), attributes, centers,
            learned["ridge"], learned["tau"],
        )
        self.assertTrue(torch.allclose(predicted, torch.tensor([4.0, 8.0]), atol=0.1))

    def test_simple_semantic_center_uses_fixed_ridge_mapping(self):
        support = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        centers = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        predicted = predict_semantic_center(
            support, torch.tensor([1.0, 1.0]), centers, ridge=1e-6,
        )
        self.assertTrue(torch.allclose(predicted, torch.tensor([2.0, 3.0]), atol=1e-4))

    def test_prototype_evaluation_recovers_perfect_labels(self):
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        result = evaluate_prototype_bank(
            features, torch.tensor([0, 0, 1, 1]), prototypes, [0], [1],
        )
        self.assertEqual(result["OA"], 100.0)
        self.assertEqual(result["H"], 100.0)

    def test_unseen_bias_calibrates_borderline_prediction(self):
        result = evaluate_prototype_bank(
            torch.tensor([[1.0, 0.0], [0.95, 0.31]]),
            torch.tensor([0, 1]),
            torch.tensor([[1.0, 0.0], [0.8, 0.6]]),
            [0], [1], unseen_bias=0.1,
        )
        self.assertEqual(result["H"], 100.0)

    def test_csd_selects_best_mean_proxy_h(self):
        biases = (0.0, 0.2, 0.4)
        scores = torch.tensor([[20.0, 10.0], [60.0, 50.0], [90.0, 80.0]])
        self.assertEqual(choose_best_bias(biases, scores), 0.4)

    def test_router_features_include_seen_density(self):
        routed = build_router_features(
            torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
            torch.tensor([[1.0, 0.2], [0.5, 0.7]]),
            [0], [1], torch.tensor([[1.0, 0.0]]), torch.tensor([[0.1, 0.1]]),
        )
        self.assertEqual(routed.shape, (2, 4))
        self.assertLess(routed[0, 3], routed[1, 3])

    def test_router_learns_proxy_unseen_separation(self):
        inputs = torch.tensor([
            [1.0, 0.1, -0.9, 0.1], [0.9, 0.2, -0.7, 0.2],
            [0.2, 0.9, 0.7, 4.0], [0.1, 1.0, 0.9, 5.0],
        ])
        model = fit_seen_unseen_router(inputs, torch.tensor([0.0, 0.0, 1.0, 1.0]))
        probabilities = router_probability(inputs, model)
        self.assertLess(float(probabilities[:2].mean()), 0.5)
        self.assertGreater(float(probabilities[2:].mean()), 0.5)

    def test_router_csd_applies_sample_level_bias(self):
        adjusted = apply_router_csd(
            torch.tensor([[0.9, 0.8], [0.9, 0.8]]),
            [1], 0.4, torch.tensor([0.0, 1.0]),
        )
        self.assertEqual(adjusted[0].argmax().item(), 0)
        self.assertEqual(adjusted[1].argmax().item(), 1)

    def test_unseen_variance_follows_nearest_seen_center(self):
        variances = torch.tensor([[0.1, 0.2], [3.0, 4.0]])
        transferred = transfer_unseen_variance(
            torch.tensor([1.0, 0.01]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            variances,
            temperature=0.01,
        )
        self.assertTrue(torch.allclose(transferred, variances[0], atol=1e-3))

    def test_density_router_measures_unseen_distance(self):
        inputs = build_density_router_features(
            torch.tensor([[0.0, 1.0], [0.8, 0.2]]),
            torch.tensor([[0.1, 1.0], [0.8, 0.3]]),
            [0], [1], torch.tensor([[1.0, 0.0]]), torch.tensor([[0.1, 0.1]]),
            torch.tensor([0.0, 1.0]), torch.tensor([0.1, 0.1]),
        )
        self.assertEqual(inputs.shape, (2, 5))
        self.assertLess(inputs[0, 4], inputs[1, 4])

    def test_seen_conformal_probability_is_higher_in_class(self):
        calibration = fit_seen_conformal(
            torch.tensor([
                [0.9, 0.1], [0.8, 0.2], [0.85, 0.15],
                [0.1, 0.9], [0.2, 0.8], [0.15, 0.85],
            ]),
            torch.tensor([0, 0, 0, 1, 1, 1]),
        )
        probability = seen_conformal_probability(
            torch.tensor([[0.88, 0.12], [0.5, 0.5]]), calibration,
        )
        self.assertGreater(probability[0], probability[1])

    def test_conformal_shield_suppresses_seen_evidence(self):
        protected = apply_conformal_shield(
            torch.tensor([0.8, 0.8]), torch.tensor([0.9, 0.0]),
        )
        self.assertTrue(torch.allclose(protected, torch.tensor([0.08, 0.8]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
