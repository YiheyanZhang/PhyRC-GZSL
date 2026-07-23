import unittest

import torch

from phyrc_gzsl.baseline.genzsl_strict import (
    fit_semantic_projection,
    topk_seen_classes,
    train_genzsl,
    validate_seen_only,
)


class GenZSLStrictTest(unittest.TestCase):
    def test_semantic_projection_uses_seen_centres(self):
        attributes = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        centres = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        projected = fit_semantic_projection(attributes, centres, ridge=1e-3)
        self.assertEqual(projected.shape, (3, 3))
        self.assertTrue(torch.isfinite(projected).all())
        self.assertGreater(projected[0, 0], projected[0, 1])

    def test_topk_never_uses_unseen_or_same_seen_class(self):
        semantics = torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.8, .2]])
        neighbours = topk_seen_classes(semantics, seen_count=3, k=2)
        self.assertEqual(neighbours.shape, (4, 2))
        self.assertTrue((neighbours < 3).all())
        for cls in range(3):
            self.assertNotIn(cls, neighbours[cls].tolist())

    def test_unseen_training_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unseen"):
            validate_seen_only(torch.tensor([0, 2]), seen_count=2)

    def test_tiny_training_returns_all_class_logits(self):
        features = torch.nn.functional.normalize(torch.randn(12, 4), dim=1)
        labels = torch.arange(3).repeat_interleave(4)
        attributes = torch.randn(4, 2)
        classifier = train_genzsl(
            features, labels, attributes, 3, seed=1, epochs=1, loops=1,
            classifier_epochs=1, batch_size=6, synthetic_per_class=2,
            hidden_dim=8, use_svd=False,
        )
        self.assertEqual(classifier(features[:2]).shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
