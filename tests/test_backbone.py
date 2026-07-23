import unittest

import torch

from models.backbone import PhysicsInvariantBackbone, SpectralMorphologyBackbone, build_backbone
from train_spectral_mae import semantic_geometry_loss, spectral_angle_loss


class SpectralMorphologyBackboneTest(unittest.TestCase):
    def test_shape_and_identity_initialization(self):
        model = SpectralMorphologyBackbone(103, feature_dim=64, hidden_dim=256)
        spectra = torch.randn(4, 103)
        self.assertEqual(model(spectra).shape, (4, 64))
        self.assertEqual(model.reconstruct(spectra).shape, spectra.shape)
        self.assertTrue(torch.allclose(model(spectra), model.base(spectra)))
        self.assertEqual(model.morphology_features(spectra).shape, (4, 64))

    def test_factory_builds_morphology_backbone(self):
        model = build_backbone(103, {
            "model": {"backbone": "spectral_morphology", "freeze_backbone": False},
            "spectral_mae": {"hidden_dim": 256},
        })
        self.assertIsInstance(model, SpectralMorphologyBackbone)

    def test_physics_backbone_starts_as_b1(self):
        model = PhysicsInvariantBackbone(103, 64, 256)
        spectra = torch.randn(4, 103)
        self.assertTrue(torch.allclose(model(spectra), model.base(spectra)))
        self.assertEqual(model.physics_features(spectra).shape, (4, 64))
        self.assertEqual(model.reconstruct(spectra).shape, spectra.shape)

    def test_semantic_geometry_matches_identical_class_layout(self):
        features = torch.tensor([[0., 0.], [0., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        semantics = features.clone()
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        direction, topology = semantic_geometry_loss(features, semantics, labels)
        self.assertLess(float(direction + topology), 1e-6)

    def test_spectral_angle_is_zero_for_same_shape(self):
        spectra = torch.rand(4, 103) + 0.1
        self.assertLess(float(spectral_angle_loss(2.0 * spectra, spectra)), 1e-6)


if __name__ == "__main__":
    unittest.main()
