import unittest
from pathlib import Path

import numpy as np


class HierarchicalBootstrapTests(unittest.TestCase):
    def test_is_deterministic_and_preserves_the_paired_mean(self):
        from phyrc_gzsl.analyze_phyrc_statistics import hierarchical_paired_bootstrap

        differences = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        first = hierarchical_paired_bootstrap(differences, resamples=2000, seed=7)
        second = hierarchical_paired_bootstrap(differences, resamples=2000, seed=7)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 2.5)
        self.assertGreater(first["ci95"][0], 0.0)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            hierarchical_paired_bootstrap(np.array([1.0, 2.0]))

    def test_aligns_rows_by_unseen_identifier_and_rejects_mismatch(self):
        from phyrc_gzsl.analyze_phyrc_statistics import align_metric_rows

        phyrc = [
            {"unseen": 1, "metrics": {"H": 8.0}},
            {"unseen": 2, "metrics": {"H": 7.0}},
        ]
        baseline = [
            {"unseen": 2, "metrics": {"H": 3.0}},
            {"unseen": 1, "metrics": {"H": 2.0}},
        ]

        unseen, differences = align_metric_rows(phyrc, baseline, "H")
        self.assertEqual(unseen, [1, 2])
        self.assertTrue(np.array_equal(differences, np.array([6.0, 4.0])))
        with self.assertRaisesRegex(ValueError, "held-out classes differ"):
            align_metric_rows(phyrc, baseline[:1], "H")

    def test_submission_analysis_includes_indian_pines(self):
        from phyrc_gzsl.analyze_phyrc_statistics import build_analysis

        project_root = Path(__file__).resolve().parents[2]
        row = build_analysis(project_root)["datasets"]["indian_pines"]
        self.assertEqual(row["baseline"], "ESZSL")
        self.assertEqual(row["paired_units"], 80)
        self.assertEqual(round(row["mean_h_difference"], 2), 18.12)


if __name__ == "__main__":
    unittest.main()
