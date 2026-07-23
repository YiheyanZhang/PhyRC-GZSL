import unittest

from run_phyrc_ablation_5seed import DATASETS, checkpoint, full_result_paths


class IndianPinesAblationTest(unittest.TestCase):
    def test_indian_pines_uses_submission_backbones_and_full_rows(self):
        self.assertIn("indian_pines", DATASETS)
        self.assertTrue(str(checkpoint("indian_pines", 42, 3)).endswith("indian_pines_p1_backbone_s3.pt"))
        self.assertTrue(str(checkpoint("indian_pines", 43, 3)).endswith("indian_pines_submission\\single\\seed43\\backbone_s3.pt"))
        paths = full_result_paths("indian_pines", 43)
        self.assertEqual(len(paths), 16)
        self.assertTrue(str(paths[0]).endswith("phyrc_s1.json"))
        self.assertTrue(str(paths[-1]).endswith("phyrc_s16.json"))


if __name__ == "__main__":
    unittest.main()
