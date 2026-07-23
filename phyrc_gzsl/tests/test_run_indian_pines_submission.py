import unittest


class IndianPinesSubmissionRunnerTests(unittest.TestCase):
    def test_manifest_freezes_complete_single_and_multi_protocols(self):
        from phyrc_gzsl.run_indian_pines_submission import build_manifest

        manifest = build_manifest()

        self.assertEqual(manifest["seeds"], [42, 43, 44, 45, 46])
        self.assertEqual(manifest["single_unseen_classes"], list(range(1, 17)))
        self.assertEqual(manifest["multi_unseen_classes"], [4, 8, 12])
        self.assertEqual(len(manifest["single_evaluations"]), 480)
        self.assertEqual(len(manifest["multi_evaluations"]), 30)
        self.assertTrue(all("metrics" not in job for job in manifest["single_evaluations"]))


if __name__ == "__main__":
    unittest.main()
