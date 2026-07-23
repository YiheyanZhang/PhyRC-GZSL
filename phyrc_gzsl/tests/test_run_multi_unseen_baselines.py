import unittest


class MultiUnseenBaselineRunnerTests(unittest.TestCase):
    def test_jobs_cover_five_methods_and_the_frozen_manifest(self):
        from phyrc_gzsl.run_multi_unseen_baselines import build_jobs

        jobs = build_jobs()

        self.assertEqual(len(jobs), 75)
        self.assertEqual(len({job["result"] for job in jobs}), 75)
        self.assertEqual(
            sorted({job["method"] for job in jobs}),
            ["cada_vae", "eszsl", "f_clswgan", "free", "sae"],
        )
        self.assertTrue(all("metrics" not in job for job in jobs))


if __name__ == "__main__":
    unittest.main()
