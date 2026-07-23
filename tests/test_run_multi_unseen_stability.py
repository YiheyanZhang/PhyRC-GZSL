import unittest

from run_multi_unseen_stability import DATASETS, build_jobs


class MultiUnseenStabilityTest(unittest.TestCase):
    def test_fixed_shifted_groups_and_job_count(self):
        self.assertEqual(DATASETS["paviau"]["groups"], [[2, 5], [4, 7]])
        self.assertEqual(DATASETS["houston"]["groups"], [[2, 6, 10], [4, 8, 12]])
        self.assertEqual(DATASETS["indian_pines"]["groups"], [[3, 7, 11], [5, 9, 13]])
        jobs = build_jobs()
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({job["backbone"] for job in jobs}), 30)
        self.assertTrue(all(set(job["results"]) == {"phyrc", "eszsl"} for job in jobs))


if __name__ == "__main__":
    unittest.main()
