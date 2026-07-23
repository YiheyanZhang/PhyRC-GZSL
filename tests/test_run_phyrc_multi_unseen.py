import unittest


class MultiUnseenRunnerTests(unittest.TestCase):
    def test_jobs_are_exactly_the_preregistered_cross_product(self):
        from run_phyrc_multi_unseen import build_jobs

        jobs = build_jobs()

        self.assertEqual(len(jobs), 15)
        self.assertEqual(len({job["result"] for job in jobs}), 15)
        expected = {"paviau": [3, 6], "houston": [3, 7, 11], "longkou": [3, 6]}
        for dataset, unseen in expected.items():
            rows = [job for job in jobs if job["dataset"] == dataset]
            self.assertEqual([job["seed"] for job in rows], list(range(42, 47)))
            self.assertTrue(all(job["unseen_classes"] == unseen for job in rows))
        self.assertTrue(all("metrics" not in job for job in jobs))

    def test_aggregation_uses_population_statistics_and_unseen_ids(self):
        from run_phyrc_multi_unseen import aggregate_results

        payloads = []
        for seed, value in ((42, 10.0), (43, 14.0)):
            metrics = {
                key: value for key in ("OA", "AA", "Kappa", "Seen_AA", "Unseen_AA", "H")
            }
            metrics["per_class"] = {0: 50.0, 1: value, 2: value + 2.0}
            payloads.append({
                "dataset": "toy", "seed": seed, "seen_classes": [1],
                "unseen_classes": [2, 3], "result": {"metrics": metrics},
            })

        summary = aggregate_results(payloads)["toy"]

        self.assertEqual(summary["H"], {"mean": 12.0, "std": 2.0})
        self.assertEqual(summary["unseen_per_class"]["2"], {"mean": 12.0, "std": 2.0})
        self.assertEqual(summary["unseen_per_class"]["3"], {"mean": 14.0, "std": 2.0})


if __name__ == "__main__":
    unittest.main()
