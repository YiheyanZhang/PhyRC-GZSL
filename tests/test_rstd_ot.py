import unittest

import torch

from diagnose_rstd_ot import sinkhorn_plan


class RstdOtTests(unittest.TestCase):
    def test_sinkhorn_has_uniform_marginals(self):
        cost = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
        plan = sinkhorn_plan(cost, epsilon=0.1, iterations=100)
        self.assertTrue(torch.allclose(plan.sum(0), torch.full((2,), 0.5), atol=1e-3))
        self.assertTrue(torch.allclose(plan.sum(1), torch.full((2,), 0.5), atol=1e-3))
        self.assertGreater(float(plan.diag().sum()), 0.99)


if __name__ == "__main__":
    unittest.main()
