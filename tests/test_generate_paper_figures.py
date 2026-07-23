import unittest
from pathlib import Path
import re
import tempfile


class PaperFigureDataTests(unittest.TestCase):
    def test_loads_all_formal_loco_candidates_without_test_metrics(self):
        from generate_paper_figures import load_loco_tradeoff_data

        root = Path(__file__).resolve().parents[1]
        data = load_loco_tradeoff_data(root)

        self.assertEqual(
            {dataset: len(units) for dataset, units in data.items()},
            {"paviau": 45, "houston": 75, "indian_pines": 80},
        )
        self.assertTrue(
            all(
                len(unit["candidates"]) == 60
                and "metrics" not in unit
                and all("feasible" in candidate for candidate in unit["candidates"])
                for units in data.values()
                for unit in units
            )
        )

    def test_reconstructs_exact_run_specific_feasibility(self):
        from generate_paper_figures import is_rcjd_feasible

        fallback = {"seen_zero": 1, "Seen_AA": 70.0, "OA": 60.0, "H": 50.0, "Unseen_AA": 40.0}
        feasible = {"seen_zero": 1, "Seen_AA": 69.0, "OA": 59.5, "H": 50.1, "Unseen_AA": 40.1}
        self.assertTrue(is_rcjd_feasible(feasible, fallback))
        for key, value in (
            ("seen_zero", 2),
            ("Seen_AA", 68.99),
            ("OA", 59.49),
            ("H", 50.0),
            ("Unseen_AA", 40.0),
        ):
            candidate = feasible | {key: value}
            self.assertFalse(is_rcjd_feasible(candidate, fallback), key)

    def test_writes_loco_tradeoff_pdf_and_png(self):
        from generate_paper_figures import write_loco_tradeoff_figure

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            pdf_path, png_path = write_loco_tradeoff_figure(root, Path(directory))
            self.assertEqual(pdf_path.suffix, ".pdf")
            self.assertEqual(png_path.suffix, ".png")
            self.assertGreater(pdf_path.stat().st_size, 10_000)
            self.assertGreater(png_path.stat().st_size, 10_000)

    def test_loads_only_formal_risk_and_failure_evidence(self):
        from generate_paper_figures import load_failure_data, load_risk_data

        root = Path(__file__).resolve().parents[1]
        risk = load_risk_data(root)
        failures = load_failure_data(root)

        self.assertEqual(set(risk), {"paviau", "houston", "longkou"})
        self.assertTrue(all(len(points) == 5 for points in risk.values()))
        self.assertAlmostEqual(risk["paviau"]["full"]["seen_mean"], 85.24501690519337)
        self.assertEqual(risk["houston"]["no_risk_constraint"]["seen_zero"], 34)
        self.assertEqual(
            [row["class_name"] for row in failures["indian_pines"]["hardest_three"]],
            ["Hay-windrowed", "Oats", "Wheat"],
        )
        self.assertAlmostEqual(
            failures["houston"]["hardest_three"][0]["mean_unseen_aa"],
            9.388083735909822,
        )

    def test_writes_self_contained_svg_figures(self):
        from generate_paper_figures import (
            _failure_svg,
            _risk_svg,
            load_failure_data,
            load_risk_data,
        )

        root = Path(__file__).resolve().parents[1]
        risk_svg = _risk_svg(load_risk_data(root))
        failure_svg = _failure_svg(load_failure_data(root))

        self.assertIn("Seen-class AA", risk_svg)
        self.assertIn("No risk constraint", risk_svg)
        self.assertIn("Hay-windrowed", failure_svg)
        self.assertIn("Indian Pines", failure_svg)
        label_y = {
            name: re.search(rf'<text x="[^"]+" y="([^"]+)"[^>]*>{name}</text>', failure_svg).group(1)
            for name in ("Hay-windrowed", "Oats")
        }
        self.assertNotEqual(label_y["Hay-windrowed"], label_y["Oats"])
        self.assertNotIn("TODO", risk_svg + failure_svg)


if __name__ == "__main__":
    unittest.main()
