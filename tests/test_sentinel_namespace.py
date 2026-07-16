"""PR18 / §23: the product surface is a single curated namespace (dvxr.sentinel) that exposes the
Sentinel product and NOTHING experimental; bench and experiments are separate namespaces."""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dvxr.sentinel as sentinel  # noqa: E402

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "dvxr")


class SentinelNamespaceTest(unittest.TestCase):
    def test_product_surface_is_exposed(self):
        for name in ("generate_risk_report", "RiskPredictionService", "AbstainingPredictionService",
                     "CgmOnlyExcursionService", "build_excursion_labels", "require_synchronized_for_fusion",
                     "select_action", "grounded_explanation", "build_patient_snapshot",
                     "LocalKeywordTextIndex", "create_product_api"):
            self.assertTrue(hasattr(sentinel, name), f"dvxr.sentinel must expose {name}")

    def test_only_v1_routes_are_the_product_contract(self):
        self.assertEqual(sentinel.PRODUCT_ROUTES,
                         ("/v1/risk-reports", "/v1/predictions/{prediction_id}"))

    def test_sentinel_does_not_expose_experimental_predictors(self):
        exported = set(getattr(sentinel, "__all__", []))
        for bad in ("llm_representation_probe", "FusedRealtimeMonitor", "InsightGenerator",
                    "personal_insight"):
            self.assertNotIn(bad, exported)

    def test_sentinel_module_does_not_import_experimental_paths(self):
        with open(os.path.join(SRC, "sentinel", "__init__.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        for bad in ("llm.insight", "llm_representation_probe", "llm.predictor",
                    "realtime.heuristic_demo", "realtime.monitor"):
            self.assertFalse(any(bad in m for m in mods),
                             f"dvxr.sentinel must not import {bad!r}")

    def test_bench_and_experiments_are_separate_packages(self):
        import dvxr.bench  # noqa: F401
        import dvxr.experiments  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
