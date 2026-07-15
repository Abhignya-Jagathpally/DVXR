"""Tests for the thin HTTP API (dvxr.serve.api).

Covers the fast, weight-free endpoints via Starlette's TestClient: /health, /tasks, /evidence,
/evidence/{task}, and error paths. Every response carries the not-a-diagnosis disclaimer, external
comparisons keep their DOI + protocol, and unknown tasks 404 rather than leak. The screening +
triage endpoints hit real cohorts/LaBraM, so they are exercised end-to-end by the CLI, not here.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from starlette.testclient import TestClient
    from dvxr.serve.api import create_app, DISCLAIMER
    _HAVE_STARLETTE = True
except Exception:  # pragma: no cover - optional api extra
    _HAVE_STARLETTE = False


@unittest.skipUnless(_HAVE_STARLETTE, "starlette not installed (api extra)")
class ApiFastEndpointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def test_health_ok_and_caveated(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("not a diagnosis", body["disclaimer"].lower())

    def test_tasks_lists_headline_aurocs(self):
        r = self.client.get("/tasks")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        names = {t["task"] for t in body["tasks"]}
        self.assertIn("mumtaz_depression", names)
        dep = next(t for t in body["tasks"] if t["task"] == "mumtaz_depression")
        self.assertAlmostEqual(dep["auroc_window"], 0.961, places=3)
        self.assertIn("not a diagnosis", body["disclaimer"].lower())

    def test_evidence_report_is_scoreboard_text(self):
        r = self.client.get("/evidence")
        self.assertEqual(r.status_code, 200)
        self.assertIn("AUROC", r.text)

    def test_external_comparison_has_doi_and_protocol(self):
        r = self.client.get("/evidence/mumtaz_depression")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("ours", body)
        ext = body.get("external", body.get("published", []))
        self.assertTrue(ext, "expected published comparators")
        for e in ext:
            self.assertTrue(e.get("doi"), f"comparator missing DOI: {e}")
            self.assertTrue(e.get("protocol"), f"comparator missing protocol: {e}")

    def test_unknown_task_404s(self):
        self.assertEqual(self.client.get("/evidence/not_a_task").status_code, 404)

    def test_screen_requires_task(self):
        r = self.client.post("/screen/subject", json={})
        self.assertEqual(r.status_code, 400)

    def test_every_json_response_carries_disclaimer(self):
        for path in ("/health", "/tasks"):
            self.assertEqual(self.client.get(path).json()["disclaimer"], DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
