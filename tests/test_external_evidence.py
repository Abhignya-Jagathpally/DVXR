"""Tests for the external published-SOTA registry (dvxr.serve.evidence.EXTERNAL_SOTA).

Every external comparator must carry provenance (citation + DOI) and a protocol label, and our own
numbers must be reported at both granularities. These are honesty tests: an external number without
its protocol, or dressed as a head-to-head win, would mislead.
"""
import unittest


class ExternalSotaTest(unittest.TestCase):
    def test_each_headline_cohort_has_published_comparators(self):
        from dvxr.serve.evidence import EXTERNAL_SOTA
        for task in ("mumtaz_depression", "wesad_stress", "eegmat_workload"):
            self.assertIn(task, EXTERNAL_SOTA)
            self.assertGreaterEqual(len(EXTERNAL_SOTA[task]), 2, f"{task}: need ≥2 comparators")

    def test_every_external_result_has_provenance_and_protocol(self):
        from dvxr.serve.evidence import EXTERNAL_SOTA
        valid_protocols = {"LOSO (cross-subject)", "subject-independent",
                           "within-subject/segment", "external-validation"}
        for task, results in EXTERNAL_SOTA.items():
            for e in results:
                self.assertTrue(e.citation.strip(), f"{task}: missing citation")
                self.assertRegex(e.doi, r"^10\.\d{4,}/", f"{task}: {e.method} DOI malformed: {e.doi}")
                self.assertIn(e.protocol, valid_protocols, f"{task}: {e.method} bad protocol")
                self.assertTrue(e.note.strip(), f"{task}: {e.method} missing cohort-match note")

    def test_external_comparison_reports_both_granularities_for_depression(self):
        from dvxr.serve.evidence import external_comparison
        d = external_comparison("mumtaz_depression")
        self.assertEqual(d["ours"]["window_auroc"], 0.961)
        self.assertEqual(d["ours"]["subject_auroc"], 0.986)      # subject-level present for diagnosis
        self.assertGreaterEqual(len(d["external"]), 3)
        self.assertIn("protocol", d["framing"].lower())

    def test_within_subject_tasks_have_no_subject_level_number(self):
        from dvxr.serve.evidence import OUR_METRICS
        for task in ("wesad_stress", "eegmat_workload"):
            self.assertIsNone(OUR_METRICS[task]["subject_auroc"],
                              f"{task} is a within-subject task; subject-level AUROC must not be faked")

    def test_report_shows_external_comparison_with_dois(self):
        from dvxr.serve.evidence import render_report
        r = render_report()
        self.assertIn("vs published SOTA", r)
        self.assertIn("10.1093/cercor/bhae505", r)     # the honest LOSO reference
        self.assertIn("subject-level", r)


if __name__ == "__main__":
    unittest.main()
