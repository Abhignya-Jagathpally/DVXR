"""Docs-consistency audit: the repo must state its divergence from the proposal HONESTLY and
consistently, so the human-facing surfaces cannot drift back toward implying the LLM-fusion thesis
won. Complements tests/test_honesty_audit.py (which governs the machine-checkable claim registry).
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _norm(p: Path) -> str:
    return " ".join(p.read_text().lower().split())


class DivergenceIsStatedTest(unittest.TestCase):
    def test_goal1_has_canonical_divergence_section(self):
        text = (ROOT / "GOAL1_COMPLIANCE.md").read_text()
        self.assertIn("## Divergence from the proposal", text,
                      "the canonical divergence section is missing")
        low = " ".join(text.lower().split())
        # the structural fact that makes cross-modal fusion untested must be stated in a human doc
        self.assertIn("no single dataset co-registers eeg+cgm+ehr", low)
        # the thesis pivot must be named, not implied
        self.assertIn("loses on all six real tasks", low)

    def test_paper_draft_is_marked_superseded(self):
        self.assertIn("superseded", _norm(ROOT / "PAPER_DRAFT.md"),
                      "the stale BCI paper draft must carry a SUPERSEDED banner")

    def test_readme_labram_framing_is_consistent(self):
        low = _norm(ROOT / "README.md")
        # README must acknowledge real LaBraM runs in the product (not only the stale 'not wired' line)
        self.assertIn("labram_real.py", low)
        # the demoted single-subject BCI run must not be sold as the headline result
        self.assertNotIn("the headline tangible result", low)


class NoWinFramingTest(unittest.TestCase):
    """No human surface may frame the excluded CACMF/LLM-predictor paths as a win."""
    SURFACES = ["README.md", "GOAL1_COMPLIANCE.md", "docs/ARCHITECTURE.md", "docs/MASTER_BRIEF.md"]
    FORBIDDEN = ["cacmf outperforms", "cacmf wins", "cacmf beats the", "learned fusion wins",
                 "llm-based prediction wins", "fusion is the win"]

    def test_no_surface_claims_a_forbidden_win(self):
        for rel in self.SURFACES:
            low = _norm(ROOT / rel)
            for bad in self.FORBIDDEN:
                self.assertNotIn(bad, low, f"{rel} frames an excluded path as a win: {bad!r}")


if __name__ == "__main__":
    unittest.main()
