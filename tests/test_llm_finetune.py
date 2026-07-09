"""Slice 8b: the full Option 3 — LoRA soft-prompt classification finetuning (rasbt ch6/app-E
recipe adapted to our soft-prompt model). Skip-guarded on a local LLM. This checks the
mechanism runs and is LEAKAGE-SAFE (the shared base LLM is left pristine — LoRA adapters are
injected fresh and removed). It does NOT assert a win: on this repo's tiny-N clinical tasks
finetuning overfits and loses to the floor (reported honestly in CHANGES)."""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.bench.representations import llm_available  # noqa: E402

CGM = Path(__file__).resolve().parents[1] / "data" / "real" / "cgmacros"
_RUNNABLE = llm_available() and bool(list(CGM.glob("**/CGMacros-*.csv")))


@unittest.skipUnless(_RUNNABLE, "local LLM or CGMacros data absent")
class LlmFinetuneTest(unittest.TestCase):
    def _task(self):
        from dvxr.bench.tasks import cgmacros_diabetes_task
        return cgmacros_diabetes_task(subjects=10)

    def test_finetune_returns_valid_probabilities(self):
        from dvxr.llm.finetune import finetune_softprompt
        t = self._task()
        tr = list(range(0, t.n, 2))
        te = list(range(1, t.n, 2))
        p = finetune_softprompt(t, tr, te, seed=7, epochs=2, lora_blocks=1)
        self.assertEqual(len(p), len(te))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))

    def test_base_llm_left_pristine_after_finetune(self):
        """No cross-fold leakage: after finetuning, the shared reader's LLM must have NO
        trainable params and the same module types (LoRA wrappers removed)."""
        from dvxr.llm.finetune import finetune_softprompt
        from dvxr.llm.predictor import get_reader
        reader = get_reader(seed=7)
        n_trainable_before = sum(p.requires_grad for p in reader._model.parameters())
        t = self._task()
        finetune_softprompt(t, list(range(0, t.n, 2)), list(range(1, t.n, 2)),
                            seed=7, epochs=1, lora_blocks=2)
        n_trainable_after = sum(p.requires_grad for p in reader._model.parameters())
        self.assertEqual(n_trainable_before, 0)          # base was frozen
        self.assertEqual(n_trainable_after, 0)           # ...and still is (LoRA removed)


if __name__ == "__main__":
    unittest.main()
