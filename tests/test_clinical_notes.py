"""Tests for the REAL unstructured clinical-notes pathway (POW Goal 2).

Two layers:
  * Always-on (offline, no torch/transformers/network needed): the TF-IDF+SVD floor
    backend, the loader's bundled real MTSamples excerpt, the two bench-task builders,
    and the sota-path text routing — all exercised on the committed ``sample.parquet``.
  * Guarded: the real Bio_ClinicalBERT backend, run only when torch+transformers are
    importable and the weights are cached (mirrors tests/test_labram_real.py).
"""
import unittest

import numpy as np
import pandas as pd

from dvxr.encoders.base import (_ClinicalNotesBackend, _TfidfSvdBackend,
                                clinical_notes_available)
from dvxr.loaders import load_clinical_notes
from dvxr.bench.tasks import (clinical_notes_specialty_task,
                              clinical_notes_surgery_task)

# Use the bundled real excerpt so these tests never hit the network.
_SAMPLE = "data/real/clinical_notes"


def _sample_df():
    # allow_download=False forces the bundled sample.parquet (offline, deterministic)
    return load_clinical_notes(data_dir=_SAMPLE, allow_download=False)


class LoaderAndFloorTest(unittest.TestCase):
    def test_bundled_sample_is_real_text(self):
        df = _sample_df()
        self.assertGreaterEqual(len(df), 20)
        for col in ("note_text", "specialty", "specialty_name", "subject_id"):
            self.assertIn(col, df.columns)
        # real transcribed notes are long free text, not synthesized key/value strings
        self.assertGreater(df["note_text"].str.len().median(), 200)
        self.assertIn("MTSamples", df.attrs.get("provenance", ""))

    def test_tfidf_floor_backend_shapes_and_no_leak(self):
        df = _sample_df()
        frame, cols = df[["note_text"]], ["note_text"]
        tr, te = frame.iloc[:30], frame.iloc[30:]
        b = _TfidfSvdBackend(16)
        ztr = b.fit_transform(tr, cols)     # vocabulary fit on TRAIN only
        zte = b.transform(te)               # TEST transformed with the train vocab
        self.assertEqual(ztr.shape[0], len(tr))
        self.assertEqual(zte.shape[1], ztr.shape[1])
        self.assertTrue(np.isfinite(ztr).all() and np.isfinite(zte).all())

    def test_hashing_floor_and_surgery_label(self):
        # exercise the stateless hashing floor + binary label logic on the bundled sample
        from dvxr.bench.tasks import _notes_hashing_features
        from dvxr.loaders import CLINICAL_NOTES_SURGERY_LABEL
        df = _sample_df()
        X, names = _notes_hashing_features(df["note_text"].tolist())
        y = (df["specialty_name"] == CLINICAL_NOTES_SURGERY_LABEL).astype(int).to_numpy()
        self.assertTrue(set(np.unique(y)) <= {0, 1})
        self.assertGreater(int(y.sum()), 0)                 # sample includes surgery notes
        self.assertEqual(X.shape[0], len(df))
        self.assertEqual(len(names), X.shape[1])

    def test_task_builders_carry_raw_text(self):
        # full builders (network corpus if available, else bundled sample)
        for builder, metric in [(clinical_notes_surgery_task, "1-AUROC"),
                                (clinical_notes_specialty_task, "macro_f1")]:
            t = builder(max_notes=40)
            self.assertEqual(t.modalities, ["ehr_notes"])
            self.assertIn("notes_text", t.extra)
            self.assertEqual(len(t.extra["notes_text"]), t.n)
            self.assertEqual(t.metric, metric)

    def test_sota_path_routes_real_text_not_hashing(self):
        # the sota embedding must come from extra['notes_text'], not task.features
        from dvxr.bench import baselines
        t = clinical_notes_surgery_task(max_notes=40)
        seen = {}

        class _Fake:
            name = "fake"

            def _embed(self, frame, cols):
                seen["cols"] = list(cols)
                seen["text"] = frame[cols[0]].iloc[0]
                return np.zeros((len(frame), 4), dtype=float)

        import dvxr.encoders.base as base  # _sota_embeddings imports it from here
        orig = base.make_primary_backend
        base.make_primary_backend = lambda modality, cfg: _Fake()
        try:
            baselines._sota_embeddings(t)
        finally:
            base.make_primary_backend = orig
        self.assertEqual(seen["cols"], ["note_text"])
        self.assertIn(seen["text"], list(t.extra["notes_text"]))


@unittest.skipUnless(clinical_notes_available(),
                     "torch+transformers required for real Bio_ClinicalBERT")
class ClinicalBertRealTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.backend = _ClinicalNotesBackend("emilyalsentzer/Bio_ClinicalBERT", d=16)
        except Exception as exc:  # weights not cached / offline
            raise unittest.SkipTest(f"Bio_ClinicalBERT not loadable: {exc}")
        cls.df = _sample_df()

    def test_embeddings_finite_and_hidden_width(self):
        frame = self.df[["note_text"]].iloc[:6]
        emb = self.backend._embed(frame, ["note_text"])
        self.assertEqual(emb.shape[0], 6)
        self.assertEqual(emb.shape[1], 768)          # BERT hidden size
        self.assertTrue(np.isfinite(emb).all())

    def test_long_note_is_chunk_pooled(self):
        long_note = self.df.loc[self.df["note_text"].str.len().idxmax(), "note_text"]
        n_tok = len(self.backend.tok(long_note, add_special_tokens=False)["input_ids"])
        # the corpus contains notes beyond BERT's 512-token limit; embedding still works
        emb = self.backend._embed(pd.DataFrame({"note_text": [long_note]}), ["note_text"])
        self.assertEqual(emb.shape, (1, 768))
        self.assertTrue(np.isfinite(emb).all())
        if n_tok <= 512:
            self.skipTest("bundled sample's longest note fits in one window")

    def test_differs_from_tfidf_floor(self):
        frame = self.df[["note_text"]].iloc[:8]
        fm = self.backend._embed(frame, ["note_text"])
        floor = _TfidfSvdBackend(16).fit_transform(frame, ["note_text"])
        self.assertNotEqual(fm.shape[1], floor.shape[1])  # 768 vs <=16


if __name__ == "__main__":
    unittest.main()
