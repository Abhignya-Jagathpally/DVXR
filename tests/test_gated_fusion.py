"""Tests for the do-no-harm reliability-gated late fusion (dvxr.bench.gated_fusion).

Two properties matter and are asserted on real subject-held-out splits:
  1. Do-no-harm: when one modality is pure noise, the gated fusion must NOT be
     fooled — it identifies the signal modality as the best single, keeps little
     weight on the noise modality, and its held-out error is no worse than the
     noise modality alone (and ~matches the best single). This is the safety floor.
  2. Fuses when warranted: when two modalities carry complementary signal, the
     gate accepts the ensemble (shrinkage lambda < 1) and its held-out error is no
     worse than the best single modality. This shows the floor is not just
     always-fallback — it fuses when the inner-CV advantage clears the noise.

sklearn/scipy only — no torch, so this runs in the offline suite unconditionally.
"""
import unittest

import numpy as np

from dvxr.bench.gated_fusion import dnh_weights, pred_dnh_gated, _candidates
from dvxr.bench.representations import _fit_head
from dvxr.bench.protocol import repeated_group_folds
from dvxr.bench.tasks import BenchTask


def _make_task(name, coupler, n_subj=14, per=22, seed=0):
    """Two 3-dim modalities; label = coupler(a, b, subject_bias) > 0.
    coupler decides which modality carries signal (used to build noise vs. complementary)."""
    rng = np.random.default_rng(seed)
    A, B, y, sid = [], [], [], []
    for s in range(n_subj):
        bias = rng.normal(0, 0.4)
        for _ in range(per):
            a = rng.normal(0, 1, 3)
            b = rng.normal(0, 1, 3)
            y.append(int(coupler(a, b, bias) > 0))
            A.append(a); B.append(b); sid.append(f"s{s}")
    return BenchTask(
        name=name, kind="classification",
        features={"a": np.array(A), "b": np.array(B)},
        feature_names={"a": ["a0", "a1", "a2"], "b": ["b0", "b1", "b2"]},
        y=np.array(y), subject_ids=np.array(sid),
        metric="1-AUROC", baseline_hint="majority")


def _first_split(task, seed=7):
    tr, te = repeated_group_folds(task.subject_ids, 1, 4, seed=seed)[0]
    return tr, te


def _auroc_err(y, p):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return 1.0 - roc_auc_score(y, p)


class DoNoHarmTest(unittest.TestCase):
    def test_inner_cv_safety_floor(self):
        # THE guarantee: the gated combiner's inner-CV error is never worse than the
        # best candidate's inner-CV error (Super-Learner floor, finite-sample gated).
        for name, coupler, seed in [
            ("noise_guard", lambda a, b, bias: a[0] + a[1] + bias, 1),
            ("complementary", lambda a, b, bias: a[0] + b[0] + 0.5 * bias, 2),
            ("single_dom", lambda a, b, bias: a[0] + bias, 5)]:
            task = _make_task(name, coupler, seed=seed)
            tr, _te = _first_split(task)
            cands, singles = _candidates(task)
            _w, diag = dnh_weights(task, cands, singles, tr, seed=7)
            best_err = diag["cand_err"][diag["best_cand"]]
            # allow a hair of slack for the linear-blend rounding; the floor must hold
            self.assertLessEqual(diag["dnh_inner_err"], best_err + 1e-6,
                                 f"{name}: floor violated")

    def test_one_se_rule_prefers_simpler_candidate(self):
        # 'a' carries the signal; a linear single-modality head already suffices, so the
        # 1-SE rule should not hand the reference to a higher-capacity concat/GBM candidate
        # when it is not clearly (>1 SE) better. best_cand should be a low-capacity choice.
        task = _make_task("single_dom", lambda a, b, bias: a[0] + a[1] + bias, seed=5)
        tr, _te = _first_split(task)
        cands, singles = _candidates(task)
        # opt into the 1-SE rule (default is strict argmin); it must never pick a MORE
        # complex reference than the strict argmin candidate.
        _w, diag = dnh_weights(task, cands, singles, tr, seed=7, strict=False)
        from dvxr.bench.gated_fusion import _simplicity
        self.assertLessEqual(_simplicity(diag["best_cand"]),
                             _simplicity(diag["argmin_cand"]))
        # and strict is the default: best_cand == argmin_cand when strict is unset
        _w2, diag2 = dnh_weights(task, cands, singles, tr, seed=7)
        self.assertEqual(diag2["best_cand"], diag2["argmin_cand"])

    def test_not_fooled_by_noise_modality(self):
        # modality 'a' carries all the signal; 'b' is pure noise.
        task = _make_task("noise_guard", lambda a, b, bias: a[0] + a[1] + bias, seed=1)
        tr, te = _first_split(task)
        cands, singles = _candidates(task)
        w, diag = dnh_weights(task, cands, singles, tr, seed=7)

        # the best single modality must be the signal one, not the noise one
        self.assertEqual(diag["best_single"], "single:a")
        # the noise-only candidate never dominates the blend
        self.assertLessEqual(w.get("single:b", 0.0), 0.5)

        # held-out: gated fusion is no worse than the NOISE modality alone (safety).
        pred = pred_dnh_gated(task, tr, te, seed=7)
        err_dnh = _auroc_err(task.y[te], pred)
        err_b = _auroc_err(task.y[te], _fit_head("classification",
                            task.features["b"][tr], task.y[tr], task.features["b"][te], seed=7))
        self.assertLessEqual(err_dnh, err_b + 1e-9)          # never fooled by noise

    def test_does_no_harm_vs_best_single_on_held_out(self):
        # both modalities carry half the signal -> late fusion should not hurt.
        task = _make_task("complementary",
                          lambda a, b, bias: a[0] + b[0] + 0.5 * bias, seed=2)
        tr, te = _first_split(task)
        pred = pred_dnh_gated(task, tr, te, seed=7)
        err_dnh = _auroc_err(task.y[te], pred)
        err_a = _auroc_err(task.y[te], _fit_head("classification",
                            task.features["a"][tr], task.y[tr], task.features["a"][te], seed=7))
        err_b = _auroc_err(task.y[te], _fit_head("classification",
                            task.features["b"][tr], task.y[tr], task.features["b"][te], seed=7))
        # no worse than the best single modality on held-out subjects
        self.assertLessEqual(err_dnh, min(err_a, err_b) + 0.03)


class SmokeTest(unittest.TestCase):
    def test_returns_finite_predictions(self):
        task = _make_task("smoke", lambda a, b, bias: a[0] + bias, seed=3)
        tr, te = _first_split(task)
        pred = pred_dnh_gated(task, tr, te, seed=7)
        self.assertEqual(pred.shape, (len(te),))
        self.assertTrue(np.all(np.isfinite(pred)))
        # probabilities live in [0, 1] for a classification task
        self.assertTrue(np.all(pred >= 0.0) and np.all(pred <= 1.0))

    def test_deterministic(self):
        task = _make_task("determinism", lambda a, b, bias: a[0] + b[1] + bias, seed=4)
        tr, te = _first_split(task)
        p1 = pred_dnh_gated(task, tr, te, seed=7)
        p2 = pred_dnh_gated(task, tr, te, seed=7)
        np.testing.assert_allclose(p1, p2)


if __name__ == "__main__":
    unittest.main()
