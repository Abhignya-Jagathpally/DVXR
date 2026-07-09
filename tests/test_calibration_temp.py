"""Slice 6a: temperature scaling. A miscalibrated (over-confident) probability set
should get a lower ECE after temperature scaling, and a well-calibrated set should be
left essentially unchanged (T ~= 1)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.calibration import (  # noqa: E402
    TemperatureScaler,
    expected_calibration_error,
    fit_temperature_scaler,
)


class TemperatureScalingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 5000
        # p_true is calibrated by construction: draw y ~ Bernoulli(p_true).
        p_true = rng.uniform(0.05, 0.95, n)
        self.y = (rng.uniform(size=n) < p_true).astype(int)
        logit = np.log(p_true / (1 - p_true))
        # report a sharpened probability -> over-confident, should need T ~= 3 to undo.
        self.overconfident = 1 / (1 + np.exp(-logit * 3.0))

    def test_scaler_reduces_ece_on_overconfident(self):
        scaler = fit_temperature_scaler(self.overconfident, self.y)
        cal = scaler.predict(self.overconfident)
        ece_before = expected_calibration_error(self.y, self.overconfident)
        ece_after = expected_calibration_error(self.y, cal)
        self.assertGreater(scaler.temperature, 1.0)  # needs softening
        self.assertLess(ece_after, ece_before)

    def test_identity_scaler_is_noop(self):
        s = TemperatureScaler(temperature=1.0)
        p = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(s.predict(p), p, atol=1e-6)

    def test_single_class_returns_identity(self):
        scaler = fit_temperature_scaler(np.array([0.2, 0.3, 0.4]), np.array([1, 1, 1]))
        self.assertEqual(scaler.temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
