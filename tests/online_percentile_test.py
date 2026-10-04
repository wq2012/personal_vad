"""Tests for personal_vad.online_percentile."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import online_percentile  # noqa: E402


class OnlinePercentileTest(unittest.TestCase):

  def test_get_percentile_value_uniform(self):
    opv = online_percentile.OnlinePercentileValue(
        bins=10, hist_range=(0.0, 1.0)
    )
    for i in range(10):
      opv.observe(i / 10.0 + 0.05)
    for i in range(10):
      self.assertAlmostEqual(
          opv.get_percentile_value(i * 10), i / 10.0, places=5
      )

  def test_get_percentile_value_arrays(self):
    opv = online_percentile.OnlinePercentileValue(
        bins=10, hist_range=(0.0, 1.0)
    )
    opv.observe(np.asarray([0.05, 0.15, 0.25]))
    opv.observe(np.asarray([0.35, 0.45]))
    self.assertAlmostEqual(opv.get_percentile_value(0), 0.0)
    self.assertAlmostEqual(opv.get_percentile_value(20), 0.1)
    self.assertAlmostEqual(opv.get_percentile_value(40), 0.2)
    self.assertAlmostEqual(opv.get_percentile_value(60), 0.3)
    self.assertAlmostEqual(opv.get_percentile_value(80), 0.4)
    self.assertAlmostEqual(opv.get_percentile_value(100), 0.5)

  def test_cosine_range_and_reset(self):
    estimator = online_percentile.OnlinePercentileValue(
        bins=200, hist_range=(-1.0, 1.0)
    )
    samples = np.linspace(-1.0, 1.0, 2000, endpoint=False)
    estimator.observe(samples)
    self.assertEqual(estimator.histogram.shape, (200,))
    self.assertAlmostEqual(estimator.get_percentile_value(50.0), 0.0, places=5)
    self.assertAlmostEqual(estimator.get_percentile_value(5.0), -0.9, places=5)
    self.assertAlmostEqual(estimator.get_percentile_value(95.0), 0.9, places=5)
    estimator.reset()
    np.testing.assert_allclose(estimator.histogram, 0.0)

  def test_rescale_cosine_scores(self):
    raw_scores = np.linspace(-0.8, 0.9, 100, dtype=np.float32)
    rescaled = online_percentile.rescale_cosine_scores(raw_scores)
    self.assertEqual(rescaled.shape, raw_scores.shape)
    self.assertGreaterEqual(float(np.min(rescaled)), 0.0)
    self.assertLessEqual(float(np.max(rescaled)), 1.0)
    self.assertAlmostEqual(float(rescaled[0]), 0.0, places=5)
    self.assertAlmostEqual(float(rescaled[-1]), 1.0, places=5)

  def test_invalid_arguments_raise(self):
    with self.assertRaises(ValueError):
      online_percentile.OnlinePercentileValue(bins=0)
    with self.assertRaises(ValueError):
      online_percentile.OnlinePercentileValue(hist_range=(1.0, 0.0))
    opv = online_percentile.OnlinePercentileValue()
    with self.assertRaises(ValueError):
      opv.get_percentile_value(50.0)
    opv.observe(0.5)
    with self.assertRaises(ValueError):
      opv.get_percentile_value(105.0)


if __name__ == "__main__":
  unittest.main()
