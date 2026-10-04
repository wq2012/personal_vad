# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

  def test_original_online_percentile_value_lib_cases(self):
    calculator = online_percentile.OnlinePercentileValue(200, (-1.0, 1.0))
    data = np.arange(-1.0, 1.0, 0.001)
    calculator.Update(data)
    self.assertEqual(calculator.histogram.shape[0], 200)
    self.assertAlmostEqual(0.0, calculator.GetPercentile(50), places=5)
    self.assertAlmostEqual(-0.9, calculator.GetPercentile(5), places=5)

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
