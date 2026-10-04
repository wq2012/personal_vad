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
"""Tests for personal_vad.loss."""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import loss  # noqa: E402


class WeightedPairwiseLossTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.logits = tf.constant(
        [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=tf.float32,
    )
    self.labels = tf.constant([1, 2, 0, 2], dtype=tf.int32)

  def test_normalize_weights_positive(self):
    weights = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 0.5}
    normalized = loss.normalize_weights(weights)
    self.assertAlmostEqual(normalized[(0, 1)], 0.4)
    self.assertAlmostEqual(normalized[(0, 2)], 0.4)
    self.assertAlmostEqual(normalized[(1, 2)], 0.2)

  def test_normalize_weights_all_zero(self):
    weights = {(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0}
    normalized = loss.normalize_weights(weights)
    self.assertEqual(normalized, {(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0})

  def test_normalize_weights_negative_raises(self):
    with self.assertRaises(ValueError):
      loss.normalize_weights({(0, 1): -1.0, (0, 2): 1.0, (1, 2): 0.5})

  def test_invalid_weights_raises(self):
    with self.assertRaises(ValueError):
      loss.WeightedPairwiseLoss(weights={(0, 1): 1.0}, num_classes=3)
    with self.assertRaises(ValueError):
      loss.WeightedPairwiseLoss(
          weights={(0, 0): 1.0, (0, 2): 1.0, (1, 2): 1.0}, num_classes=3
      )
    with self.assertRaises(ValueError):
      loss.WeightedPairwiseLoss(
          weights={(1, 0): 1.0, (0, 2): 1.0, (1, 2): 1.0},
          num_classes=3,
          symmetric_weights=True,
      )

  def test_get_pairwise_weight(self):
    weights = {(0, 1): 1.0, (0, 2): 0.6, (1, 2): 0.4}
    loss_fn = loss.WeightedPairwiseLoss(weights=weights, num_classes=3)
    class2 = tf.constant([0, 1, 2, 1], dtype=tf.int32)
    w0 = loss_fn.get_pairwise_weight(0, class2).numpy()
    w1 = loss_fn.get_pairwise_weight(1, class2).numpy()
    w2 = loss_fn.get_pairwise_weight(2, class2).numpy()
    np.testing.assert_allclose(w0, [0.0, 0.5, 0.3, 0.5], atol=1e-5)
    np.testing.assert_allclose(w1, [0.5, 0.0, 0.2, 0.0], atol=1e-5)
    np.testing.assert_allclose(w2, [0.3, 0.2, 0.0, 0.2], atol=1e-5)

  def test_compute_loss_zero_weights(self):
    weights = {(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0}
    loss_fn = loss.WeightedPairwiseLoss(weights=weights, num_classes=3)
    val = float(loss_fn.compute_loss(self.logits, self.labels).numpy())
    self.assertAlmostEqual(val, 0.0, places=5)

  def test_compute_loss_equal_weights(self):
    probability_value = np.array(
        [
            [0.3, 0.4, 0.1],
            [0.3, 0.2, 0.2],
            [0.4, 0.3, 0.1],
            [0.2, 0.3, 0.2],
            [0.4, 0.1, 0.3],
            [0.2, 0.2, 0.3],
        ],
        dtype=np.float32,
    )
    logits = tf.constant(np.log(probability_value), dtype=tf.float32)
    labels = tf.constant([0, 0, 1, 1, 2, 2], dtype=tf.int32)
    weights = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0}
    loss_fn = loss.WeightedPairwiseLoss(weights=weights, num_classes=3)
    per_frame = loss_fn.compute_per_frame_loss(logits, labels).numpy()
    loss_1 = (
        -0.5
        * (np.log(0.3) - np.log(0.3 + 0.4) + np.log(0.3) - np.log(0.3 + 0.1))
        / 3.0
    )
    loss_2 = -0.5 * ((np.log(0.3) - np.log(0.3 + 0.2)) * 2.0) / 3.0
    expected = np.array([loss_1, loss_2, loss_1, loss_2, loss_1, loss_2])
    np.testing.assert_allclose(per_frame, expected, atol=1e-5)
    val = float(loss_fn.compute_loss(logits, labels).numpy())
    self.assertAlmostEqual(val, float(np.mean(expected)), places=5)

  def test_compute_loss_unequal_weights(self):
    probability_value = np.array(
        [
            [0.3, 0.4, 0.1],
            [0.3, 0.2, 0.2],
            [0.4, 0.3, 0.1],
            [0.2, 0.3, 0.2],
            [0.4, 0.1, 0.3],
            [0.2, 0.2, 0.3],
        ],
        dtype=np.float32,
    )
    logits = tf.constant(np.log(probability_value), dtype=tf.float32)
    labels = tf.constant([0, 0, 1, 1, 2, 2], dtype=tf.int32)
    weights = {(0, 1): 1.0, (0, 2): 0.5, (1, 2): 0.25}
    loss_fn = loss.WeightedPairwiseLoss(weights=weights, num_classes=3)
    per_frame = loss_fn.compute_per_frame_loss(logits, labels).numpy()
    loss_1 = (
        -0.5
        * (
            np.log(0.3)
            - np.log(0.3 + 0.4)
            + (np.log(0.3) - np.log(0.3 + 0.1)) * 0.5
        )
        / 1.75
    )
    loss_2 = (
        -0.5
        * (
            np.log(0.3)
            - np.log(0.3 + 0.2)
            + (np.log(0.3) - np.log(0.3 + 0.2)) * 0.5
        )
        / 1.75
    )
    loss_3 = (
        -0.5
        * (
            np.log(0.3)
            - np.log(0.3 + 0.4)
            + (np.log(0.3) - np.log(0.3 + 0.1)) * 0.25
        )
        / 1.75
    )
    loss_4 = (
        -0.5
        * (
            np.log(0.3)
            - np.log(0.3 + 0.2)
            + (np.log(0.3) - np.log(0.3 + 0.2)) * 0.25
        )
        / 1.75
    )
    loss_5 = (
        -0.5
        * (
            (np.log(0.3) - np.log(0.3 + 0.4)) * 0.5
            + (np.log(0.3) - np.log(0.3 + 0.1)) * 0.25
        )
        / 1.75
    )
    loss_6 = (
        -0.5
        * (
            (np.log(0.3) - np.log(0.3 + 0.2)) * 0.5
            + (np.log(0.3) - np.log(0.3 + 0.2)) * 0.25
        )
        / 1.75
    )
    expected = np.array([loss_1, loss_2, loss_3, loss_4, loss_5, loss_6])
    np.testing.assert_allclose(per_frame, expected, atol=1e-5)
    val = float(loss_fn.compute_loss(logits, labels).numpy())
    self.assertAlmostEqual(val, float(np.mean(expected)), places=5)

  def test_compute_loss_asymmetric_weights(self):
    weights = {
        (0, 1): 1.0,
        (0, 2): 0.5,
        (1, 0): 0.33,
        (1, 2): 0.25,
        (2, 0): 0.5,
        (2, 1): 0.25,
    }
    loss_fn = loss.WeightedPairwiseLoss(
        weights=weights, num_classes=3, symmetric_weights=False
    )
    labels = tf.constant([0, 1, 1, 2], dtype=tf.int32)
    w_alt1 = loss_fn.get_pairwise_weight(1, labels).numpy()
    sum_w = sum(weights.values())
    np.testing.assert_allclose(
        w_alt1, np.array([1.0, 0.0, 0.0, 0.25]) / sum_w, atol=1e-5
    )
    val = float(loss_fn.compute_loss(self.logits, self.labels).numpy())
    self.assertGreater(val, 0.0)

  def test_compute_loss_with_label_smoothing_and_mask(self):
    probability_value = np.array(
        [
            [0.3, 0.4, 0.1],
            [0.3, 0.2, 0.2],
            [0.4, 0.3, 0.1],
            [0.2, 0.3, 0.2],
            [0.4, 0.1, 0.3],
            [0.2, 0.2, 0.3],
        ],
        dtype=np.float32,
    )
    logits = tf.constant(np.log(probability_value), dtype=tf.float32)
    labels = tf.constant([0, 0, 1, 1, 2, 2], dtype=tf.int32)
    weights = {(0, 1): 1.0, (0, 2): 0.5, (1, 2): 0.25}
    loss_fn = loss.WeightedPairwiseLoss(
        weights=weights, num_classes=3, label_smoothing=0.1
    )
    per_frame = loss_fn.compute_per_frame_loss(logits, labels).numpy()
    expected = np.array(
        [0.28692007, 0.22761382, 0.2624477, 0.18967819, 0.14346004, 0.1138069]
    )
    np.testing.assert_allclose(per_frame, expected, atol=1e-5)
    mask = tf.constant([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=tf.float32)
    val = float(loss_fn.compute_loss(logits, labels, mask=mask).numpy())
    self.assertAlmostEqual(val, float(np.mean(expected[:3])), places=5)

  def test_cross_entropy_loss(self):
    ce_fn = loss.CrossEntropyLoss(num_classes=3, label_smoothing=0.05)
    mask = tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32)
    val = float(ce_fn.compute_loss(self.logits, self.labels, mask=mask).numpy())
    self.assertGreater(val, 0.0)


if __name__ == "__main__":
  unittest.main()
