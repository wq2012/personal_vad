"""Tests for personal_vad.frontend."""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import configs  # noqa: E402
from personal_vad import frontend  # noqa: E402


class FrontendTest(unittest.TestCase):

  def test_concat_meanstd(self):
    mean_1 = np.ones((40,), dtype=np.float32) * 2.0
    std_1 = np.ones((40,), dtype=np.float32) * 3.0
    mean_cat, std_cat = frontend.concat_meanstd(mean_1, std_1, dim_2=256)
    self.assertEqual(mean_cat.shape, (296,))
    self.assertEqual(std_cat.shape, (296,))
    np.testing.assert_allclose(mean_cat[:40], 2.0)
    np.testing.assert_allclose(mean_cat[40:], 0.0)
    np.testing.assert_allclose(std_cat[:40], 3.0)
    np.testing.assert_allclose(std_cat[40:], 1.0)

  def test_stack_and_subsample_frames(self):
    x = tf.reshape(tf.range(2 * 10 * 4, dtype=tf.float32), [2, 10, 4])
    stacked = frontend.stack_and_subsample_frames(
        x, left_context=3, right_context=0, frame_stride=3
    )
    # 10 frames with stride 3 -> frames 0, 3, 6, 9 -> 4 output frames
    # 4 dims * (1 + 3) context -> 16 dims
    self.assertEqual(stacked.shape, (2, 4, 16))
    # First output frame has 3 left-padded zero frames followed by frame 0
    np.testing.assert_allclose(stacked.numpy()[:, 0, :12], 0.0)
    np.testing.assert_allclose(stacked.numpy()[:, 0, 12:], x.numpy()[:, 0, :])

  def test_log_mel_frontend_pvad_v1(self):
    fe = frontend.LogMelFrontend(configs.FrontendConfig.pvad_v1())
    # 0.5 seconds at 16kHz = 8000 samples -> (8000 - 400) // 160 + 1 = 48 frames
    waveform = tf.random.normal([2, 8000], seed=1)
    features = fe(waveform)
    self.assertEqual(features.shape, (2, 48, 40))

    # 1-D waveform input
    features_1d = fe(waveform[0])
    self.assertEqual(features_1d.shape, (48, 40))
    np.testing.assert_allclose(
        features.numpy()[0], features_1d.numpy(), atol=1e-5
    )

  def test_log_mel_frontend_pvad_v2(self):
    fe = frontend.LogMelFrontend(configs.FrontendConfig.pvad_v2())
    # 0.5 seconds at 16kHz = 8000 samples -> (8000 - 512) // 160 + 1 = 47 frames
    # Subsampled by 3 -> ceil(47 / 3) = 16 frames of 512 dims
    waveform = tf.random.normal([1, 8000], seed=2)
    features = fe(waveform)
    self.assertEqual(features.shape, (1, 16, 512))


if __name__ == "__main__":
  unittest.main()
