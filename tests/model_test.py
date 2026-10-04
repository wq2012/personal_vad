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
"""Tests for personal_vad.model and training/evaluation scripts."""

import os
import sys
import tempfile
import unittest
import numpy as np
import tensorflow as tf

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import configs  # noqa: E402
from personal_vad import model as model_lib  # noqa: E402
from scripts import evaluate  # noqa: E402
from scripts import prepare_dataset  # noqa: E402
from scripts import train  # noqa: E402


class ModelTest(unittest.TestCase):

  def test_pvad_v1_all_conditioning_modes(self):
    features = tf.random.normal([2, 10, 40])
    spk_emb = tf.random.normal([2, 256])
    cos_score = tf.random.uniform([2, 10, 1], minval=-1.0, maxval=1.0)

    for mode in (
        configs.ConditioningMode.SC,
        configs.ConditioningMode.ST,
        configs.ConditioningMode.ET,
        configs.ConditioningMode.SET,
    ):
      cfg = configs.ModelConfig.pvad_v1(conditioning_mode=mode)
      pvad = model_lib.PersonalVadModel(config=cfg)
      logits = pvad(
          {
              "features": features,
              "speaker_embedding": spk_emb,
              "cosine_score": cos_score,
          },
          training=False,
      )
      expected_classes = 2 if mode == configs.ConditioningMode.SC else 3
      self.assertEqual(logits.shape, (2, 10, expected_classes))

  def test_pvad_v2_conformer_all_conditioning_modes_and_streaming(self):
    features = tf.random.normal([1, 6, 32])
    spk_emb = tf.random.normal([1, 16])

    for mode in (
        configs.ConditioningMode.CONCAT,
        configs.ConditioningMode.FILM_DVECTOR,
        configs.ConditioningMode.FILM_COS,
        configs.ConditioningMode.CONCAT_COS,
        configs.ConditioningMode.FILM_DVECTOR_COS,
    ):
      cfg = configs.ModelConfig(
          backbone=configs.BackboneType.CONFORMER,
          conditioning_mode=mode,
          num_classes=3,
          feature_dim=32,
          speaker_embedding_dim=16,
          conformer_num_layers=2,
          conformer_dim=16,
          conformer_num_heads=4,
          conformer_left_context=4,
          conformer_kernel_size=3,
          conformer_ffn_multiplier=2,
          prenet_num_layers=1,
          prenet_dim=16,
          prenet_output_dim=16,
      )
      pvad = model_lib.PersonalVadModel(config=cfg)
      full_probs = pvad.predict_proba(
          features, speaker_embedding=spk_emb
      ).numpy()
      self.assertEqual(full_probs.shape, (1, 6, 3))

      state = pvad.init_streaming_state(batch_size=1)
      stream_probs_list = []
      for t in range(6):
        _, step_probs, state = pvad.stream_step(
            features[:, t : t + 1, :], speaker_embedding=spk_emb, state=state
        )
        stream_probs_list.append(step_probs.numpy())
      stream_probs = np.concatenate(stream_probs_list, axis=1)
      np.testing.assert_allclose(full_probs, stream_probs, atol=1e-5)

  def test_pvad_v2_lstm_and_streaming(self):
    cfg = configs.ModelConfig(
        backbone=configs.BackboneType.LSTM_V2,
        conditioning_mode=configs.ConditioningMode.FILM_DVECTOR,
        num_classes=3,
        feature_dim=32,
        speaker_embedding_dim=16,
        lstm_num_layers=2,
        lstm_units=16,
        fc_units=16,
        use_layer_norm=True,
    )
    pvad = model_lib.PersonalVadModel(config=cfg)
    features = tf.random.normal([1, 5, 32])
    spk_emb = tf.random.normal([1, 16])
    full_probs = pvad.predict_proba(
        features, speaker_embedding=spk_emb
    ).numpy()

    state = pvad.init_streaming_state(batch_size=1)
    stream_probs_list = []
    for t in range(5):
      _, step_probs, state = pvad.stream_step(
          features[:, t : t + 1, :], speaker_embedding=spk_emb, state=state
      )
      stream_probs_list.append(step_probs.numpy())
    stream_probs = np.concatenate(stream_probs_list, axis=1)
    np.testing.assert_allclose(full_probs, stream_probs, atol=1e-5)

  def test_end_to_end_prepare_train_evaluate_pipeline(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_npz = os.path.join(tmpdir, "train.npz")
      ckpt_dir = os.path.join(tmpdir, "ckpt")
      eval_dir = os.path.join(tmpdir, "eval")

      prepare_dataset.main([
          "--output_npz",
          dataset_npz,
          "--generate_synthetic",
          "--num_utterances",
          "12",
          "--frontend_version",
          "v1",
      ])
      self.assertTrue(os.path.exists(dataset_npz))

      train.main([
          "--train_npz",
          dataset_npz,
          "--checkpoint_dir",
          ckpt_dir,
          "--backbone",
          "lstm_v1",
          "--conditioning_mode",
          "et",
          "--epochs",
          "2",
          "--batch_size",
          "4",
      ])
      self.assertTrue(
          os.path.exists(os.path.join(ckpt_dir, "model.weights.h5"))
      )

      metrics = evaluate.main([
          "--eval_npz",
          dataset_npz,
          "--checkpoint_dir",
          ckpt_dir,
          "--output_dir",
          eval_dir,
      ])
      self.assertIsNotNone(metrics.mean_average_precision_micro)
      self.assertTrue(os.path.exists(os.path.join(eval_dir, "metrics.json")))
      self.assertTrue(os.path.exists(os.path.join(eval_dir, "report.html")))


if __name__ == "__main__":
  unittest.main()
