"""Tests for personal_vad.tflite_export and scripts.export_tflite."""

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
from personal_vad import tflite_export  # noqa: E402
from scripts import export_tflite  # noqa: E402


class TFLiteExportTest(unittest.TestCase):

  def test_export_conformer_unquantized_and_quantized(self):
    tf.random.set_seed(101)
    cfg = configs.ModelConfig(
        backbone=configs.BackboneType.CONFORMER,
        conditioning_mode=configs.ConditioningMode.FILM_DVECTOR,
        num_classes=3,
        feature_dim=32,
        speaker_embedding_dim=32,
        conformer_num_layers=2,
        conformer_dim=32,
        conformer_num_heads=4,
        conformer_left_context=4,
        conformer_kernel_size=3,
        conformer_ffn_multiplier=4,
    )
    pvad = model_lib.PersonalVadModel(config=cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
      fp32_path = os.path.join(tmpdir, "model_fp32.tflite")
      int8_path = os.path.join(tmpdir, "model_int8.tflite")

      fp32_bytes = tflite_export.export_to_tflite(
          pvad, output_path=fp32_path, quantize=False, sequence_length=8
      )
      int8_bytes = tflite_export.export_to_tflite(
          pvad, output_path=int8_path, quantize=True, sequence_length=8
      )
      self.assertTrue(os.path.exists(fp32_path))
      self.assertTrue(os.path.exists(int8_path))
      self.assertLess(len(int8_bytes), len(fp32_bytes))

      features = np.random.randn(8, 32).astype(np.float32)
      spk_emb = np.random.randn(32).astype(np.float32)

      expected_probs = pvad.predict_proba(
          features[None, ...], speaker_embedding=spk_emb[None, ...]
      ).numpy()[0]

      runner_fp32 = tflite_export.TFLitePersonalVadRunner(fp32_path)
      tflite_probs = runner_fp32.predict(features, speaker_embedding=spk_emb)
      np.testing.assert_allclose(expected_probs, tflite_probs, atol=1e-4)

      # Test arbitrary sequence length (15 frames across two 8-frame chunks)
      long_features = np.random.randn(15, 32).astype(np.float32)
      runner_int8 = tflite_export.TFLitePersonalVadRunner(int8_bytes)
      long_probs = runner_int8.predict(
          long_features, speaker_embedding=spk_emb
      )
      self.assertEqual(long_probs.shape, (15, 3))

  def test_export_lstm_v1_and_cli_script(self):
    cfg = configs.ModelConfig(
        backbone=configs.BackboneType.LSTM_V1,
        conditioning_mode=configs.ConditioningMode.ET,
        num_classes=3,
        feature_dim=16,
        speaker_embedding_dim=16,
        lstm_num_layers=2,
        lstm_units=16,
        fc_units=16,
    )
    pvad = model_lib.PersonalVadModel(config=cfg)
    tflite_bytes = tflite_export.export_to_tflite(
        pvad, quantize=False, sequence_length=4
    )
    runner = tflite_export.TFLitePersonalVadRunner(tflite_bytes)
    features = np.random.randn(4, 16).astype(np.float32)
    spk_emb = np.random.randn(16).astype(np.float32)

    expected = pvad.predict_proba(
        features[None, ...], speaker_embedding=spk_emb[None, ...]
    ).numpy()[0]
    actual = runner.predict(features, speaker_embedding=spk_emb)
    np.testing.assert_allclose(expected, actual, atol=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
      cli_tflite = os.path.join(tmpdir, "cli.tflite")
      export_tflite.main([
          "--backbone",
          "lstm_v1",
          "--conditioning_mode",
          "et",
          "--sequence_length",
          "4",
          "--output_tflite",
          cli_tflite,
      ])
      self.assertTrue(os.path.exists(cli_tflite))


if __name__ == "__main__":
  tf.test.main()
