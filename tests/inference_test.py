"""Unit tests for `personal_vad.inference` and `scripts.inference`."""

import os
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf
import tensorflow as tf

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import personal_vad  # noqa: E402
from scripts import inference as inference_cli  # noqa: E402
from scripts import train  # noqa: E402


class InferenceTest(unittest.TestCase):

  def test_load_pretrained_and_predict_audio(self):
    tf.keras.utils.set_random_seed(42)
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = personal_vad.ModelConfig.pvad_v1(
          conditioning_mode=personal_vad.ConditioningMode.ET
      )
      model = personal_vad.PersonalVadModel(config=cfg)
      dummy_feat = tf.zeros([1, 4, cfg.feature_dim], dtype=tf.float32)
      dummy_spk = tf.zeros([1, cfg.speaker_embedding_dim], dtype=tf.float32)
      _ = model(
          {"features": dummy_feat, "speaker_embedding": dummy_spk},
          training=False,
      )
      train.save_checkpoint(model, tmpdir)
      personal_vad.export_to_tflite(
          model,
          output_path=os.path.join(tmpdir, "model_quantized.tflite"),
          quantize=True,
          sequence_length=8,
      )

      rng = np.random.default_rng(7)
      wav_in = (0.2 * rng.standard_normal(16000)).astype(np.float32)
      wav_enroll = (0.2 * rng.standard_normal(16000)).astype(np.float32)
      in_path = os.path.join(tmpdir, "input.wav")
      enroll_path = os.path.join(tmpdir, "enroll.wav")
      out_wav = os.path.join(tmpdir, "filtered.wav")
      out_json = os.path.join(tmpdir, "result.json")
      sf.write(in_path, wav_in, 16000)
      sf.write(enroll_path, wav_enroll, 16000)

      engine = personal_vad.load_pretrained(tmpdir)
      res = engine.predict_audio(
          in_path, enrollment_audio_or_embedding=enroll_path
      )
      self.assertEqual(res["posteriors"].shape[1], 3)
      self.assertEqual(len(res["filtered_audio"]), 16000)
      self.assertAlmostEqual(
          float(np.sum(res["posteriors"][0])), 1.0, places=4
      )

      # Verify CLI with TFLite execution
      cli_res = inference_cli.main([
          "--model",
          tmpdir,
          "--audio_path",
          in_path,
          "--enrollment_path",
          enroll_path,
          "--use_tflite",
          "--output_wav",
          out_wav,
          "--output_json",
          out_json,
      ])
      self.assertEqual(cli_res["posteriors"].shape[1], 3)
      self.assertTrue(os.path.exists(out_wav))
      self.assertTrue(os.path.exists(out_json))


if __name__ == "__main__":
  unittest.main()
