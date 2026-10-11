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

      # Verify documented dict keys and infer_from_files API
      self.assertIn("probabilities", res)
      self.assertIn("target_mask", res)
      res_files = engine.infer_from_files(in_path, enroll_path)
      np.testing.assert_allclose(
          res["posteriors"], res_files["probabilities"], atol=1e-5
      )

      # Verify CLI with documented --audio_wav / --enroll_wav / --use_tflite
      cli_res = inference_cli.main([
          "--model",
          tmpdir,
          "--audio_wav",
          in_path,
          "--enroll_wav",
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

      # Verify --enrollment_less CLI flag
      cli_no_enroll = inference_cli.main([
          "--model",
          tmpdir,
          "--audio_wav",
          in_path,
          "--enrollment_less",
      ])
      self.assertTrue(cli_no_enroll["enrollment_less"])

  def test_causal_prefix_invariance_and_pure_log_mel_v2(self):
    tf.keras.utils.set_random_seed(99)
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = personal_vad.ModelConfig.pvad_v2_conformer(
          conditioning_mode=personal_vad.ConditioningMode.FILM_DVECTOR_COS
      )
      model = personal_vad.PersonalVadModel(config=cfg)
      dummy_feat = tf.zeros([1, 4, cfg.feature_dim], dtype=tf.float32)
      dummy_spk = tf.zeros([1, cfg.speaker_embedding_dim], dtype=tf.float32)
      _ = model(
          {"features": dummy_feat, "speaker_embedding": dummy_spk},
          training=False,
      )
      train.save_checkpoint(model, tmpdir)

      rng = np.random.default_rng(123)
      full_wav = (0.25 * rng.standard_normal(32000)).astype(np.float32)
      prefix_wav = full_wav[:16000].copy()
      enroll_a = (0.25 * rng.standard_normal(16000)).astype(np.float32)
      enroll_b = (0.25 * rng.standard_normal(16000)).astype(np.float32)

      engine = personal_vad.load_pretrained(tmpdir)
      res_prefix = engine.predict_audio(
          prefix_wav, enrollment_audio_or_embedding=enroll_a, use_tflite=False
      )
      res_full = engine.predict_audio(
          full_wav, enrollment_audio_or_embedding=enroll_a, use_tflite=False
      )
      self.assertAlmostEqual(res_full["frame_step_sec"], 0.03, places=5)
      n_pref = res_prefix["features"].shape[0]
      np.testing.assert_allclose(
          res_prefix["features"], res_full["features"][:n_pref], atol=1e-6
      )
      np.testing.assert_allclose(
          res_prefix["posteriors"], res_full["posteriors"][:n_pref], atol=1e-5
      )

      # Verify features are 100% pure log-Mel (invariant to target speaker)
      res_other_spk = engine.predict_audio(
          prefix_wav, enrollment_audio_or_embedding=enroll_b, use_tflite=False
      )
      np.testing.assert_array_equal(
          res_prefix["features"], res_other_spk["features"]
      )

      # Verify automatic resampling from 48 kHz stereo input
      wav_48k_stereo = np.stack([prefix_wav, prefix_wav], axis=-1)
      wav_48k_path = os.path.join(tmpdir, "stereo_48k.wav")
      sf.write(wav_48k_path, wav_48k_stereo, 48000)
      res_48k = engine.predict_audio(
          wav_48k_path, enrollment_audio_or_embedding=enroll_a, use_tflite=False
      )
      self.assertGreater(res_48k["posteriors"].shape[0], 5)


if __name__ == "__main__":
  unittest.main()
