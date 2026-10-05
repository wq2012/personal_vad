"""Tests for personal_vad.layers."""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import layers  # noqa: E402


class LayersTest(unittest.TestCase):

  def test_safe_l2_normalize_zero_and_nonzero(self):
    x = tf.constant([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
    normed = layers.safe_l2_normalize(x).numpy()
    np.testing.assert_allclose(normed[0], [0.6, 0.8, 0.0], atol=1e-5)
    np.testing.assert_allclose(normed[1], [0.0, 0.0, 0.0], atol=1e-6)

  def test_feature_wise_modulation_layer(self):
    film = layers.FeatureWiseModulationLayer(
        feature_dim=16, has_bias=True, apply_residual=True
    )
    feature = tf.random.normal([2, 10, 16])
    modulator_2d = tf.random.normal([2, 32])
    out_2d = film(feature, modulator_2d)
    self.assertEqual(out_2d.shape, (2, 10, 16))

    modulator_3d = tf.random.normal([2, 10, 32])
    out_3d = film(feature, modulator_3d)
    self.assertEqual(out_3d.shape, (2, 10, 16))

  def test_conformer_block_streaming_equivalence(self):
    tf.random.set_seed(10)
    block = layers.ConformerBlock(
        model_dim=16,
        num_heads=4,
        left_context=5,
        right_context=0,
        kernel_size=3,
        ffn_multiplier=2,
        dropout_rate=0.0,
    )
    inputs = tf.random.normal([1, 8, 16])
    full_out = block(inputs, training=False).numpy()

    attn_c, valid_c, conv_c = block.init_cache(batch_size=1)
    stream_outputs = []
    for t in range(8):
      step_in = inputs[:, t:t + 1, :]
      step_out, attn_c, valid_c, conv_c = block.stream_step(
          step_in, attn_c, valid_c, conv_c
      )
      stream_outputs.append(step_out.numpy())
    stream_out = np.concatenate(stream_outputs, axis=1)
    np.testing.assert_allclose(full_out, stream_out, atol=1e-5)

  def test_speaker_prenet_and_streaming(self):
    tf.random.set_seed(20)
    prenet = layers.SpeakerPreNet(
        num_layers=2,
        model_dim=16,
        output_dim=32,
        num_heads=4,
        left_context=4,
        kernel_size=3,
        ffn_multiplier=2,
    )
    features = tf.random.normal([1, 6, 20])
    target_emb = tf.random.normal([1, 32])
    cos_sim, prenet_embs = prenet(
        features, target_emb, training=False, return_embeddings=True
    )
    self.assertEqual(cos_sim.shape, (1, 6, 1))
    self.assertEqual(prenet_embs.shape, (1, 6, 32))

    caches = prenet.init_cache(batch_size=1)
    stream_sims = []
    for t in range(6):
      step_sim, caches = prenet.stream_step(
          features[:, t:t + 1, :], target_emb, caches
      )
      stream_sims.append(step_sim.numpy())
    stream_sims_concat = np.concatenate(stream_sims, axis=1)
    np.testing.assert_allclose(cos_sim.numpy(), stream_sims_concat, atol=1e-5)

    # Zero enrollment d-vector should yield exact zero cosine similarity
    zero_emb = tf.zeros([1, 32], dtype=tf.float32)
    zero_sim = prenet(features, zero_emb, training=False).numpy()
    np.testing.assert_allclose(zero_sim, 0.0, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
