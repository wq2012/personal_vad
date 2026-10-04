"""Tests for personal_vad.dataset."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import configs  # noqa: E402
from personal_vad import dataset  # noqa: E402


class DatasetTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.utt1 = dataset.UtteranceData(
        utt_id="utt1",
        speaker_id="spk_a",
        features=np.ones((4, 8), dtype=np.float32),
        labels=np.asarray([1, 0, 0, 1], dtype=np.int32),
        speaker_embedding=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    self.utt2 = dataset.UtteranceData(
        utt_id="utt2",
        speaker_id="spk_b",
        features=np.ones((3, 8), dtype=np.float32) * 2.0,
        labels=np.asarray([0, 0, 1], dtype=np.int32),
        speaker_embedding=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )

  def test_concat_utterance_sequence_target_first(self):
    concat = dataset.concat_utterance_sequence(
        [self.utt1, self.utt2], target_index=0
    )
    self.assertEqual(concat.utt_id, "utt1+utt2")
    self.assertEqual(concat.speaker_id, "spk_a")
    self.assertEqual(concat.features.shape, (7, 8))
    # utt1 (target spk_a): [1, 0, 0, 1]
    # utt2 (non-target spk_b): [0->2, 0->2, 1->1]
    np.testing.assert_array_equal(concat.labels, [1, 0, 0, 1, 2, 2, 1])
    # Cosine similarity should be 1.0 on utt1 frames and 0.0 on utt2 frames
    np.testing.assert_allclose(concat.cosine_scores[:4, 0], 1.0, atol=1e-5)
    np.testing.assert_allclose(concat.cosine_scores[4:, 0], 0.0, atol=1e-5)

  def test_concat_utterance_sequence_vad_baseline(self):
    concat = dataset.concat_utterance_sequence(
        [self.utt1, self.utt2], target_index=0, convert_for_vad_baseline=True
    )
    np.testing.assert_array_equal(concat.labels, [1, 0, 0, 1, 0, 0, 1])

  def test_convert_alignment_for_vad_baseline(self):
    labels = np.asarray([0, 1, 2, 2, 1, 0], dtype=np.int32)
    converted = dataset.convert_alignment_for_vad_baseline(labels)
    np.testing.assert_array_equal(converted, [0, 1, 0, 0, 1, 0])

  def test_apply_enrollment_less_conditioning(self):
    emb = np.asarray([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
    labels = np.asarray([0, 1, 2, 2, 1], dtype=np.int32)
    new_emb, new_labels, dropped = dataset.apply_enrollment_less_conditioning(
        emb, labels, enrollment_less_prob=1.0
    )
    self.assertTrue(dropped)
    np.testing.assert_allclose(new_emb, 0.0)
    np.testing.assert_array_equal(new_labels, [0, 1, 0, 0, 1])

    kept_emb, kept_labels, dropped_zero = (
        dataset.apply_enrollment_less_conditioning(
            emb, labels, enrollment_less_prob=0.0
        )
    )
    self.assertFalse(dropped_zero)
    np.testing.assert_allclose(kept_emb, emb)
    np.testing.assert_array_equal(kept_labels, labels)

  def test_use_true_speaker_embedding_for_utterance(self):
    self.assertFalse(
        dataset.use_true_speaker_embedding_for_utterance("utt_001", 0.0)
    )
    self.assertTrue(
        dataset.use_true_speaker_embedding_for_utterance("utt_001", 1.0)
    )
    # Deterministic across calls
    first = dataset.use_true_speaker_embedding_for_utterance("utt_abc", 0.5)
    second = dataset.use_true_speaker_embedding_for_utterance("utt_abc", 0.5)
    self.assertEqual(first, second)

  def test_concat_utterance_group_and_tf_dataset(self):
    rng = np.random.default_rng(123)
    utts = [self.utt1, self.utt2, self.utt1, self.utt2]
    grouped = dataset.concat_utterance_group(
        utts, min_utterances=2, max_utterances=2, rng=rng
    )
    self.assertEqual(len(grouped), 2)
    ds = dataset.create_tf_dataset(grouped, batch_size=2, shuffle=False)
    for batch_inputs, batch_labels, batch_mask in ds:
      self.assertEqual(batch_inputs["features"].shape, (2, 7, 8))
      self.assertEqual(batch_inputs["speaker_embedding"].shape, (2, 4))
      self.assertEqual(batch_labels.shape, (2, 7))
      self.assertEqual(batch_mask.shape, (2, 7))
      self.assertIn(
          int(configs.FrameLabel.SPEECH_FROM_NON_TARGET_SPEAKER),
          batch_labels.numpy(),
      )


if __name__ == "__main__":
  unittest.main()
