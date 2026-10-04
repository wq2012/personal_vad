"""Tests for personal_vad.eval_lib."""

import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import eval_lib  # noqa: E402


class EvalLibTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.scores = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.3, 0.3, 0.4],
            [0.35, 0.45, 0.2],
            [0.2, 0.5, 0.3],
        ],
        dtype=np.float32,
    )
    self.labels = np.asarray([0, 2, 0, 1], dtype=np.int32)

  def test_personal_vad_accuracy_and_ap(self):
    evaluator = eval_lib.PersonalVadEvaluator(eval_mode="personal_vad")
    result = evaluator.evaluate(self.scores, self.labels)
    self.assertEqual(result.num_frames, 4)
    self.assertIsNotNone(result.class_accuracies)
    self.assertAlmostEqual(result.class_accuracies["Speech (tss)"], 0.5)
    self.assertAlmostEqual(result.class_accuracies["Silence (ns)"], 1.0)
    self.assertAlmostEqual(
        result.class_accuracies["Non-Target Speech (ntss)"], 1.0
    )
    self.assertAlmostEqual(result.overall_accuracy, 0.75)

    self.assertIsNotNone(result.average_precisions)
    self.assertAlmostEqual(result.average_precisions["Speech (tss)"], 1.0)
    self.assertAlmostEqual(result.average_precisions["Silence (ns)"], 1.0)
    self.assertAlmostEqual(
        result.average_precisions["Non-Target Speech (ntss)"], 1.0
    )
    self.assertAlmostEqual(result.mean_average_precision_micro, 0.8875)

    self.assertIsNotNone(result.roc_aucs)
    for cls_name in evaluator.class_names:
      self.assertAlmostEqual(result.roc_aucs[cls_name], 1.0)

  def test_std_vad_eval_modes(self):
    for mode in ("personal_vad_std_eval", "std_vad_std_eval"):
      evaluator = eval_lib.PersonalVadEvaluator(eval_mode=mode)
      result = evaluator.evaluate(self.scores, self.labels)
      self.assertEqual(result.eval_mode, mode)
      self.assertIn("Speech", result.average_precisions)
      self.assertIn("Silence", result.average_precisions)
      self.assertGreater(result.mean_average_precision_micro, 0.5)

  def test_sc_baseline_conversion_and_report_files(self):
    sc_input = np.asarray(
        [
            [0.9, 0.1, 0.8],
            [0.85, 0.15, -0.7],
            [0.9, 0.1, 0.75],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float32,
    )
    converted = eval_lib.convert_scores_for_sc_baseline(sc_input)
    self.assertEqual(converted.shape, (4, 3))
    # Row 0 has high speech score and high cosine similarity -> high tss score
    self.assertGreater(converted[0, 0], converted[0, 2])
    # Row 1 has high speech score and low cosine similarity -> high ntss score
    self.assertGreater(converted[1, 2], converted[1, 0])

    with tempfile.TemporaryDirectory() as tmpdir:
      evaluator = eval_lib.PersonalVadEvaluator(eval_mode="personal_vad")
      result = evaluator.evaluate(
          sc_input, self.labels, sc_baseline=True, output_dir=tmpdir
      )
      self.assertTrue(os.path.exists(os.path.join(tmpdir, "roc_curves.png")))
      self.assertTrue(os.path.exists(os.path.join(tmpdir, "pr_curves.png")))
      self.assertTrue(os.path.exists(os.path.join(tmpdir, "report.html")))
      self.assertIn("overall_accuracy", result.to_dict())


if __name__ == "__main__":
  unittest.main()
