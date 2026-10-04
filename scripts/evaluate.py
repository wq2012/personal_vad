"""CLI script to evaluate Personal VAD models or precomputed score tables.

Computes per-class Accuracy, overall Accuracy, per-class Average Precision (AP),
micro-averaged mean Average Precision (mAP), ROC AUC, PR curves, ROC curves, and
generates an HTML report (`report.html`) and JSON summary (`metrics.json`).
"""

import argparse
from collections.abc import Sequence
import json
import os
import sys
from typing import Optional
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import eval_lib  # noqa: E402
from personal_vad import model as model_lib  # noqa: E402
from personal_vad import tflite_export  # noqa: E402
from scripts import prepare_dataset  # noqa: E402
from scripts import train  # noqa: E402


def collect_model_predictions(
    pvad_model: model_lib.PersonalVadModel,
    utterances: Sequence[prepare_dataset.dataset.UtteranceData],
) -> tuple[np.ndarray, np.ndarray]:
  """Runs inference with a Keras `PersonalVadModel` over `utterances`."""
  all_scores = []
  all_labels = []
  for utt in utterances:
    features = np.expand_dims(utt.features, axis=0)
    spk_emb = np.expand_dims(utt.speaker_embedding, axis=0)
    cos = (
        np.expand_dims(utt.cosine_scores.reshape(-1, 1), axis=0)
        if utt.cosine_scores is not None
        else None
    )
    probs = pvad_model.predict_proba(
        features, speaker_embedding=spk_emb, cosine_score=cos
    ).numpy()[0]
    all_scores.append(probs)
    all_labels.append(utt.labels)
  return (
      np.concatenate(all_scores, axis=0),
      np.concatenate(all_labels, axis=0),
  )


def collect_tflite_predictions(
    runner: tflite_export.TFLitePersonalVadRunner,
    utterances: Sequence[prepare_dataset.dataset.UtteranceData],
) -> tuple[np.ndarray, np.ndarray]:
  """Runs inference with a `TFLitePersonalVadRunner` over `utterances`."""
  all_scores = []
  all_labels = []
  for utt in utterances:
    probs = runner.predict(
        features=utt.features,
        speaker_embedding=utt.speaker_embedding,
        cosine_score=utt.cosine_scores,
    )
    all_scores.append(probs)
    all_labels.append(utt.labels)
  return (
      np.concatenate(all_scores, axis=0),
      np.concatenate(all_labels, axis=0),
  )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Evaluate Personal VAD model predictions."
  )
  parser.add_argument(
      "--eval_npz",
      type=str,
      default=None,
      help="Path to evaluation dataset .npz file.",
  )
  parser.add_argument(
      "--checkpoint_dir",
      type=str,
      default=None,
      help="Path to trained Keras checkpoint directory.",
  )
  parser.add_argument(
      "--tflite_path",
      type=str,
      default=None,
      help="Path to exported .tflite model file.",
  )
  parser.add_argument(
      "--scores_npz",
      type=str,
      default=None,
      help="Optional path to .npz with precomputed `scores` and `labels`.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Directory to write metrics.json, report.html, and curve plots.",
  )
  parser.add_argument(
      "--eval_mode",
      type=str,
      choices=["personal_vad", "personal_vad_std_eval", "std_vad_std_eval"],
      default="personal_vad",
      help="Evaluation mode.",
  )
  parser.add_argument(
      "--sc_baseline",
      action="store_true",
      help="Apply Score Combination (SC) baseline conversion to scores.",
  )
  return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> eval_lib.EvaluationMetrics:
  args = parse_args(argv)

  if args.scores_npz is not None:
    with np.load(args.scores_npz, allow_pickle=False) as data:
      scores = data["scores"]
      labels = data["labels"]
  elif args.eval_npz is not None:
    utterances = prepare_dataset.load_utterances_from_npz(args.eval_npz)
    if args.tflite_path is not None:
      runner = tflite_export.TFLitePersonalVadRunner(args.tflite_path)
      scores, labels = collect_tflite_predictions(runner, utterances)
    elif args.checkpoint_dir is not None:
      pvad_model = train.load_checkpoint(args.checkpoint_dir)
      scores, labels = collect_model_predictions(pvad_model, utterances)
    else:
      raise ValueError(
          "Must specify either --checkpoint_dir or --tflite_path when "
          "evaluating with --eval_npz."
      )
  else:
    raise ValueError("Must specify either --scores_npz or --eval_npz.")

  evaluator = eval_lib.PersonalVadEvaluator(eval_mode=args.eval_mode)
  result = evaluator.evaluate(
      scores=scores,
      labels=labels,
      sc_baseline=args.sc_baseline,
      output_dir=args.output_dir,
  )
  metrics_json_path = os.path.join(args.output_dir, "metrics.json")
  with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(result.to_dict(), f, indent=2)
  return result


if __name__ == "__main__":
  main()
