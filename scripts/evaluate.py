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
    batch_size: int = 64,
    include_cosine_column: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """Runs batched inference with a `PersonalVadModel` over `utterances`."""
  if not utterances:
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

  max_frames = max(u.features.shape[0] for u in utterances)
  num_utts = len(utterances)
  feat_dim = utterances[0].features.shape[1]
  emb_dim = utterances[0].speaker_embedding.shape[0]
  has_cos = any(u.cosine_scores is not None for u in utterances)

  features_pad = np.zeros((num_utts, max_frames, feat_dim), dtype=np.float32)
  spk_pad = np.zeros((num_utts, emb_dim), dtype=np.float32)
  cos_pad = (
      np.zeros((num_utts, max_frames, 1), dtype=np.float32)
      if has_cos
      else None
  )
  for i, utt in enumerate(utterances):
    length = utt.features.shape[0]
    features_pad[i, :length, :] = utt.features
    spk_pad[i, :] = utt.speaker_embedding
    if cos_pad is not None and utt.cosine_scores is not None:
      cos_pad[i, :length, :] = utt.cosine_scores.reshape(-1, 1)[:length]

  all_scores = []
  all_labels = []
  for start in range(0, num_utts, batch_size):
    end = min(start + batch_size, num_utts)
    b_feat = features_pad[start:end]
    b_spk = spk_pad[start:end]
    b_cos = cos_pad[start:end] if cos_pad is not None else None
    probs = pvad_model.predict_proba(
        b_feat, speaker_embedding=b_spk, cosine_score=b_cos
    ).numpy()
    for j, utt in enumerate(utterances[start:end]):
      length = utt.features.shape[0]
      utt_probs = probs[j, :length]
      if (
          include_cosine_column
          and utt_probs.shape[1] == 2
          and utt.cosine_scores is not None
      ):
        utt_cos = utt.cosine_scores.reshape(-1, 1)[:length]
        utt_probs = np.concatenate([utt_probs, utt_cos], axis=1)
      all_scores.append(utt_probs)
      all_labels.append(utt.labels)

  return (
      np.concatenate(all_scores, axis=0),
      np.concatenate(all_labels, axis=0),
  )


def collect_tflite_predictions(
    runner: tflite_export.TFLitePersonalVadRunner,
    utterances: Sequence[prepare_dataset.dataset.UtteranceData],
    include_cosine_column: bool = False,
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
    if (
        include_cosine_column
        and probs.shape[1] == 2
        and utt.cosine_scores is not None
    ):
      utt_cos = utt.cosine_scores.reshape(-1, 1)[: probs.shape[0]]
      probs = np.concatenate([probs, utt_cos], axis=1)
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
      scores, labels = collect_tflite_predictions(
          runner, utterances, include_cosine_column=args.sc_baseline
      )
    elif args.checkpoint_dir is not None:
      pvad_model = train.load_checkpoint(args.checkpoint_dir)
      scores, labels = collect_model_predictions(
          pvad_model, utterances, include_cosine_column=args.sc_baseline
      )
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
