"""CLI script to prepare concatenated multi-speaker Personal VAD datasets.

Can either:
1. Load single-speaker utterances from an input `.npz` archive (containing
   `features` or `waveforms`, `labels`, `speaker_ids`, `speaker_embeddings`),
   extract acoustic features if needed, concatenate random groups of 1 to 3
   utterances from different speakers, and write the concatenated dataset to
   `--output_npz`.
2. Generate a synthetic multi-speaker dataset (`--generate_synthetic`) for
   end-to-end pipeline verification and unit testing.
"""

import argparse
from collections.abc import Sequence
import os
import sys
from typing import Optional
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import configs  # noqa: E402
from personal_vad import dataset  # noqa: E402


def generate_synthetic_single_speaker_utterances(
    num_utterances: int = 30,
    num_speakers: int = 6,
    min_frames: int = 20,
    max_frames: int = 40,
    feature_dim: int = 40,
    speaker_embedding_dim: int = 256,
    seed: int = 42,
) -> list[dataset.UtteranceData]:
  """Generates synthetic single-speaker utterances with distinct speaker IDs.

  Args:
    num_utterances: Total number of single-speaker utterances to generate.
    num_speakers: Number of distinct speakers in the pool.
    min_frames: Minimum number of frames per single utterance.
    max_frames: Maximum number of frames per single utterance.
    feature_dim: Acoustic feature dimension (40 for PVAD 1.0, 512 for PVAD 2.0).
    speaker_embedding_dim: Speaker d-vector dimension (256).
    seed: Random seed for reproducibility.

  Returns:
    List of single-speaker `UtteranceData` instances.
  """
  rng = np.random.default_rng(seed)
  speaker_embeddings = {}
  speaker_signatures = {}
  for spk_idx in range(num_speakers):
    spk_id = f"spk_{spk_idx:03d}"
    raw_emb = rng.standard_normal(speaker_embedding_dim).astype(np.float32)
    speaker_embeddings[spk_id] = dataset.l2_normalize_numpy(raw_emb)
    speaker_signatures[spk_id] = rng.standard_normal(
        feature_dim
    ).astype(np.float32)

  utterances: list[dataset.UtteranceData] = []
  speaker_ids = list(speaker_embeddings.keys())
  for i in range(num_utterances):
    spk_id = speaker_ids[i % num_speakers]
    num_frames = int(rng.integers(min_frames, max_frames + 1))
    # Construct silence margins at beginning/end and speech in the middle
    labels = np.zeros((num_frames,), dtype=np.int32)
    sil_head = max(2, num_frames // 6)
    sil_tail = max(2, num_frames // 6)
    labels[:sil_head] = int(configs.FrameLabel.SILENCE)
    labels[-sil_tail:] = int(configs.FrameLabel.SILENCE)

    speech_mask = (labels == int(configs.FrameLabel.SPEECH)).astype(
        np.float32
    )[:, None]
    noise = 0.25 * rng.standard_normal((num_frames, feature_dim)).astype(
        np.float32
    )
    sig = speaker_signatures[spk_id][None, :]
    features = speech_mask * (sig + 1.5) + (1.0 - speech_mask) * (-1.5) + noise

    utterances.append(
        dataset.UtteranceData(
            utt_id=f"utt_{i:04d}",
            speaker_id=spk_id,
            features=features.astype(np.float32),
            labels=labels,
            speaker_embedding=speaker_embeddings[spk_id],
        )
    )
  return utterances


def save_utterances_to_npz(
    utterances: Sequence[dataset.UtteranceData],
    output_path: str,
    max_frames: Optional[int] = None,
) -> dict[str, np.ndarray]:
  """Pads and saves a sequence of `UtteranceData` to a `.npz` file."""
  if not utterances:
    raise ValueError("Cannot save an empty utterance list.")

  if max_frames is None:
    max_frames = max(u.features.shape[0] for u in utterances)

  num_examples = len(utterances)
  feature_dim = utterances[0].features.shape[1]
  emb_dim = utterances[0].speaker_embedding.shape[0]

  features = np.zeros(
      (num_examples, max_frames, feature_dim), dtype=np.float32
  )
  speaker_embeddings = np.zeros((num_examples, emb_dim), dtype=np.float32)
  cosine_scores = np.zeros((num_examples, max_frames, 1), dtype=np.float32)
  labels = np.zeros((num_examples, max_frames), dtype=np.int32)
  masks = np.zeros((num_examples, max_frames), dtype=np.float32)
  lengths = np.zeros((num_examples,), dtype=np.int32)

  for i, utt in enumerate(utterances):
    length = min(utt.features.shape[0], max_frames)
    lengths[i] = length
    features[i, :length, :] = utt.features[:length]
    speaker_embeddings[i, :] = utt.speaker_embedding
    labels[i, :length] = utt.labels[:length]
    masks[i, :length] = 1.0
    if utt.cosine_scores is not None:
      cosine_scores[i, :length, :] = np.asarray(
          utt.cosine_scores[:length], dtype=np.float32
      ).reshape(length, 1)

  payload = {
      "features": features,
      "speaker_embeddings": speaker_embeddings,
      "cosine_scores": cosine_scores,
      "labels": labels,
      "masks": masks,
      "lengths": lengths,
  }
  parent_dir = os.path.dirname(output_path)
  if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)
  np.savez_compressed(output_path, **payload)
  return payload


def load_utterances_from_npz(npz_path: str) -> list[dataset.UtteranceData]:
  """Loads a list of `UtteranceData` from a saved `.npz` file."""
  with np.load(npz_path, allow_pickle=False) as data:
    features = data["features"]
    speaker_embeddings = data["speaker_embeddings"]
    labels = data["labels"]
    lengths = (
        data["lengths"]
        if "lengths" in data
        else np.full((features.shape[0],), features.shape[1], dtype=np.int32)
    )
    cosine_scores = data["cosine_scores"] if "cosine_scores" in data else None
    speaker_ids = (
        [str(s) for s in data["speaker_ids"]]
        if "speaker_ids" in data
        else [f"spk_{i}" for i in range(features.shape[0])]
    )

  utterances: list[dataset.UtteranceData] = []
  for i in range(features.shape[0]):
    length = int(lengths[i])
    cos = cosine_scores[i, :length] if cosine_scores is not None else None
    utterances.append(
        dataset.UtteranceData(
            utt_id=f"utt_{i:04d}",
            speaker_id=speaker_ids[i],
            features=features[i, :length].astype(np.float32),
            labels=labels[i, :length].astype(np.int32),
            speaker_embedding=speaker_embeddings[i].astype(np.float32),
            cosine_scores=cos,
        )
    )
  return utterances


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Prepare multi-speaker concatenated Personal VAD dataset."
  )
  parser.add_argument(
      "--input_npz",
      type=str,
      default=None,
      help="Path to input single-speaker utterances .npz file.",
  )
  parser.add_argument(
      "--output_npz",
      type=str,
      required=True,
      help="Path to output concatenated multi-speaker dataset .npz file.",
  )
  parser.add_argument(
      "--generate_synthetic",
      action="store_true",
      help="Generate synthetic single-speaker utterances before concatenation.",
  )
  parser.add_argument(
      "--num_utterances",
      type=int,
      default=60,
      help="Number of synthetic single-speaker utterances to generate.",
  )
  parser.add_argument(
      "--frontend_version",
      type=str,
      choices=["v1", "v2"],
      default="v1",
      help="Frontend configuration version (v1=40d, v2=512d).",
  )
  parser.add_argument(
      "--min_utterances",
      type=int,
      default=1,
      help="Minimum utterances per concatenated sequence.",
  )
  parser.add_argument(
      "--max_utterances",
      type=int,
      default=3,
      help="Maximum utterances per concatenated sequence.",
  )
  parser.add_argument(
      "--enrollment_less_prob",
      type=float,
      default=0.0,
      help=(
          "Probability p_0 of zeroing speaker embedding and mapping ntss->tss."
      ),
  )
  parser.add_argument(
      "--convert_for_vad_baseline",
      action="store_true",
      help="Convert 3-class labels to 2-class standard VAD labels.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed.",
  )
  return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
  args = parse_args(argv)
  fe_cfg = (
      configs.FrontendConfig.pvad_v1()
      if args.frontend_version == "v1"
      else configs.FrontendConfig.pvad_v2()
  )

  if args.generate_synthetic or args.input_npz is None:
    single_utterances = generate_synthetic_single_speaker_utterances(
        num_utterances=args.num_utterances,
        feature_dim=fe_cfg.output_dim,
        seed=args.seed,
    )
  else:
    single_utterances = load_utterances_from_npz(args.input_npz)

  rng = np.random.default_rng(args.seed)
  concat_utterances = dataset.concat_utterance_group(
      single_utterances,
      min_utterances=args.min_utterances,
      max_utterances=args.max_utterances,
      convert_for_vad_baseline=args.convert_for_vad_baseline,
      enrollment_less_prob=args.enrollment_less_prob,
      rng=rng,
  )
  save_utterances_to_npz(concat_utterances, args.output_npz)


if __name__ == "__main__":
  main()
