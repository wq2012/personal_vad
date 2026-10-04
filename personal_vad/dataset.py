"""Dataset preparation and utterance concatenation for Personal VAD.

Implements:
- Multi-speaker utterance concatenation (`concat_utterance_sequence` and
  `concat_utterance_group`) for constructing training and evaluation examples
  (Personal VAD 1.0, arXiv:1908.04284, Section 3.1).
- Standard 2-class VAD label conversion (`convert_alignment_for_vad_baseline`).
- Deterministic speaker embedding selection
  (`use_true_speaker_embedding_for_utterance`).
- Personal VAD 2.0 Algorithm 1 (arXiv:2204.03793): joint enrolled and
  enrollment-less conditioning where, with probability `p_0 = 0.2`, the target
  speaker d-vector is replaced by a zero vector and `ntss` (2) labels are
  mapped to `tss` (0).
"""

from collections.abc import Sequence
import dataclasses
import hashlib
from typing import Any, Optional
import numpy as np
from . import configs

try:
  import tensorflow as tf
except ImportError:  # pragma: no cover
  tf = None  # type: ignore[assignment]


@dataclasses.dataclass
class UtteranceData:
  """Represents a single speaker's utterance or concatenated multi-speaker clip.

  Attributes:
    utt_id: Unique utterance identifier.
    speaker_id: Speaker identifier string (or target speaker ID after
      concatenation).
    features: Float32 acoustic features of shape `[num_frames, feature_dim]`.
    labels: Int32 frame-level labels of shape `[num_frames]` using
      `configs.FrameLabel` (`0=SPEECH/tss`, `1=SILENCE/ns`,
      `2=SPEECH_FROM_NON_TARGET_SPEAKER/ntss`).
    speaker_embedding: Float32 target speaker d-vector of shape
      `[speaker_embedding_dim]`.
    frame_speaker_embeddings: Optional float32 frame-level speaker embeddings of
      shape `[num_frames, speaker_embedding_dim]` (used to compute framewise
      cosine similarity scores `s_t` in `SC`, `ST`, and `SET` modes).
    cosine_scores: Optional float32 frame-level cosine similarity scores `s_t`
      of shape `[num_frames, 1]` or `[num_frames]`.
  """

  utt_id: str
  speaker_id: str
  features: np.ndarray
  labels: np.ndarray
  speaker_embedding: np.ndarray
  frame_speaker_embeddings: Optional[np.ndarray] = None
  cosine_scores: Optional[np.ndarray] = None


def l2_normalize_numpy(
    vec: np.ndarray, axis: int = -1, eps: float = 1e-12
) -> np.ndarray:
  """L2-normalizes a NumPy array safely (preserving zero vectors)."""
  vec = np.asarray(vec, dtype=np.float32)
  norm = np.linalg.norm(vec, axis=axis, keepdims=True)
  return np.where(norm > eps, vec / np.maximum(norm, eps), 0.0)


def compute_frame_cosine_similarity(
    frame_embeddings: np.ndarray,
    target_embedding: np.ndarray,
) -> np.ndarray:
  """Computes frame-level cosine similarity `cos(e_t, e_target)`.

  Args:
    frame_embeddings: Array of shape `[num_frames, embedding_dim]`.
    target_embedding: Array of shape `[embedding_dim]`.

  Returns:
    Float32 array of shape `[num_frames, 1]` with values in `[-1.0, 1.0]`
    (or `0.0` if either vector is all zeros).
  """
  norm_frames = l2_normalize_numpy(frame_embeddings, axis=-1)
  norm_target = l2_normalize_numpy(target_embedding, axis=-1).reshape(-1, 1)
  return np.matmul(norm_frames, norm_target).astype(np.float32)


def use_true_speaker_embedding_for_utterance(
    utt_id: str, true_speaker_prob: float = 0.5
) -> bool:
  """Deterministically decides whether to use the true speaker's embedding.

  Hashes `utt_id` to a uniform value in `[0, 1)` and returns `True` if it is
  less than `true_speaker_prob`.

  Args:
    utt_id: Unique utterance key string.
    true_speaker_prob: Probability in `[0.0, 1.0]` of selecting the true speaker
      embedding.

  Returns:
    Boolean indicating whether to use the true speaker's embedding.

  Raises:
    ValueError: If `true_speaker_prob` is outside `[0.0, 1.0]`.
  """
  if not 0.0 <= true_speaker_prob <= 1.0:
    raise ValueError(
        f"true_speaker_prob must be in [0, 1], got {true_speaker_prob}"
    )
  digest = hashlib.sha256(utt_id.encode("utf-8")).digest()
  hash_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
  uniform_val = float(hash_int % 1_000_000) / 1_000_000.0
  return uniform_val < true_speaker_prob


def convert_alignment_for_vad_baseline(labels: np.ndarray) -> np.ndarray:
  """Converts 3-class Personal VAD labels to 2-class Standard VAD labels.

  Maps `SPEECH_FROM_NON_TARGET_SPEAKER (2)` to `SPEECH (0)` while keeping
  `SILENCE (1)` unchanged.

  Args:
    labels: Integer NumPy array of frame labels.

  Returns:
    Integer NumPy array of the same shape with values in `{0, 1}`.
  """
  labels = np.asarray(labels, dtype=np.int32)
  return np.where(
      labels == int(configs.FrameLabel.SPEECH_FROM_NON_TARGET_SPEAKER),
      int(configs.FrameLabel.SPEECH),
      labels,
  )


def apply_enrollment_less_conditioning(
    speaker_embedding: np.ndarray,
    labels: np.ndarray,
    enrollment_less_prob: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
  """Applies joint enrolled / enrollment-less conditioning (PVAD 2.0 Alg. 1).

  In Personal VAD 2.0 (arXiv:2204.03793, Section 2.3, Algorithm 1), with
  probability `p_0 = 0.2`, the enrolled speaker d-vector `e` is replaced by a
  zero vector `0`, and ground-truth labels `ntss` (2) are replaced by `tss` (0).
  This trains a single unified model that operates as Personal VAD when a target
  speaker d-vector is provided and seamlessly falls back to Standard VAD when a
  zero vector is passed.

  Args:
    speaker_embedding: Float32 array of shape `[speaker_embedding_dim]`.
    labels: Int32 array of shape `[num_frames]`.
    enrollment_less_prob: Probability `p_0` in `[0.0, 1.0]` of dropping the
      speaker enrollment.
    rng: Optional NumPy random Generator.

  Returns:
    Tuple `(new_speaker_embedding, new_labels, was_dropped)`.
  """
  if not 0.0 <= enrollment_less_prob <= 1.0:
    raise ValueError(
        "enrollment_less_prob must be in [0, 1], got "
        f"{enrollment_less_prob}"
    )
  speaker_embedding = np.asarray(speaker_embedding, dtype=np.float32)
  labels = np.asarray(labels, dtype=np.int32)
  if enrollment_less_prob <= 0.0:
    return speaker_embedding.copy(), labels.copy(), False

  if rng is None:
    rng = np.random.default_rng()

  if float(rng.uniform(0.0, 1.0)) < enrollment_less_prob:
    zero_emb = np.zeros_like(speaker_embedding, dtype=np.float32)
    converted_labels = convert_alignment_for_vad_baseline(labels)
    return zero_emb, converted_labels, True
  return speaker_embedding.copy(), labels.copy(), False


def concat_utterance_sequence(
    utterances: Sequence[UtteranceData],
    target_index: int,
    convert_for_vad_baseline: bool = False,
) -> UtteranceData:
  """Concatenates a sequence of utterances and assigns 3-class PVAD labels.

  Label assignment rules:
  - The speaker at `utterances[target_index]` is chosen as the target speaker.
  - For any constituent utterance whose `speaker_id` matches the target
    speaker's `speaker_id`, frames with label `SPEECH (0)` remain `SPEECH (0)`.
  - For constituent utterances from a different speaker, frames with label
    `SPEECH (0)` are relabeled to `SPEECH_FROM_NON_TARGET_SPEAKER (2)` (unless
    `convert_for_vad_baseline=True`, in which case they remain `SPEECH (0)`).
  - Frames with label `SILENCE (1)` remain `SILENCE (1)`.

  Args:
    utterances: Non-empty sequence of `UtteranceData` objects.
    target_index: Index in `[0, len(utterances))` of the target speaker.
    convert_for_vad_baseline: Whether to keep non-target speech labeled as
      `SPEECH (0)` for baseline 2-class VAD training.

  Returns:
    A new `UtteranceData` representing the concatenated multi-speaker utterance.

  Raises:
    ValueError: If `utterances` is empty or `target_index` is out of range.
  """
  if not utterances:
    raise ValueError("utterances must be non-empty.")
  if not 0 <= target_index < len(utterances):
    raise ValueError(
        f"target_index {target_index} out of range for {len(utterances)} "
        "utterances."
    )

  target_utt = utterances[target_index]
  target_speaker_id = target_utt.speaker_id
  target_embedding = l2_normalize_numpy(target_utt.speaker_embedding)

  features_list = []
  labels_list = []
  frame_embs_list = []
  has_all_frame_embs = all(
      u.frame_speaker_embeddings is not None for u in utterances
  )
  utt_ids = []

  for utt in utterances:
    utt_ids.append(utt.utt_id)
    feat = np.asarray(utt.features, dtype=np.float32)
    lbl = np.asarray(utt.labels, dtype=np.int32)
    if feat.shape[0] != lbl.shape[0]:
      raise ValueError(
          f"Utterance {utt.utt_id} has {feat.shape[0]} feature frames but "
          f"{lbl.shape[0]} label frames."
      )
    features_list.append(feat)

    is_target_speaker = utt.speaker_id == target_speaker_id
    if is_target_speaker or convert_for_vad_baseline:
      relabeled = np.where(
          lbl == int(configs.FrameLabel.SPEECH_FROM_NON_TARGET_SPEAKER),
          int(configs.FrameLabel.SPEECH),
          lbl,
      )
    else:
      relabeled = np.where(
          lbl == int(configs.FrameLabel.SPEECH),
          int(configs.FrameLabel.SPEECH_FROM_NON_TARGET_SPEAKER),
          lbl,
      )
    labels_list.append(relabeled)

    if has_all_frame_embs and utt.frame_speaker_embeddings is not None:
      frame_embs_list.append(
          np.asarray(utt.frame_speaker_embeddings, dtype=np.float32)
      )
    else:
      # Broadcast utterance-level speaker embedding to every frame so cosine
      # similarity can still be computed for ST/SET/SC modes.
      tiled_emb = np.tile(
          l2_normalize_numpy(utt.speaker_embedding).reshape(1, -1),
          (feat.shape[0], 1),
      )
      frame_embs_list.append(tiled_emb)

  concat_features = np.concatenate(features_list, axis=0)
  concat_labels = np.concatenate(labels_list, axis=0)
  concat_frame_embs = np.concatenate(frame_embs_list, axis=0)
  cosine_scores = compute_frame_cosine_similarity(
      concat_frame_embs, target_embedding
  )

  return UtteranceData(
      utt_id="+".join(utt_ids),
      speaker_id=target_speaker_id,
      features=concat_features,
      labels=concat_labels,
      speaker_embedding=target_embedding,
      frame_speaker_embeddings=concat_frame_embs,
      cosine_scores=cosine_scores,
  )


def concat_utterance_group(
    utterances: Sequence[UtteranceData],
    min_utterances: int = 1,
    max_utterances: int = 3,
    convert_for_vad_baseline: bool = False,
    enrollment_less_prob: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> list[UtteranceData]:
  """Randomly groups and concatenates utterances into multi-speaker sequences.

  For each group:
  1. Draws a group size `n ~ Uniform({min_utterances, ..., max_utterances})`.
  2. Draws a random `target_index ~ Uniform({0, ..., n - 1})`.
  3. Concatenates the `n` utterances via `concat_utterance_sequence`.
  4. Optionally applies enrollment-less conditioning with probability
     `enrollment_less_prob` (Algorithm 1 in Personal VAD 2.0).

  Args:
    utterances: Sequence of single-speaker `UtteranceData` objects.
    min_utterances: Minimum number of utterances per concatenated sequence.
    max_utterances: Maximum number of utterances per concatenated sequence.
    convert_for_vad_baseline: Whether to map `ntss` (2) to `tss` (0).
    enrollment_less_prob: Probability of zeroing the target speaker embedding
      and mapping `ntss` (2) to `tss` (0).
    rng: Optional NumPy random Generator for reproducible sampling.

  Returns:
    List of concatenated `UtteranceData` examples.
  """
  if min_utterances < 1 or max_utterances < min_utterances:
    raise ValueError(
        f"Invalid (min_utterances={min_utterances}, "
        f"max_utterances={max_utterances})."
    )
  if rng is None:
    rng = np.random.default_rng()

  outputs: list[UtteranceData] = []
  idx = 0
  total = len(utterances)
  while idx < total:
    group_size = int(rng.integers(min_utterances, max_utterances + 1))
    group = utterances[idx : min(idx + group_size, total)]
    idx += len(group)

    target_index = int(rng.integers(0, len(group)))
    concat_utt = concat_utterance_sequence(
        group,
        target_index=target_index,
        convert_for_vad_baseline=convert_for_vad_baseline,
    )
    if enrollment_less_prob > 0.0:
      new_emb, new_labels, dropped = apply_enrollment_less_conditioning(
          concat_utt.speaker_embedding,
          concat_utt.labels,
          enrollment_less_prob=enrollment_less_prob,
          rng=rng,
      )
      concat_utt.speaker_embedding = new_emb
      concat_utt.labels = new_labels
      if dropped and concat_utt.cosine_scores is not None:
        concat_utt.cosine_scores = np.zeros_like(
            concat_utt.cosine_scores, dtype=np.float32
        )
    outputs.append(concat_utt)

  return outputs


def create_tf_dataset(
    utterances: Sequence[UtteranceData],
    batch_size: int = 16,
    max_frames: Optional[int] = None,
    shuffle: bool = True,
    enrollment_less_prob: float = 0.0,
    seed: Optional[int] = None,
) -> Any:
  """Creates a batched `tf.data.Dataset` from a sequence of `UtteranceData`.

  Each element in the returned dataset is a tuple `(inputs_dict, labels, mask)`
  where:
    - `inputs_dict`:
      - `"features"`: float32 tensor `[batch, time, feature_dim]`
      - `"speaker_embedding"`: float32 tensor `[batch, speaker_embedding_dim]`
      - `"cosine_score"`: float32 tensor `[batch, time, 1]`
    - `labels`: int32 tensor `[batch, time]`
    - `mask`: float32 tensor `[batch, time]` (`1.0` for valid frames, `0.0` for
      padded frames).

  Args:
    utterances: Sequence of `UtteranceData` examples.
    batch_size: Batch size.
    max_frames: Optional maximum frame length. If None, uses the maximum length
      across `utterances`.
    shuffle: Whether to shuffle examples each epoch.
    enrollment_less_prob: Optional dynamic enrollment-less conditioning
      probability applied per example.
    seed: Optional random seed.

  Returns:
    A batched `tf.data.Dataset`.
  """
  if tf is None:
    raise ImportError("TensorFlow is required for create_tf_dataset.")
  if not utterances:
    raise ValueError("Cannot create a dataset from an empty utterance list.")

  if max_frames is None:
    max_frames = max(u.features.shape[0] for u in utterances)

  feature_dim = utterances[0].features.shape[1]
  emb_dim = utterances[0].speaker_embedding.shape[0]
  num_examples = len(utterances)

  all_features = np.zeros(
      (num_examples, max_frames, feature_dim), dtype=np.float32
  )
  all_embeddings = np.zeros((num_examples, emb_dim), dtype=np.float32)
  all_cosine_scores = np.zeros((num_examples, max_frames, 1), dtype=np.float32)
  all_labels = np.zeros((num_examples, max_frames), dtype=np.int32)
  all_masks = np.zeros((num_examples, max_frames), dtype=np.float32)

  rng = np.random.default_rng(seed)
  for i, utt in enumerate(utterances):
    length = min(utt.features.shape[0], max_frames)
    all_features[i, :length, :] = utt.features[:length]
    emb, lbl, dropped = apply_enrollment_less_conditioning(
        utt.speaker_embedding,
        utt.labels[:length],
        enrollment_less_prob=enrollment_less_prob,
        rng=rng,
    )
    all_embeddings[i, :] = emb
    all_labels[i, :length] = lbl
    all_masks[i, :length] = 1.0
    if utt.cosine_scores is not None and not dropped:
      cos = np.asarray(utt.cosine_scores[:length], dtype=np.float32).reshape(
          length, 1
      )
      all_cosine_scores[i, :length, :] = cos

  ds = tf.data.Dataset.from_tensor_slices((
      {
          "features": all_features,
          "speaker_embedding": all_embeddings,
          "cosine_score": all_cosine_scores,
      },
      all_labels,
      all_masks,
  ))
  if shuffle:
    ds = ds.shuffle(buffer_size=num_examples, seed=seed)
  return ds.batch(batch_size)
