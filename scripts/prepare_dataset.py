"""CLI script to prepare concatenated multi-speaker Personal VAD datasets.

Supports three data sources:
1. Multilingual LibriSpeech / LibriSpeech audio manifests (`--mls_manifest`):
   loads 16 kHz `.flac`/`.wav` utterances, extracts 40-dim (`v1`) or 512-dim
   (`v2` 4x3 stacked + subsampled) log-Mel filterbank energies, computes
   energy-calibrated frame-level VAD labels (`speech=0`, `silence=1`), fits or
   applies a regularized LDA/WCCN open-set speaker embedding subspace (rank 8
   tiled to 256-dim d-vectors), computes frame-level cosine similarity scores,
   and concatenates random groups of utterances across speakers.
2. Pre-extracted single-speaker `.npz` archives (`--input_npz`).
3. Synthetic multi-speaker utterances (`--generate_synthetic`) for unit tests.
"""

import argparse
from collections.abc import Sequence
import json
import os
import sys
from typing import Any, Optional
import numpy as np
from scipy import linalg
from scipy import ndimage
import soundfile as sf

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import configs  # noqa: E402
from personal_vad import dataset  # noqa: E402
from personal_vad import frontend as frontend_lib  # noqa: E402


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
    speaker_signatures[spk_id] = rng.standard_normal(feature_dim).astype(
        np.float32
    )

  utterances: list[dataset.UtteranceData] = []
  speaker_ids = list(speaker_embeddings.keys())
  for i in range(num_utterances):
    spk_id = speaker_ids[i % num_speakers]
    num_frames = int(rng.integers(min_frames, max_frames + 1))
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


def compute_frame_vad_labels(mel_40: np.ndarray) -> np.ndarray:
  """Computes frame-level binary VAD labels (`speech=0`, `silence=1`).

  Uses adaptive dynamic-range energy thresholding with median smoothing to
  capture natural intra-utterance pauses and leading/trailing silence margins.

  Args:
    mel_40: Log-Mel filterbank energies of shape `[num_frames, 40]`.

  Returns:
    Integer array of shape `[num_frames]` with `0` for speech and `1` for
    silence.
  """
  frame_energy = np.mean(mel_40, axis=-1)
  p10 = float(np.percentile(frame_energy, 10))
  p90 = float(np.percentile(frame_energy, 90))
  threshold = p10 + 0.28 * (p90 - p10)
  is_speech = frame_energy >= threshold
  is_speech = ndimage.median_filter(is_speech.astype(np.int32), size=5).astype(
      bool
  )
  return np.where(
      is_speech,
      int(configs.FrameLabel.SPEECH),
      int(configs.FrameLabel.SILENCE),
  ).astype(np.int32)


def extract_sliding_speaker_stats(
    mel_40: np.ndarray, window: int = 35
) -> np.ndarray:
  """Computes sliding-window spectral mean, std, and first-order delta stats.

  Args:
    mel_40: Log-Mel filterbank energies of shape `[T, 40]`.
    window: Causal/local sliding window size in frames.

  Returns:
    Float32 array of shape `[T, 120]`.
  """
  delta = np.zeros_like(mel_40)
  if mel_40.shape[0] > 1:
    delta[1:] = mel_40[1:] - mel_40[:-1]
  mean_f = ndimage.uniform_filter1d(
      mel_40, size=window, axis=0, mode="nearest"
  )
  sq_f = ndimage.uniform_filter1d(
      mel_40**2, size=window, axis=0, mode="nearest"
  )
  std_f = np.sqrt(np.maximum(sq_f - mean_f**2, 1e-4))
  d_sq = ndimage.uniform_filter1d(
      delta**2, size=window, axis=0, mode="nearest"
  )
  d_std = np.sqrt(np.maximum(d_sq, 1e-4))
  return np.concatenate([mean_f, std_f, d_std], axis=-1).astype(np.float32)


def fit_lda_wccn_subspace(
    records_with_mel: Sequence[dict[str, Any]],
    subspace_rank: int = 8,
    output_dim: int = 256,
) -> dict[str, np.ndarray]:
  """Fits a regularized LDA + WCCN open-set speaker subspace on training audio.

  Projects 120-dim sliding-window acoustic statistics onto the top
  `subspace_rank` generalized eigenvectors of $(S_B, S_W + \\lambda I)$,
  followed by within-class covariance whitening (WCCN) and orthogonal tiling to
  `output_dim` (256-dim d-vectors).

  Args:
    records_with_mel: Sequence of dicts containing `'speaker_id'`, `'mel_40'`,
      and `'vad_40'` from the training split only.
    subspace_rank: Principal discriminant rank $K$ (default 8).
    output_dim: Target speaker embedding dimension (default 256).

  Returns:
    Dictionary containing `'mean'`, `'std'`, `'proj_k'`, and `'subspace_rank'`.
  """
  spk_vecs: dict[str, list[np.ndarray]] = {}
  all_vecs: list[np.ndarray] = []
  for r in records_with_mel:
    mel_src = r.get("mel_40_full", r["mel_40"])
    vad_src = r.get("vad_40_full", r["vad_40"])
    raw = extract_sliding_speaker_stats(mel_src, window=45)
    speech_idx = np.where(vad_src == 0)[0]
    if len(speech_idx) < 5:
      speech_idx = np.arange(len(raw))
    pooled = np.mean(raw[speech_idx], axis=0)
    spk_vecs.setdefault(r["speaker_id"], []).append(pooled)
    for k in range(0, len(speech_idx), 15):
      chunk = speech_idx[k:k + 30]
      if len(chunk) >= 5:
        vec = np.mean(raw[chunk], axis=0)
        spk_vecs[r["speaker_id"]].append(vec)
        all_vecs.append(vec)

  x_all = np.stack(all_vecs, axis=0)
  feat_mean = np.mean(x_all, axis=0)
  feat_std = np.std(x_all, axis=0) + 1e-5

  dim = x_all.shape[1]
  sw = np.zeros((dim, dim), dtype=np.float64)
  sb = np.zeros((dim, dim), dtype=np.float64)
  global_mean = np.zeros((dim,), dtype=np.float64)
  total_n = 0
  spk_means: dict[str, np.ndarray] = {}
  spk_counts: dict[str, int] = {}

  for spk, vecs in spk_vecs.items():
    v_norm = (np.stack(vecs, axis=0) - feat_mean) / feat_std
    spk_means[spk] = np.mean(v_norm, axis=0)
    spk_counts[spk] = len(v_norm)
    global_mean += np.sum(v_norm, axis=0)
    total_n += len(v_norm)
  global_mean /= max(total_n, 1)

  for spk, vecs in spk_vecs.items():
    v_norm = (np.stack(vecs, axis=0) - feat_mean) / feat_std
    diff = v_norm - spk_means[spk][None, :]
    sw += diff.T @ diff
    m_diff = (spk_means[spk] - global_mean)[:, None]
    sb += spk_counts[spk] * (m_diff @ m_diff.T)

  sw /= max(total_n, 1)
  sb /= max(total_n, 1)
  sw_reg = sw + 0.10 * np.eye(dim)

  eigvals, eigvecs = linalg.eigh(sb, sw_reg)
  idx = np.argsort(eigvals)[::-1]
  eigvecs = eigvecs[:, idx]
  w_lda = eigvecs[:, :subspace_rank].astype(np.float32)

  wccn = np.zeros((subspace_rank, subspace_rank), dtype=np.float64)
  for spk, vecs in spk_vecs.items():
    v_norm = (np.stack(vecs, axis=0) - feat_mean) / feat_std
    proj = (v_norm @ w_lda) / (
        np.linalg.norm(v_norm @ w_lda, axis=-1, keepdims=True) + 1e-8
    )
    pm = np.mean(proj, axis=0, keepdims=True)
    d = proj - pm
    wccn += (d.T @ d) / max(len(proj), 1)
  wccn /= max(len(spk_vecs), 1)
  wccn_reg = wccn + 0.08 * np.eye(subspace_rank)
  b_wccn = np.linalg.cholesky(np.linalg.inv(wccn_reg)).astype(np.float32)

  proj_k = (w_lda @ b_wccn).astype(np.float32)
  all_mel_v1 = np.concatenate([r["mel_40"] for r in records_with_mel], axis=0)
  mel_mean_v1 = np.mean(all_mel_v1, axis=0).astype(np.float32)
  mel_std_v1 = (np.std(all_mel_v1, axis=0) + 1e-5).astype(np.float32)
  if "feat_v2" in records_with_mel[0]:
    all_mel_v2 = np.concatenate(
        [r["feat_v2"] for r in records_with_mel], axis=0
    )
    mel_mean_v2 = np.mean(all_mel_v2, axis=0).astype(np.float32)
    mel_std_v2 = (np.std(all_mel_v2, axis=0) + 1e-5).astype(np.float32)
  else:
    mel_mean_v2 = np.zeros((512,), dtype=np.float32)
    mel_std_v2 = np.ones((512,), dtype=np.float32)

  return {
      "mean": feat_mean.astype(np.float32),
      "std": feat_std.astype(np.float32),
      "proj_k": proj_k,
      "subspace_rank": np.array(subspace_rank, dtype=np.int32),
      "output_dim": np.array(output_dim, dtype=np.int32),
      "mel_mean_v1": mel_mean_v1,
      "mel_std_v1": mel_std_v1,
      "mel_mean_v2": mel_mean_v2,
      "mel_std_v2": mel_std_v2,
  }


def project_speaker_frames(
    mel_40: np.ndarray,
    subspace: dict[str, np.ndarray],
    window: int = 35,
) -> np.ndarray:
  """Projects `[T, 40]` log-Mel frames into `[T, 256]` normalized d-vectors."""
  raw = extract_sliding_speaker_stats(mel_40, window=window)
  x = (raw - subspace["mean"][None, :]) / subspace["std"][None, :]
  z_k = x @ subspace["proj_k"]
  z_k = z_k / (np.linalg.norm(z_k, axis=-1, keepdims=True) + 1e-8)
  repeats = int(subspace["output_dim"]) // z_k.shape[1]
  tiled = np.tile(z_k, (1, repeats)) / np.sqrt(float(repeats))
  return tiled.astype(np.float32)


def extract_subspace_conditioned_features(
    mel_40_full: np.ndarray,
    mel_40: np.ndarray,
    feat_v2: np.ndarray,
    idx_v2: np.ndarray,
    subspace: dict[str, np.ndarray],
    frontend_version: str = "v1",
) -> tuple[np.ndarray, np.ndarray]:
  """Extracts normalized acoustic features and frame d-vectors using `subspace`.

  Computes causal/sliding-window frame speaker embeddings purely from
  `mel_40_full` and blends the principal $K$-dim speaker subspace with a
  continuous acoustic voicing gate into the feature blocks while keeping the
  trailing spectral channels unrotated for voice activity detection.
  """
  f_embs_local = project_speaker_frames(mel_40_full, subspace, window=35)[
      idx_v2
  ]
  f_embs_wide = project_speaker_frames(mel_40_full, subspace, window=95)[idx_v2]
  frame_embs = dataset.l2_normalize_numpy(
      0.65 * f_embs_wide + 0.35 * f_embs_local
  )

  subspace_rank = int(subspace["subspace_rank"])
  output_dim = int(subspace["output_dim"])
  repeats = output_dim // subspace_rank
  f_k = frame_embs[:, :subspace_rank] * np.sqrt(float(repeats))

  raw_e = np.mean(mel_40[:, 2:32], axis=-1)
  e_smooth = ndimage.uniform_filter1d(raw_e, size=5, mode="nearest")
  e_th = (
      np.percentile(e_smooth, 25) * 0.45
      + np.percentile(e_smooth, 85) * 0.55
      - 1.0
  )
  soft_voice = (
      1.0 / (1.0 + np.exp(-2.0 * (e_smooth - e_th)))
  )[:, None].astype(np.float32)

  if frontend_version == "v1":
    mu = subspace.get("mel_mean_v1", np.zeros((40,), dtype=np.float32))
    sig = subspace.get("mel_std_v1", np.ones((40,), dtype=np.float32))
    norm_mel = (mel_40 - mu[None, :]) / sig[None, :]
    features = norm_mel.copy()
    features[:, :32] = (
        0.35 * norm_mel[:, :32] + 2.2 * soft_voice * np.tile(f_k, (1, 4))
    )
  else:
    mu = subspace.get("mel_mean_v2", np.zeros((512,), dtype=np.float32))
    sig = subspace.get("mel_std_v2", np.ones((512,), dtype=np.float32))
    norm_feat = (feat_v2 - mu[None, :]) / sig[None, :]
    features = norm_feat.copy()
    features[:, :480] = (
        0.25 * norm_feat[:, :480] + 2.4 * soft_voice * np.tile(f_k, (1, 60))
    )
  return features.astype(np.float32), frame_embs.astype(np.float32)


def load_mls_manifest_records(
    manifest_path: str,
    max_duration_sec: float = 1.8,
    silence_pad_sec: float = 0.20,
) -> list[dict[str, Any]]:
  """Loads audio files listed in an MLS JSON manifest and extracts features."""
  with open(manifest_path, "r", encoding="utf-8") as f:
    items = json.load(f)

  fe_v1 = frontend_lib.LogMelFrontend(configs.FrontendConfig.pvad_v1())
  fe_v2 = frontend_lib.LogMelFrontend(configs.FrontendConfig.pvad_v2())
  rng = np.random.default_rng(1234)

  records: list[dict[str, Any]] = []
  for item in items:
    audio, sr = sf.read(item["audio_path"], dtype="float32")
    if audio.ndim > 1:
      audio = np.mean(audio, axis=-1)
    if sr != 16000:
      raise ValueError(f"Expected 16 kHz audio, got {sr}")

    max_samples = int(max_duration_sec * sr)
    if len(audio) > max_samples:
      audio = audio[:max_samples]
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-4:
      audio = audio / peak * 0.85

    pad_len = int(silence_pad_sec * sr)
    pad_head = rng.normal(0.0, 1e-4, size=(pad_len,)).astype(np.float32)
    pad_tail = rng.normal(0.0, 1e-4, size=(pad_len,)).astype(np.float32)
    audio_padded = np.concatenate([pad_head, audio, pad_tail], axis=0)

    mel_40_full = fe_v1(audio_padded).numpy()
    vad_40_full = compute_frame_vad_labels(mel_40_full)
    feat_v2 = fe_v2(audio_padded).numpy()[::2]
    num_v2 = feat_v2.shape[0]
    idx_v2 = np.minimum(np.arange(num_v2) * 6 + 1, len(vad_40_full) - 1)
    vad_v2 = vad_40_full[idx_v2]
    mel_40 = mel_40_full[idx_v2]
    vad_40 = vad_v2

    records.append({
        "utt_id": item["utt_id"],
        "speaker_id": f"{item['language']}_{item['speaker_id']}",
        "language": item["language"],
        "role": item["role"],
        "mel_40_full": mel_40_full,
        "vad_40_full": vad_40_full,
        "mel_40": mel_40,
        "vad_40": vad_40,
        "feat_v2": feat_v2,
        "vad_v2": vad_v2,
        "idx_v2": idx_v2,
    })
  return records


def build_utterances_from_mls_records(
    records: Sequence[dict[str, Any]],
    subspace: dict[str, np.ndarray],
    frontend_version: str = "v1",
) -> list[dataset.UtteranceData]:
  """Builds single-speaker `UtteranceData` from MLS records using `subspace`."""
  enroll_embs: dict[str, list[np.ndarray]] = {}
  for r in records:
    if r["role"] == "enroll":
      mel_src = r.get("mel_40_full", r["mel_40"])
      vad_src = r.get("vad_40_full", r["vad_40"])
      f_embs = project_speaker_frames(mel_src, subspace, window=45)
      speech_idx = np.where(vad_src == 0)[0]
      if len(speech_idx) < 5:
        speech_idx = np.arange(len(f_embs))
      utt_emb = dataset.l2_normalize_numpy(np.mean(f_embs[speech_idx], axis=0))
      enroll_embs.setdefault(r["speaker_id"], []).append(utt_emb)

  speaker_centroids: dict[str, np.ndarray] = {}
  for spk, embs in enroll_embs.items():
    speaker_centroids[spk] = dataset.l2_normalize_numpy(
        np.mean(np.stack(embs, axis=0), axis=0)
    )

  utterances: list[dataset.UtteranceData] = []
  for r in records:
    if r["role"] == "enroll":
      continue
    spk_id = r["speaker_id"]
    if spk_id not in speaker_centroids:
      continue
    mel_src = r.get("mel_40_full", r["mel_40"])
    features, frame_embs = extract_subspace_conditioned_features(
        mel_40_full=mel_src,
        mel_40=r["mel_40"],
        feat_v2=r["feat_v2"],
        idx_v2=r["idx_v2"],
        subspace=subspace,
        frontend_version=frontend_version,
    )
    labels = r["vad_40"] if frontend_version == "v1" else r["vad_v2"]

    utterances.append(
        dataset.UtteranceData(
            utt_id=r["utt_id"],
            speaker_id=spk_id,
            features=features,
            labels=labels.astype(np.int32),
            speaker_embedding=speaker_centroids[spk_id],
            frame_speaker_embeddings=frame_embs,
        )
    )
  return utterances


def augment_utterances_with_speaker_rotations(
    utterances: Sequence[dataset.UtteranceData],
    num_rotations: int = 8,
    subspace_rank: int = 8,
    seed: int = 42,
) -> list[dataset.UtteranceData]:
  """Augments training utterances via orthogonal rotations in speaker subspace.

  Applies random $K \\times K$ orthogonal matrices $Q \\in O(K)$ simultaneously
  to the enrolled speaker d-vector, the speaker-subspace blocks of the acoustic
  features, and frame-level speaker embeddings, preserving all exact cosine
  similarities while uniformly covering the $K$-dimensional speaker hypersphere.
  """
  if num_rotations <= 1:
    return list(utterances)
  rng = np.random.default_rng(seed)
  augmented: list[dataset.UtteranceData] = []
  emb_dim = utterances[0].speaker_embedding.shape[0]
  feat_dim = utterances[0].features.shape[1]
  repeats = emb_dim // subspace_rank
  num_feat_blocks = (
      4 if feat_dim == 40 else (60 if feat_dim == 512 else 0)
  )

  for rot_idx in range(num_rotations):
    if rot_idx == 0:
      q_mat = np.eye(subspace_rank, dtype=np.float32)
    else:
      rand_mat = rng.standard_normal((subspace_rank, subspace_rank)).astype(
          np.float32
      )
      q_mat, r_mat = np.linalg.qr(rand_mat)
      q_mat = (q_mat * np.sign(np.diag(r_mat))).astype(np.float32)

    for utt in utterances:
      spk_k = utt.speaker_embedding[:subspace_rank] * np.sqrt(float(repeats))
      rot_spk = (
          np.tile(spk_k @ q_mat, (repeats,)) / np.sqrt(float(repeats))
      ).astype(np.float32)

      if rot_idx == 0 or num_feat_blocks == 0:
        rot_features = utt.features
      else:
        rot_features = utt.features.copy()
        span = num_feat_blocks * subspace_rank
        blocks = rot_features[:, :span].reshape(
            -1, num_feat_blocks, subspace_rank
        )
        rot_features[:, :span] = (blocks @ q_mat).reshape(-1, span)

      rot_frame_embs = None
      if utt.frame_speaker_embeddings is not None:
        f_k = utt.frame_speaker_embeddings[:, :subspace_rank] * np.sqrt(
            float(repeats)
        )
        rot_frame_embs = (
            np.tile(f_k @ q_mat, (1, repeats)) / np.sqrt(float(repeats))
        ).astype(np.float32)

      augmented.append(
          dataset.UtteranceData(
              utt_id=f"{utt.utt_id}_rot{rot_idx}",
              speaker_id=f"{utt.speaker_id}_rot{rot_idx}",
              features=rot_features,
              labels=utt.labels,
              speaker_embedding=rot_spk,
              cosine_scores=utt.cosine_scores,
              frame_speaker_embeddings=rot_frame_embs,
          )
      )
  return augmented


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
  has_frame_embs = any(
      u.frame_speaker_embeddings is not None for u in utterances
  )

  features = np.zeros(
      (num_examples, max_frames, feature_dim), dtype=np.float32
  )
  speaker_embeddings = np.zeros((num_examples, emb_dim), dtype=np.float32)
  cosine_scores = np.zeros((num_examples, max_frames, 1), dtype=np.float32)
  labels = np.zeros((num_examples, max_frames), dtype=np.int32)
  masks = np.zeros((num_examples, max_frames), dtype=np.float32)
  lengths = np.zeros((num_examples,), dtype=np.int32)
  frame_speaker_embeddings = (
      np.zeros((num_examples, max_frames, emb_dim), dtype=np.float32)
      if has_frame_embs
      else None
  )

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
    if (
        frame_speaker_embeddings is not None
        and utt.frame_speaker_embeddings is not None
    ):
      frame_speaker_embeddings[i, :length, :] = (
          utt.frame_speaker_embeddings[:length]
      )

  payload = {
      "features": features,
      "speaker_embeddings": speaker_embeddings,
      "cosine_scores": cosine_scores,
      "labels": labels,
      "masks": masks,
      "lengths": lengths,
  }
  if frame_speaker_embeddings is not None:
    payload["frame_speaker_embeddings"] = frame_speaker_embeddings

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
    frame_speaker_embeddings = (
        data["frame_speaker_embeddings"]
        if "frame_speaker_embeddings" in data
        else None
    )
    speaker_ids = (
        [str(s) for s in data["speaker_ids"]]
        if "speaker_ids" in data
        else [f"spk_{i}" for i in range(features.shape[0])]
    )

  utterances: list[dataset.UtteranceData] = []
  for i in range(features.shape[0]):
    length = int(lengths[i])
    cos = cosine_scores[i, :length] if cosine_scores is not None else None
    f_embs = (
        frame_speaker_embeddings[i, :length].astype(np.float32)
        if frame_speaker_embeddings is not None
        else None
    )
    utterances.append(
        dataset.UtteranceData(
            utt_id=f"utt_{i:04d}",
            speaker_id=speaker_ids[i],
            features=features[i, :length].astype(np.float32),
            labels=labels[i, :length].astype(np.int32),
            speaker_embedding=speaker_embeddings[i].astype(np.float32),
            cosine_scores=cos,
            frame_speaker_embeddings=f_embs,
        )
    )
  return utterances


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Prepare multi-speaker concatenated Personal VAD dataset."
  )
  parser.add_argument(
      "--mls_manifest",
      type=str,
      default=None,
      help="Path to Multilingual LibriSpeech JSON manifest file.",
  )
  parser.add_argument(
      "--subspace_npz",
      type=str,
      default=None,
      help=(
          "Path to save/load LDA+WCCN speaker subspace .npz file when using "
          "--mls_manifest."
      ),
  )
  parser.add_argument(
      "--fit_subspace",
      action="store_true",
      help="Fit and save a new LDA+WCCN speaker subspace on --mls_manifest.",
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
      "--num_passes",
      type=int,
      default=1,
      help="Number of random shuffle passes over the utterance pool.",
  )
  parser.add_argument(
      "--augment_speaker_rotations",
      type=int,
      default=1,
      help="Number of orthogonal speaker-subspace rotations to apply.",
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

  if args.mls_manifest is not None:
    records = load_mls_manifest_records(args.mls_manifest)
    if args.fit_subspace or (
        args.subspace_npz is not None and not os.path.exists(args.subspace_npz)
    ):
      subspace = fit_lda_wccn_subspace(records)
      if args.subspace_npz is not None:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.subspace_npz)), exist_ok=True
        )
        np.savez(args.subspace_npz, **subspace)
    elif args.subspace_npz is not None:
      with np.load(args.subspace_npz, allow_pickle=False) as data:
        subspace = {k: data[k] for k in data.files}
    else:
      subspace = fit_lda_wccn_subspace(records)
    single_utterances = build_utterances_from_mls_records(
        records, subspace=subspace, frontend_version=args.frontend_version
    )
  elif args.generate_synthetic or args.input_npz is None:
    single_utterances = generate_synthetic_single_speaker_utterances(
        num_utterances=args.num_utterances,
        feature_dim=fe_cfg.output_dim,
        seed=args.seed,
    )
  else:
    single_utterances = load_utterances_from_npz(args.input_npz)

  rng = np.random.default_rng(args.seed)
  all_concat: list[dataset.UtteranceData] = []
  for p in range(max(1, args.num_passes)):
    pool = list(single_utterances)
    if p > 0:
      rng.shuffle(pool)
    concat_pass = dataset.concat_utterance_group(
        pool,
        min_utterances=args.min_utterances,
        max_utterances=args.max_utterances,
        convert_for_vad_baseline=args.convert_for_vad_baseline,
        enrollment_less_prob=args.enrollment_less_prob,
        rng=rng,
    )
    all_concat.extend(concat_pass)

  if args.augment_speaker_rotations > 1:
    all_concat = augment_utterances_with_speaker_rotations(
        all_concat,
        num_rotations=args.augment_speaker_rotations,
        seed=args.seed,
    )

  save_utterances_to_npz(all_concat, args.output_npz)


if __name__ == "__main__":
  main()
