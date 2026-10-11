"""CLI script to prepare concatenated multi-speaker Personal VAD datasets.

Supports three data sources:
1. Multilingual LibriSpeech / LibriSpeech audio manifests (`--mls_manifest`):
   loads 16 kHz `.flac`/`.wav` utterances, computes clean reference frame-level
   VAD alignments (`speech=0`, `silence=1`), optionally applies Multistyle
   Training (MTR) reverberation and additive background noise, extracts strictly
   causal 40-dim (`v1`, 10ms frame step) or 512-dim (`v2`, 4x128 stacked + 3x
   subsampled, 30ms frame step) log-Mel filterbank energies, fits or applies the
   360-dim -> 256-dim regularized LDA/WCCN + PCA open-set speaker subspace, and
   concatenates multi-speaker utterance groups (`n ~ Uniform(1, 3)`).
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
from personal_vad import inference as inference_lib  # noqa: E402


def generate_synthetic_single_speaker_utterances(
    num_utterances: int = 30,
    num_speakers: int = 6,
    min_frames: int = 20,
    max_frames: int = 40,
    feature_dim: int = 40,
    speaker_embedding_dim: int = 256,
    seed: int = 42,
) -> list[dataset.UtteranceData]:
  """Generates synthetic single-speaker utterances with distinct speaker IDs."""
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
  """Computes reference frame-level binary VAD labels (`speech=0`, `silence=1`).

  Combines vocal-band log-Mel energy (`mel_40[:, 2:34]`) and spectral peak
  prominence with morphological hang-over smoothing on the clean reference
  audio before any MTR noise/reverberation is mixed.

  Args:
    mel_40: Clean 40-dim log-Mel filterbank energies of shape
      `[num_frames, 40]`.

  Returns:
    Integer array of shape `[num_frames]` with `0` for speech and `1` for
    silence.
  """
  if mel_40.shape[0] == 0:
    return np.zeros((0,), dtype=np.int32)
  vocal_energy = np.mean(mel_40[:, 2:34], axis=-1)
  spectral_var = np.std(mel_40[:, 2:34], axis=-1)
  score = vocal_energy + 0.35 * spectral_var
  p10 = float(np.percentile(score, 10))
  p90 = float(np.percentile(score, 90))
  threshold = p10 + 0.26 * (p90 - p10)
  is_speech = score >= threshold
  # Hang-over closing + median smoothing to bridge short plosive closures
  is_speech = ndimage.binary_closing(is_speech, structure=np.ones((5,))).astype(
      bool
  )
  is_speech = ndimage.median_filter(is_speech.astype(np.int32), size=5).astype(
      bool
  )
  return np.where(
      is_speech,
      int(configs.FrameLabel.SPEECH),
      int(configs.FrameLabel.SILENCE),
  ).astype(np.int32)


def fit_lda_wccn_subspace(
    records_with_mel: Sequence[dict[str, Any]],
    output_dim: int = 256,
) -> dict[str, np.ndarray]:
  """Fits a 360-D -> 256-D regularized LDA + WCCN + PCA speaker subspace.

  Uses strictly causal 360-dim multi-resolution acoustic statistics from the
  training split only:
  1. Computes training-set acoustic normalization (`mel_mean_v1`, `mel_std_v1`,
     `mel_mean_v2`, `mel_std_v2`) and fixed causal voicing threshold
     `energy_threshold`.
  2. Fits regularized Fisher LDA (`min(num_speakers - 1, 256)` directions)
     whitened via Within-Class Covariance Normalization (WCCN) and completed to
     full `output_dim = 256` with orthogonal PCA eigenvectors of the residual
     covariance (`proj_lda` of shape `(360, 256)`).
  3. Fits linear ridge-regression projections `w_x_v1` (`40 x 32`), `w_x_v2`
     (`512 x 32`), and `w_e_32` (`256 x 32`) on training frames for optional
     neural weight initialization.
  """
  all_mel_v1 = np.concatenate(
      [r["mel_40_full"] for r in records_with_mel], axis=0
  )
  mel_mean_v1 = np.mean(all_mel_v1, axis=0).astype(np.float32)
  mel_std_v1 = (np.std(all_mel_v1, axis=0) + 1e-5).astype(np.float32)

  all_mel_v2 = np.concatenate(
      [r["feat_v2_full"] for r in records_with_mel], axis=0
  )
  mel_mean_v2 = np.mean(all_mel_v2, axis=0).astype(np.float32)
  mel_std_v2 = (np.std(all_mel_v2, axis=0) + 1e-5).astype(np.float32)

  vocal_energies = np.mean(all_mel_v1[:, 2:34], axis=-1)
  energy_threshold = float(np.percentile(vocal_energies, 25))

  spk_vecs: dict[str, list[np.ndarray]] = {}
  all_vecs: list[np.ndarray] = []
  for r in records_with_mel:
    mel_40 = r["mel_40_full"]
    mel_80 = r["mel_80_full"]
    vad_40 = r["vad_40_full"]
    utt_360 = inference_lib.extract_360d_utterance_stats(
        mel_40=mel_40,
        mel_80=mel_80,
        mel_mean_v1=mel_mean_v1,
        energy_threshold=energy_threshold,
    )
    spk_vecs.setdefault(r["speaker_id"], []).append(utt_360)
    all_vecs.append(utt_360)

    causal_360 = inference_lib.extract_causal_360d_frame_stats(
        mel_40=mel_40,
        mel_80=mel_80,
        window=35,
        mel_mean_v1=mel_mean_v1,
        energy_threshold=energy_threshold,
    )
    speech_idx = np.where(vad_40 == 0)[0]
    for k in range(10, len(speech_idx), 20):
      frame_idx = speech_idx[k]
      vec = causal_360[frame_idx]
      spk_vecs[r["speaker_id"]].append(vec)
      all_vecs.append(vec)

  x_all = np.stack(all_vecs, axis=0).astype(np.float64)
  feat_mean = np.mean(x_all, axis=0)
  feat_std = np.std(x_all, axis=0) + 1e-5
  x_norm_all = (x_all - feat_mean[None, :]) / feat_std[None, :]

  dim = x_all.shape[1]  # 360
  sw = np.zeros((dim, dim), dtype=np.float64)
  sb = np.zeros((dim, dim), dtype=np.float64)
  global_mean = np.mean(x_norm_all, axis=0)
  total_n = len(x_norm_all)

  spk_means: dict[str, np.ndarray] = {}
  spk_counts: dict[str, int] = {}
  for spk, vecs in spk_vecs.items():
    v_norm = (np.stack(vecs, axis=0).astype(np.float64) - feat_mean) / feat_std
    spk_means[spk] = np.mean(v_norm, axis=0)
    spk_counts[spk] = len(v_norm)
    diff = v_norm - spk_means[spk][None, :]
    sw += diff.T @ diff
    m_diff = (spk_means[spk] - global_mean)[:, None]
    sb += spk_counts[spk] * (m_diff @ m_diff.T)

  sw /= max(total_n, 1)
  sb /= max(total_n, 1)
  sw_reg = sw + 0.08 * np.eye(dim, dtype=np.float64)

  eigvals, eigvecs = linalg.eigh(sb, sw_reg)
  order = np.argsort(eigvals)[::-1]
  eigvals = np.maximum(eigvals[order], 0.0)
  eigvecs = eigvecs[:, order]

  lda_rank = min(max(len(spk_vecs) - 1, 1), output_dim)
  v_lda = eigvecs[:, :lda_rank]

  # WCCN whitening within the LDA subspace paired with Fisher eigenvalue scaling
  wccn = np.zeros((lda_rank, lda_rank), dtype=np.float64)
  for spk, vecs in spk_vecs.items():
    v_norm = (np.stack(vecs, axis=0).astype(np.float64) - feat_mean) / feat_std
    z = v_norm @ v_lda
    d = z - np.mean(z, axis=0, keepdims=True)
    wccn += (d.T @ d) / max(len(z), 1)
  wccn /= max(len(spk_vecs), 1)
  wccn_reg = wccn + 0.10 * np.eye(lda_rank, dtype=np.float64)
  b_wccn = np.linalg.cholesky(np.linalg.inv(wccn_reg))
  fisher_scale = np.power(eigvals[:lda_rank] / (eigvals[0] + 1e-8) + 0.02, 0.45)
  proj_disc = v_lda @ b_wccn @ np.diag(fisher_scale)

  # Orthonormalize v_lda basis to find orthogonal PCA completion to 256-D
  if lda_rank < output_dim:
    rem_dim = output_dim - lda_rank
    q_lda, _ = np.linalg.qr(v_lda)
    proj_perp = np.eye(dim, dtype=np.float64) - q_lda @ q_lda.T
    cov_total = (x_norm_all.T @ x_norm_all) / max(total_n, 1)
    cov_res = proj_perp @ cov_total @ proj_perp
    pca_vals, pca_vecs = linalg.eigh(cov_res)
    pca_order = np.argsort(pca_vals)[::-1]
    v_pca = pca_vecs[:, pca_order[:rem_dim]] * 0.08
    proj_lda = np.concatenate([proj_disc, v_pca], axis=1).astype(np.float32)
  else:
    proj_lda = proj_disc[:, :output_dim].astype(np.float32)

  proj_bias = np.zeros((output_dim,), dtype=np.float32)
  proj_bias[-1] = 0.45
  proj_lda[:, -1] *= 0.05

  # Fit linear ridge-regression acoustic-to-speaker-subspace warm-start maps
  # on the training set (mapping pure normalized log-Mel x_t to top 32 speaker
  # coordinates).
  coord_dim = 32
  spk_centroid_list = []
  x_v1_list = []
  x_v2_list = []
  y_v1_list = []
  y_v2_list = []

  for spk, vecs in spk_vecs.items():
    v_norm = (np.stack(vecs, axis=0) - feat_mean) / feat_std
    e_256 = dataset.l2_normalize_numpy(
        np.mean(v_norm @ proj_lda, axis=0) + proj_bias
    )
    spk_centroid_list.append(e_256)

  spk_mat = np.stack(spk_centroid_list, axis=0).copy()
  spk_mat[:, -1] = 0.0
  _, _, vt_e = np.linalg.svd(spk_mat, full_matrices=False)
  actual_k = min(coord_dim, vt_e.shape[0])
  w_e_32 = np.zeros((output_dim, coord_dim), dtype=np.float32)
  w_e_32[:, :actual_k] = vt_e[:actual_k].T.astype(np.float32)
  w_e_32[-1, :] = 0.0

  spk_to_emb256: dict[str, np.ndarray] = {}
  for spk, e_256 in zip(spk_vecs.keys(), spk_centroid_list):
    spk_to_emb256[spk] = e_256

  for r in records_with_mel:
    spk = r["speaker_id"]
    target_z = (spk_to_emb256[spk] @ w_e_32).astype(np.float32)  # [32]
    vad1 = r["vad_40_full"]
    sp_idx1 = np.where(vad1 == 0)[0]
    if len(sp_idx1) > 0:
      norm_m1 = (r["mel_40_full"][sp_idx1] - mel_mean_v1[None, :]) / mel_std_v1[
          None, :
      ]
      x_v1_list.append(norm_m1)
      y_v1_list.append(np.tile(target_z[None, :], (len(sp_idx1), 1)))
    vad2 = r["vad_v2_full"]
    sp_idx2 = np.where(vad2 == 0)[0]
    if len(sp_idx2) > 0:
      norm_m2 = (
          r["feat_v2_full"][sp_idx2] - mel_mean_v2[None, :]
      ) / mel_std_v2[None, :]
      x_v2_list.append(norm_m2)
      y_v2_list.append(np.tile(target_z[None, :], (len(sp_idx2), 1)))

  x1_cat = np.concatenate(x_v1_list, axis=0).astype(np.float64)
  y1_cat = np.concatenate(y_v1_list, axis=0).astype(np.float64)
  w_x_v1 = linalg.solve(
      x1_cat.T @ x1_cat + 50.0 * np.eye(40), x1_cat.T @ y1_cat, assume_a="pos"
  ).astype(np.float32)

  x2_cat = np.concatenate(x_v2_list, axis=0).astype(np.float64)
  y2_cat = np.concatenate(y_v2_list, axis=0).astype(np.float64)
  w_x_v2 = linalg.solve(
      x2_cat.T @ x2_cat + 250.0 * np.eye(512),
      x2_cat.T @ y2_cat,
      assume_a="pos",
  ).astype(np.float32)

  return {
      "feat_mean": feat_mean.astype(np.float32),
      "feat_std": feat_std.astype(np.float32),
      "proj_lda": proj_lda,
      "proj_bias": proj_bias,
      "subspace_rank": np.array(
          int(np.linalg.matrix_rank(proj_lda)), dtype=np.int32
      ),
      "lda_rank": np.array(lda_rank, dtype=np.int32),
      "output_dim": np.array(output_dim, dtype=np.int32),
      "energy_threshold": np.array(energy_threshold, dtype=np.float32),
      "mel_mean_v1": mel_mean_v1,
      "mel_std_v1": mel_std_v1,
      "mel_mean_v2": mel_mean_v2,
      "mel_std_v2": mel_std_v2,
      "w_x_v1": w_x_v1,
      "w_x_v2": w_x_v2,
      "w_e_32": w_e_32,
  }


def extract_pure_acoustic_features(
    mel_40_full: np.ndarray,
    mel_80_full: np.ndarray,
    feat_v2_full: np.ndarray,
    subspace: dict[str, np.ndarray],
    frontend_version: str = "v1",
) -> tuple[np.ndarray, np.ndarray]:
  """Extracts pure normalized log-Mel features and causal frame d-vectors.

  - `v1`: 40-dim log-Mel at exact 10ms frame rate (`[T_10ms, 40]`).
  - `v2`: 512-dim stacked log-Mel at exact 30ms frame rate (`[T_30ms, 512]`).
  Zero speaker information is injected into `features`.
  """
  frame_embs_10ms = inference_lib.project_speaker_frames(
      mel_40=mel_40_full,
      mel_80=mel_80_full,
      subspace=subspace,
      window=14,
  )
  if frontend_version == "v1":
    mu = subspace["mel_mean_v1"]
    sig = np.maximum(subspace["mel_std_v1"], 1e-5)
    features = (mel_40_full - mu[None, :]) / sig[None, :]
    return features.astype(np.float32), frame_embs_10ms.astype(np.float32)

  mu = subspace["mel_mean_v2"]
  sig = np.maximum(subspace["mel_std_v2"], 1e-5)
  features = (feat_v2_full - mu[None, :]) / sig[None, :]
  num_v2 = features.shape[0]
  idx_30ms = np.minimum(np.arange(num_v2) * 3, frame_embs_10ms.shape[0] - 1)
  frame_embs_30ms = frame_embs_10ms[idx_30ms]
  return features.astype(np.float32), frame_embs_30ms.astype(np.float32)


def load_mls_manifest_records(
    manifest_path: str,
    max_duration_sec: float = 1.45,
    pause_margin_sec: float = 0.12,
    apply_mtr: bool = False,
    mtr_seed: int = 1234,
) -> list[dict[str, Any]]:
  """Loads audio files in an MLS JSON manifest and extracts causal features."""
  with open(manifest_path, "r", encoding="utf-8") as f:
    items = json.load(f)

  fe_v1 = frontend_lib.LogMelFrontend(configs.FrontendConfig.pvad_v1())
  fe_80 = frontend_lib.LogMelFrontend(
      configs.FrontendConfig(
          sample_rate=16000,
          frame_size_ms=25.0,
          frame_step_ms=10.0,
          num_mel_bins=80,
          lower_edge_hertz=80.0,
          upper_edge_hertz=7600.0,
          stack_left_context=0,
          stack_right_context=0,
          frame_stride=1,
      )
  )
  fe_v2 = frontend_lib.LogMelFrontend(configs.FrontendConfig.pvad_v2())
  rng = np.random.default_rng(mtr_seed)

  clean_waves: list[np.ndarray] = []
  proc_waves: list[np.ndarray] = []
  wave_lengths: list[int] = []
  for item in items:
    audio, sr = sf.read(item["audio_path"], dtype="float32")
    if audio.ndim > 1:
      audio = np.mean(audio, axis=-1)
    if sr != 16000:
      raise ValueError(f"Expected 16 kHz audio, got {sr}")

    max_samples = int(max_duration_sec * sr)
    if len(audio) > max_samples:
      audio = audio[:max_samples]

    # Extract natural low-energy room tone from the utterance boundaries to
    # form realistic inter-turn silence margins.
    win = 400
    n_wins = max(len(audio) // win, 1)
    trimmed = audio[: n_wins * win].reshape(n_wins, win)
    win_rms = np.sqrt(np.mean(trimmed**2, axis=-1))
    quiet_idx = int(np.argmin(win_rms))
    room_tone_unit = trimmed[quiet_idx] * 0.25
    pad_len = int(pause_margin_sec * sr)
    reps = int(np.ceil(pad_len / max(len(room_tone_unit), 1)))
    natural_pause = np.tile(room_tone_unit, reps)[:pad_len].astype(np.float32)
    clean_audio = np.concatenate(
        [natural_pause, audio, natural_pause], axis=0
    ).astype(np.float32)

    if apply_mtr and item.get("role") != "enroll":
      proc_audio = dataset.apply_mtr_augmentation(
          clean_audio,
          sample_rate=16000,
          snr_db_range=(6.0, 20.0),
          rt60_range=(0.10, 0.30),
          reverb_prob=0.5,
          noise_prob=0.7,
          rng=rng,
      )
    else:
      proc_audio = clean_audio

    clean_waves.append(clean_audio)
    proc_waves.append(proc_audio)
    wave_lengths.append(len(clean_audio))

  records: list[dict[str, Any]] = []
  pad_frames_10ms = max(int(pause_margin_sec / 0.010) - 1, 1)
  batch_size = 128
  for start in range(0, len(items), batch_size):
    end = min(start + batch_size, len(items))
    max_len = max(wave_lengths[start:end])
    b_clean = np.zeros((end - start, max_len), dtype=np.float32)
    b_proc = np.zeros((end - start, max_len), dtype=np.float32)
    for j, idx in enumerate(range(start, end)):
      length = wave_lengths[idx]
      b_clean[j, :length] = clean_waves[idx]
      b_proc[j, :length] = proc_waves[idx]

    b_clean_mel_40 = fe_v1(b_clean).numpy()
    b_mel_40 = (
        fe_v1(b_proc).numpy() if apply_mtr else b_clean_mel_40
    )
    b_mel_80 = fe_80(b_proc).numpy()
    b_feat_v2 = fe_v2(b_proc).numpy()

    for j, idx in enumerate(range(start, end)):
      item = items[idx]
      length = wave_lengths[idx]
      n_v1 = max(0, (length - 400) // 160 + 1)
      n_v2_10ms = max(0, (length - 512) // 160 + 1)
      n_v2 = (n_v2_10ms + 2) // 3

      clean_mel_40 = b_clean_mel_40[j, :n_v1]
      vad_40_full = compute_frame_vad_labels(clean_mel_40)
      vad_40_full[:pad_frames_10ms] = int(configs.FrameLabel.SILENCE)
      vad_40_full[-pad_frames_10ms:] = int(configs.FrameLabel.SILENCE)

      mel_40_full = b_mel_40[j, :n_v1]
      mel_80_full = b_mel_80[j, :n_v1]
      feat_v2_full = b_feat_v2[j, :n_v2]
      idx_30ms = np.minimum(
          np.arange(n_v2) * 3 + 1, len(vad_40_full) - 1
      )
      vad_v2_full = vad_40_full[idx_30ms]

      spk_id = str(item["speaker_id"])
      lang = str(item["language"])
      if not spk_id.startswith(f"{lang}_"):
        spk_id = f"{lang}_{spk_id}"

      records.append({
          "utt_id": item["utt_id"],
          "speaker_id": spk_id,
          "language": lang,
          "role": item["role"],
          "mel_40_full": mel_40_full,
          "mel_80_full": mel_80_full,
          "vad_40_full": vad_40_full,
          "feat_v2_full": feat_v2_full,
          "vad_v2_full": vad_v2_full,
      })
  return records


def build_utterances_from_mls_records(
    records: Sequence[dict[str, Any]],
    subspace: dict[str, np.ndarray],
    frontend_version: str = "v1",
) -> list[dataset.UtteranceData]:
  """Builds single-speaker `UtteranceData` from MLS records using `subspace`."""
  enroll_embs: dict[str, list[np.ndarray]] = {}
  e_th = float(
      subspace.get("energy_threshold", np.array(-7.5)).reshape(-1)[0]
  )
  mel_mean_v1 = subspace.get("mel_mean_v1", None)

  for r in records:
    if r["role"] == "enroll":
      raw_360 = inference_lib.extract_360d_utterance_stats(
          mel_40=r["mel_40_full"],
          mel_80=r["mel_80_full"],
          mel_mean_v1=mel_mean_v1,
          energy_threshold=e_th,
      )
      x_norm = (raw_360 - subspace["feat_mean"]) / np.maximum(
          subspace["feat_std"], 1e-5
      )
      emb_raw = x_norm @ subspace["proj_lda"]
      if "proj_bias" in subspace:
        emb_raw = emb_raw + subspace["proj_bias"]
      utt_emb = dataset.l2_normalize_numpy(emb_raw)
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
    features, frame_embs = extract_pure_acoustic_features(
        mel_40_full=r["mel_40_full"],
        mel_80_full=r["mel_80_full"],
        feat_v2_full=r["feat_v2_full"],
        subspace=subspace,
        frontend_version=frontend_version,
    )
    labels = (
        r["vad_40_full"] if frontend_version == "v1" else r["vad_v2_full"]
    )

    utterances.append(
        dataset.UtteranceData(
            utt_id=r["utt_id"],
            speaker_id=spk_id,
            features=features,
            labels=labels.astype(np.int32),
            speaker_embedding=speaker_centroids[spk_id],
            frame_speaker_embeddings=frame_embs,
            language=r.get("language"),
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
  speaker_ids = np.array([u.speaker_id for u in utterances], dtype="<U64")
  languages = np.array(
      [u.language or "unknown" for u in utterances], dtype="<U16"
  )
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
      "speaker_ids": speaker_ids,
      "languages": languages,
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
    languages = (
        [str(lang_val) for lang_val in data["languages"]]
        if "languages" in data
        else [None for _ in range(features.shape[0])]
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
            language=languages[i],
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
          "Path to save/load 360-D -> 256-D speaker subspace .npz file when "
          "using --mls_manifest."
      ),
  )
  parser.add_argument(
      "--fit_subspace",
      action="store_true",
      help="Fit and save a new speaker subspace on --mls_manifest.",
  )
  parser.add_argument(
      "--apply_mtr",
      action="store_true",
      help="Apply Multistyle Training (MTR) RIR reverberation and noise.",
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
      help="Frontend configuration version (v1=40d@10ms, v2=512d@30ms).",
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
    records = load_mls_manifest_records(
        args.mls_manifest, apply_mtr=args.apply_mtr, mtr_seed=args.seed
    )
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
  for _ in range(max(1, args.num_passes)):
    pool = list(single_utterances)
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

  save_utterances_to_npz(all_concat, args.output_npz)


if __name__ == "__main__":
  main()
