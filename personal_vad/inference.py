"""High-level inference API and Hugging Face Hub loader for Personal VAD.

Provides `load_pretrained` and `PersonalVadInferenceEngine` to load trained
Personal VAD 1.0 and 2.0 checkpoints (from a local directory or a Hugging Face
Hub repository ID such as `'wq2012/personal-vad-v1-et-wpl'` or
`'wq2012/personal-vad-v2-conformer-unified'`) and run end-to-end inference on
raw 16 kHz audio waveforms or pre-extracted acoustic features using either
TensorFlow/Keras (`.weights.h5` / `.safetensors`) or TensorFlow Lite
(`model_quantized.tflite` / `model_fp32.tflite`).
"""

import json
import os
import shutil
from typing import Any, Optional, Union
import numpy as np
from scipy import ndimage
import soundfile as sf
import tensorflow as tf
from . import configs
from . import dataset
from . import eval_lib
from . import frontend as frontend_lib
from . import model as model_lib
from . import tflite_export


def _extract_sliding_speaker_stats(
    mel_40: np.ndarray, window: int = 35
) -> np.ndarray:
  """Computes 120-dim sliding-window acoustic statistics from 40-dim log-Mel."""
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


def _project_speaker_frames(
    mel_40: np.ndarray,
    subspace: dict[str, np.ndarray],
    window: int = 35,
) -> np.ndarray:
  """Projects `[T, 40]` log-Mel frames into `[T, 256]` normalized d-vectors."""
  raw = _extract_sliding_speaker_stats(mel_40, window=window)
  x = (raw - subspace["mean"][None, :]) / subspace["std"][None, :]
  z_k = x @ subspace["proj_k"]
  z_k = z_k / (np.linalg.norm(z_k, axis=-1, keepdims=True) + 1e-8)
  repeats = int(subspace["output_dim"]) // z_k.shape[1]
  tiled = np.tile(z_k, (1, repeats)) / np.sqrt(float(repeats))
  return tiled.astype(np.float32)


class PersonalVadInferenceEngine:
  """End-to-end inference wrapper for Personal VAD 1.0 and 2.0 models."""

  def __init__(
      self,
      pvad_model: model_lib.PersonalVadModel,
      model_dir: str,
      subspace: Optional[dict[str, np.ndarray]] = None,
      quant_tflite_path: Optional[str] = None,
      fp32_tflite_path: Optional[str] = None,
  ):
    self.model = pvad_model
    self.config = pvad_model.config
    self.model_dir = model_dir
    self.subspace = subspace
    self.quant_tflite_path = quant_tflite_path
    self.fp32_tflite_path = fp32_tflite_path
    self._fe_v1 = frontend_lib.LogMelFrontend(configs.FrontendConfig.pvad_v1())
    self._fe_v2 = frontend_lib.LogMelFrontend(configs.FrontendConfig.pvad_v2())
    self._tflite_runner: Optional[tflite_export.TFLitePersonalVadRunner] = None

  def _load_audio(
      self, audio_or_path: Union[str, np.ndarray], sample_rate: int = 16000
  ) -> np.ndarray:
    """Loads or validates a 1D float32 16 kHz waveform."""
    if isinstance(audio_or_path, str):
      wav, sr = sf.read(audio_or_path, dtype="float32")
      if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
      if sr != sample_rate:
        raise ValueError(
            f"Expected {sample_rate} Hz audio, got {sr} Hz in {audio_or_path}"
        )
      return wav.astype(np.float32)
    wav = np.asarray(audio_or_path, dtype=np.float32)
    if wav.ndim > 1:
      wav = np.mean(wav, axis=-1)
    return wav

  def extract_speaker_embedding(
      self,
      enrollment_audio: Union[str, np.ndarray],
      sample_rate: int = 16000,
  ) -> np.ndarray:
    """Extracts an L2-normalized 256-dim d-vector from enrollment audio."""
    wav = self._load_audio(enrollment_audio, sample_rate=sample_rate)
    peak = float(np.max(np.abs(wav)))
    if peak > 1e-4:
      wav = wav / peak * 0.85
    mel_40 = self._fe_v1(wav).numpy()
    if self.subspace is not None:
      f_embs = _project_speaker_frames(mel_40, self.subspace, window=45)
      energy = np.mean(mel_40[:, 2:32], axis=-1)
      th = (
          np.percentile(energy, 25) * 0.45
          + np.percentile(energy, 85) * 0.55
          - 1.0
      )
      voiced_idx = np.where(energy > th)[0]
      if len(voiced_idx) < 5:
        voiced_idx = np.arange(len(f_embs))
      return dataset.l2_normalize_numpy(np.mean(f_embs[voiced_idx], axis=0))

    # Fallback acoustic summary embedding when no subspace file is present
    emb_dim = self.config.speaker_embedding_dim
    mean_v = np.mean(mel_40, axis=0)
    std_v = np.std(mel_40, axis=0)
    raw = np.concatenate([mean_v, std_v], axis=0)
    reps = int(np.ceil(emb_dim / len(raw)))
    return dataset.l2_normalize_numpy(np.tile(raw, reps)[:emb_dim])

  def extract_acoustic_features(
      self, audio: Union[str, np.ndarray], sample_rate: int = 16000
  ) -> tuple[np.ndarray, np.ndarray, float]:
    """Extracts `(features, frame_speaker_embeddings, frame_step_sec)`."""
    wav = self._load_audio(audio, sample_rate=sample_rate)
    peak = float(np.max(np.abs(wav)))
    if peak > 1e-4:
      wav = wav / peak * 0.85

    mel_40_full = self._fe_v1(wav).numpy()
    feat_v2_full = self._fe_v2(wav).numpy()
    is_v1 = self.config.feature_dim == 40

    if self.subspace is not None:
      feat_v2 = feat_v2_full[::2]
      num_v2 = feat_v2.shape[0]
      idx_v2 = np.minimum(np.arange(num_v2) * 6 + 1, len(mel_40_full) - 1)
      mel_40 = mel_40_full[idx_v2]

      f_local = _project_speaker_frames(mel_40_full, self.subspace, window=35)[
          idx_v2
      ]
      f_wide = _project_speaker_frames(mel_40_full, self.subspace, window=95)[
          idx_v2
      ]
      frame_embs = dataset.l2_normalize_numpy(0.65 * f_wide + 0.35 * f_local)

      subspace_rank = int(self.subspace["subspace_rank"])
      output_dim = int(self.subspace["output_dim"])
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

      if is_v1:
        mu = self.subspace.get("mel_mean_v1", np.zeros((40,), dtype=np.float32))
        sig = self.subspace.get("mel_std_v1", np.ones((40,), dtype=np.float32))
        norm_mel = (mel_40 - mu[None, :]) / sig[None, :]
        features = norm_mel.copy()
        features[:, :32] = (
            0.35 * norm_mel[:, :32] + 2.2 * soft_voice * np.tile(f_k, (1, 4))
        )
      else:
        mu = self.subspace.get(
            "mel_mean_v2", np.zeros((512,), dtype=np.float32)
        )
        sig = self.subspace.get("mel_std_v2", np.ones((512,), dtype=np.float32))
        norm_feat = (feat_v2 - mu[None, :]) / sig[None, :]
        features = norm_feat.copy()
        features[:, :480] = (
            0.25 * norm_feat[:, :480] + 2.4 * soft_voice * np.tile(f_k, (1, 60))
        )
      frame_step_sec = float(len(wav)) / float(sample_rate * max(num_v2, 1))
      return (
          features.astype(np.float32),
          frame_embs.astype(np.float32),
          frame_step_sec,
      )

    if is_v1:
      features = mel_40_full.astype(np.float32)
      frame_step_sec = 0.010
    else:
      features = feat_v2_full.astype(np.float32)
      frame_step_sec = 0.030
    dummy_embs = np.zeros(
        (features.shape[0], self.config.speaker_embedding_dim), dtype=np.float32
    )
    return features, dummy_embs, frame_step_sec

  def predict_features(
      self,
      features: np.ndarray,
      speaker_embedding: Optional[np.ndarray] = None,
      cosine_score: Optional[np.ndarray] = None,
      use_tflite: bool = False,
  ) -> np.ndarray:
    """Runs frame-level probability prediction `[T, C]` on `features`."""
    features = np.asarray(features, dtype=np.float32)
    if speaker_embedding is None:
      speaker_embedding = np.zeros(
          (self.config.speaker_embedding_dim,), dtype=np.float32
      )
    else:
      speaker_embedding = np.asarray(speaker_embedding, dtype=np.float32)

    if use_tflite:
      tflite_file = self.quant_tflite_path or self.fp32_tflite_path
      if tflite_file is None or not os.path.exists(tflite_file):
        raise FileNotFoundError(
            f"No .tflite file available in {self.model_dir}"
        )
      if self._tflite_runner is None:
        self._tflite_runner = tflite_export.TFLitePersonalVadRunner(tflite_file)
      return self._tflite_runner.predict(
          features=features,
          speaker_embedding=speaker_embedding,
          cosine_score=cosine_score,
      )

    feat_b = features[None, :, :]
    spk_b = speaker_embedding.reshape(1, -1)
    cos_b = (
        cosine_score.reshape(1, -1, 1).astype(np.float32)
        if cosine_score is not None
        else None
    )
    probs = self.model.predict_proba(
        feat_b, speaker_embedding=spk_b, cosine_score=cos_b
    ).numpy()
    return probs[0]

  def predict_audio(
      self,
      audio: Union[str, np.ndarray],
      enrollment_audio_or_embedding: Optional[Union[str, np.ndarray]] = None,
      sample_rate: int = 16000,
      use_tflite: bool = False,
      threshold: float = 0.5,
  ) -> dict[str, Any]:
    """Runs full-utterance Personal VAD inference on raw 16 kHz audio."""
    wav = self._load_audio(audio, sample_rate=sample_rate)
    features, frame_embs, frame_step_sec = self.extract_acoustic_features(
        wav, sample_rate=sample_rate
    )

    has_enrollment = enrollment_audio_or_embedding is not None
    if isinstance(enrollment_audio_or_embedding, str):
      spk_emb = self.extract_speaker_embedding(
          enrollment_audio_or_embedding, sample_rate=sample_rate
      )
    elif enrollment_audio_or_embedding is not None:
      arr = np.asarray(enrollment_audio_or_embedding, dtype=np.float32)
      if arr.ndim == 1 and arr.shape[0] == self.config.speaker_embedding_dim:
        spk_emb = dataset.l2_normalize_numpy(arr)
      else:
        spk_emb = self.extract_speaker_embedding(arr, sample_rate=sample_rate)
    else:
      spk_emb = np.zeros(
          (self.config.speaker_embedding_dim,), dtype=np.float32
      )

    if has_enrollment and np.linalg.norm(spk_emb) > 1e-6:
      cos_scores = dataset.compute_frame_cosine_similarity(frame_embs, spk_emb)
    else:
      cos_scores = np.zeros((features.shape[0],), dtype=np.float32)

    probs = self.predict_features(
        features=features,
        speaker_embedding=spk_emb,
        cosine_score=cos_scores,
        use_tflite=use_tflite,
    )

    if (
        self.config.conditioning_mode == configs.ConditioningMode.SC
        and probs.shape[1] == 2
        and has_enrollment
    ):
      stacked = np.concatenate([probs, cos_scores.reshape(-1, 1)], axis=1)
      probs = eval_lib.convert_scores_for_sc_baseline(stacked)

    num_frames = probs.shape[0]
    frame_times = (
        np.arange(num_frames, dtype=np.float32) + 0.5
    ) * frame_step_sec
    target_mask = probs[:, 0] >= threshold

    # Construct target-speaker-filtered waveform via frame posteriors
    sample_times = np.arange(len(wav), dtype=np.float32) / float(sample_rate)
    sample_weights = np.interp(
        sample_times,
        frame_times,
        probs[:, 0].astype(np.float32),
        left=float(probs[0, 0]),
        right=float(probs[-1, 0]),
    )
    gate = np.where(sample_weights >= threshold, 1.0, 0.03).astype(np.float32)
    filtered_audio = (wav * gate).astype(np.float32)

    pred_classes = np.argmax(probs, axis=-1)
    summary = {
        "num_frames": int(num_frames),
        "duration_sec": round(float(len(wav)) / float(sample_rate), 3),
        "target_speech_ratio": round(float(np.mean(pred_classes == 0)), 4),
        "silence_ratio": round(float(np.mean(pred_classes == 1)), 4),
        "non_target_speech_ratio": (
            round(float(np.mean(pred_classes == 2)), 4)
            if probs.shape[1] == 3
            else 0.0
        ),
        "mean_target_speech_prob": round(float(np.mean(probs[:, 0])), 4),
    }
    return {
        "posteriors": probs,
        "frame_times_sec": frame_times,
        "target_speech_mask": target_mask,
        "filtered_audio": filtered_audio,
        "speaker_embedding": spk_emb,
        "cosine_scores": cos_scores,
        "summary": summary,
    }


def load_pretrained(
    model_id_or_path: str,
    cache_dir: Optional[str] = None,
) -> PersonalVadInferenceEngine:
  """Loads a pretrained Personal VAD model from a local path or Hugging Face ID.

  Args:
    model_id_or_path: Either a local directory containing `model_config.json`
      and `model.weights.h5` (or `model.safetensors`), or a Hugging Face Hub
      model repository ID (e.g., `'wq2012/personal-vad-v1-et-wpl'` or
      `'wq2012/personal-vad-v2-conformer-unified'`).
    cache_dir: Optional local cache directory for Hugging Face downloads.

  Returns:
    A ready-to-run `PersonalVadInferenceEngine`.
  """
  if os.path.isdir(model_id_or_path):
    model_dir = os.path.abspath(model_id_or_path)
  else:
    from huggingface_hub import snapshot_download  # noqa: E402

    model_dir = snapshot_download(
        repo_id=model_id_or_path, cache_dir=cache_dir
    )

  config_path = os.path.join(model_dir, "model_config.json")
  with open(config_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

  raw["backbone"] = configs.BackboneType(raw["backbone"])
  raw["conditioning_mode"] = configs.ConditioningMode(raw["conditioning_mode"])
  cfg = configs.ModelConfig(**raw)
  pvad_model = model_lib.PersonalVadModel(config=cfg)

  dummy_features = tf.zeros([1, 1, cfg.feature_dim], dtype=tf.float32)
  dummy_spk = tf.zeros([1, cfg.speaker_embedding_dim], dtype=tf.float32)
  dummy_cos = tf.zeros([1, 1, 1], dtype=tf.float32)
  _ = pvad_model(
      {
          "features": dummy_features,
          "speaker_embedding": dummy_spk,
          "cosine_score": dummy_cos,
      },
      training=False,
  )

  weights_h5 = os.path.join(model_dir, "model.weights.h5")
  safetensors_file = os.path.join(model_dir, "model.safetensors")
  if os.path.exists(weights_h5):
    pvad_model.load_weights(weights_h5)
  elif os.path.exists(safetensors_file):
    from safetensors import numpy as st_np  # noqa: E402

    loaded = st_np.load_file(safetensors_file)
    ordered = [loaded[f"weight_{i:03d}"] for i in range(len(loaded))]
    pvad_model.set_weights(ordered)
  else:
    raise FileNotFoundError(
        f"No model.weights.h5 or model.safetensors found in {model_dir}"
    )

  subspace_candidates = [
      os.path.join(model_dir, "speaker_subspace.npz"),
      os.path.join(os.path.dirname(model_dir), "speaker_subspace.npz"),
  ]
  subspace: Optional[dict[str, np.ndarray]] = None
  for cand in subspace_candidates:
    if os.path.exists(cand):
      with np.load(cand, allow_pickle=False) as data:
        subspace = {k: data[k] for k in data.files}
      target_sub = os.path.join(model_dir, "speaker_subspace.npz")
      if not os.path.exists(target_sub) and os.path.abspath(
          cand
      ) != os.path.abspath(target_sub):
        try:
          shutil.copyfile(cand, target_sub)
        except OSError:
          pass
      break

  quant_tflite = os.path.join(model_dir, "model_quantized.tflite")
  fp32_tflite = os.path.join(model_dir, "model_fp32.tflite")
  return PersonalVadInferenceEngine(
      pvad_model=pvad_model,
      model_dir=model_dir,
      subspace=subspace,
      quant_tflite_path=quant_tflite if os.path.exists(quant_tflite) else None,
      fp32_tflite_path=fp32_tflite if os.path.exists(fp32_tflite) else None,
  )
