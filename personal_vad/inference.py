"""High-level causal inference API and Hugging Face Hub loader for Personal VAD.

Provides `load_pretrained` and `PersonalVadInferenceEngine` to load trained
Personal VAD 1.0 and 2.0 checkpoints (from a local directory or a Hugging Face
Hub repository ID such as `'wq2012/personal-vad-v1-et-wpl'` or
`'wq2012/personal-vad-v2-conformer-unified'`) and run strictly causal
end-to-end inference on raw audio waveforms (automatically resampled to 16 kHz)
or pre-extracted acoustic features using either TensorFlow/Keras
(`.weights.h5` / `.safetensors`) or TensorFlow Lite
(`model_quantized.tflite` / `model_fp32.tflite`).
"""

import json
import math
import os
import shutil
from typing import Any, Optional, Union
import numpy as np
from scipy import signal
import soundfile as sf
import tensorflow as tf
from . import configs
from . import dataset
from . import eval_lib
from . import frontend as frontend_lib
from . import model as model_lib
from . import tflite_export


def _causal_window_sums(x: np.ndarray, window: int) -> np.ndarray:
  """Computes causal sliding-window sums over `[max(0, t - window + 1), t]`."""
  csum = np.cumsum(x, axis=0, dtype=np.float64)
  if x.shape[0] <= window:
    return csum.astype(np.float32)
  shifted = np.zeros_like(csum)
  shifted[window:] = csum[:-window]
  return (csum - shifted).astype(np.float32)


def extract_causal_360d_frame_stats(
    mel_40: np.ndarray,
    mel_80: np.ndarray,
    window: int = 45,
    mel_mean_v1: Optional[np.ndarray] = None,
    energy_threshold: float = -7.5,
) -> np.ndarray:
  """Computes strictly causal 360-dim multi-resolution speaker statistics.

  For each frame `t`, aggregates only past and current frames in
  `[max(0, t - window + 1), t]` weighted by a causal frame-energy voicing gate:
    - 40-D log-Mel causal weighted mean (40)
    - 40-D log-Mel causal weighted standard deviation (40)
    - 40-D log-Mel positive formant deviation above reference mean (40)
    - 40-D log-Mel negative valley deviation below reference mean (40)
    - 40-D log-Mel causal first-order difference RMS (40)
    - 80-D high-resolution log-Mel causal weighted mean (80)
    - 80-D high-resolution log-Mel causal weighted standard deviation (80)
  Total dimension: `40 * 5 + 80 * 2 = 360`.

  Args:
    mel_40: Float32 array of shape `[T, 40]`.
    mel_80: Float32 array of shape `[T, 80]`.
    window: Causal sliding window length in 10ms frames.
    mel_mean_v1: Optional fixed 40-D reference log-Mel mean.
    energy_threshold: Fixed scalar log-Mel voicing threshold.

  Returns:
    Float32 array of shape `[T, 360]`.
  """
  mel_40 = np.asarray(mel_40, dtype=np.float32)
  mel_80 = np.asarray(mel_80, dtype=np.float32)
  num_frames = mel_40.shape[0]
  if mel_80.shape[0] != num_frames:
    min_t = min(num_frames, mel_80.shape[0])
    mel_40 = mel_40[:min_t]
    mel_80 = mel_80[:min_t]
    num_frames = min_t

  if num_frames == 0:
    return np.zeros((0, 360), dtype=np.float32)

  ref_mean = (
      np.asarray(mel_mean_v1, dtype=np.float32).reshape(1, 40)
      if mel_mean_v1 is not None
      else np.full((1, 40), energy_threshold, dtype=np.float32)
  )

  # Strictly causal frame voicing weight (depends only on frame t)
  frame_energy = np.mean(mel_40[:, 2:34], axis=-1, keepdims=True)
  z_gate = np.clip((frame_energy - float(energy_threshold)) * 1.5, -15.0, 15.0)
  weights = (1.0 / (1.0 + np.exp(-z_gate))).astype(np.float32) + 0.05

  delta_40 = np.zeros_like(mel_40)
  if num_frames > 1:
    delta_40[1:] = mel_40[1:] - mel_40[:-1]

  pos_dev = np.maximum(mel_40 - ref_mean, 0.0)
  neg_dev = np.maximum(ref_mean - mel_40, 0.0)

  w_sum = np.maximum(_causal_window_sums(weights, window), 1e-4)
  mean_40 = _causal_window_sums(weights * mel_40, window) / w_sum
  sq_40 = _causal_window_sums(weights * (mel_40**2), window) / w_sum
  std_40 = np.sqrt(np.maximum(sq_40 - mean_40**2, 1e-4))

  pos_40 = _causal_window_sums(weights * pos_dev, window) / w_sum
  neg_40 = _causal_window_sums(weights * neg_dev, window) / w_sum
  d_rms_40 = np.sqrt(
      np.maximum(
          _causal_window_sums(weights * (delta_40**2), window) / w_sum, 1e-4
      )
  )

  mean_80 = _causal_window_sums(weights * mel_80, window) / w_sum
  sq_80 = _causal_window_sums(weights * (mel_80**2), window) / w_sum
  std_80 = np.sqrt(np.maximum(sq_80 - mean_80**2, 1e-4))

  return np.concatenate(
      [mean_40, std_40, pos_40, neg_40, d_rms_40, mean_80, std_80], axis=-1
  ).astype(np.float32)


def extract_360d_utterance_stats(
    mel_40: np.ndarray,
    mel_80: np.ndarray,
    mel_mean_v1: Optional[np.ndarray] = None,
    energy_threshold: float = -7.5,
) -> np.ndarray:
  """Computes a 360-dim utterance summary vector from 40-D and 80-D log-Mel."""
  frame_stats = extract_causal_360d_frame_stats(
      mel_40=mel_40,
      mel_80=mel_80,
      window=max(mel_40.shape[0], 1),
      mel_mean_v1=mel_mean_v1,
      energy_threshold=energy_threshold,
  )
  if frame_stats.shape[0] == 0:
    return np.zeros((360,), dtype=np.float32)
  return frame_stats[-1].astype(np.float32)


def project_speaker_frames(
    mel_40: np.ndarray,
    mel_80: np.ndarray,
    subspace: dict[str, np.ndarray],
    window: int = 45,
) -> np.ndarray:
  """Projects `[T, 40]` + `[T, 80]` log-Mel causally into `[T, 256]` d-vectors.
  """
  e_th = float(subspace.get("energy_threshold", np.array(-7.5)).reshape(-1)[0])
  mel_mean_v1 = subspace.get("mel_mean_v1", None)
  raw_360 = extract_causal_360d_frame_stats(
      mel_40=mel_40,
      mel_80=mel_80,
      window=window,
      mel_mean_v1=mel_mean_v1,
      energy_threshold=e_th,
  )
  feat_mean = subspace["feat_mean"]
  feat_std = subspace["feat_std"]
  proj_lda = subspace["proj_lda"]
  x_norm = (raw_360 - feat_mean[None, :]) / np.maximum(feat_std[None, :], 1e-5)
  emb_256 = x_norm @ proj_lda
  if "proj_bias" in subspace:
    emb_256 = emb_256 + subspace["proj_bias"][None, :]
  return dataset.l2_normalize_numpy(emb_256, axis=-1)


class PersonalVadInferenceEngine:
  """Strictly causal end-to-end inference engine for Personal VAD 1.0 & 2.0."""

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
    self._fe_80 = frontend_lib.LogMelFrontend(
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
    self._fe_v2 = frontend_lib.LogMelFrontend(configs.FrontendConfig.pvad_v2())
    self._tflite_runner: Optional[tflite_export.TFLitePersonalVadRunner] = None

  def _load_audio(
      self, audio_or_path: Union[str, np.ndarray], sample_rate: int = 16000
  ) -> np.ndarray:
    """Loads or validates a 1-D float32 16 kHz waveform (with resampling)."""
    if isinstance(audio_or_path, str):
      wav, sr = sf.read(audio_or_path, dtype="float32")
    else:
      wav = np.asarray(audio_or_path, dtype=np.float32)
      sr = sample_rate

    if wav.ndim > 1:
      wav = np.mean(wav, axis=-1)
    wav = wav.reshape(-1).astype(np.float32)
    if sr != 16000 and wav.size > 0:
      g = math.gcd(int(sr), 16000)
      up = 16000 // g
      down = int(sr) // g
      wav = signal.resample_poly(wav, up, down).astype(np.float32)
    return wav

  def extract_speaker_embedding(
      self,
      enrollment_audio: Union[str, np.ndarray],
      sample_rate: int = 16000,
  ) -> np.ndarray:
    """Extracts an L2-normalized 256-dim d-vector from enrollment audio."""
    wav = self._load_audio(enrollment_audio, sample_rate=sample_rate)
    mel_40 = self._fe_v1(wav).numpy()
    mel_80 = self._fe_80(wav).numpy()
    if self.subspace is not None and "proj_lda" in self.subspace:
      e_th = float(
          self.subspace.get("energy_threshold", np.array(-7.5)).reshape(-1)[0]
      )
      mel_mean_v1 = self.subspace.get("mel_mean_v1", None)
      raw_360 = extract_360d_utterance_stats(
          mel_40=mel_40,
          mel_80=mel_80,
          mel_mean_v1=mel_mean_v1,
          energy_threshold=e_th,
      )
      x_norm = (raw_360 - self.subspace["feat_mean"]) / np.maximum(
          self.subspace["feat_std"], 1e-5
      )
      emb_256 = x_norm @ self.subspace["proj_lda"]
      if "proj_bias" in self.subspace:
        emb_256 = emb_256 + self.subspace["proj_bias"]
      return dataset.l2_normalize_numpy(emb_256)

    # Fallback acoustic summary embedding when no subspace file is present
    emb_dim = self.config.speaker_embedding_dim
    raw = extract_360d_utterance_stats(mel_40, mel_80)
    if emb_dim <= raw.shape[0]:
      return dataset.l2_normalize_numpy(raw[:emb_dim])
    reps = int(np.ceil(emb_dim / len(raw)))
    return dataset.l2_normalize_numpy(np.tile(raw, reps)[:emb_dim])

  def extract_acoustic_features(
      self, audio: Union[str, np.ndarray], sample_rate: int = 16000
  ) -> tuple[np.ndarray, np.ndarray, float]:
    """Extracts strictly causal `(features, frame_speaker_embeddings, step_s)`.

    - Personal VAD 1.0 (`feature_dim == 40`): 40-dim log-Mel at 10ms frame step.
    - Personal VAD 2.0 (`feature_dim == 512`): 512-dim stacked log-Mel (4x128)
      at 30ms frame step.
    No speaker or non-causal utterance-level statistics are injected into
    `features`.
    """
    wav = self._load_audio(audio, sample_rate=sample_rate)
    mel_40_full = self._fe_v1(wav).numpy()
    mel_80_full = self._fe_80(wav).numpy()
    is_v1 = self.config.feature_dim == 40

    # Causal frame-level speaker embeddings for SC / ST / SET modes
    if self.subspace is not None and "proj_lda" in self.subspace:
      # Use a short causal window (25 frames = 250ms) for frame-level speaker
      # embeddings, reflecting frame-level speaker verification without neural
      # temporal adaptation.
      frame_embs_10ms = project_speaker_frames(
          mel_40_full, mel_80_full, self.subspace, window=25
      )
    else:
      frame_embs_10ms = np.zeros(
          (mel_40_full.shape[0], self.config.speaker_embedding_dim),
          dtype=np.float32,
      )

    if is_v1:
      if self.subspace is not None and "mel_mean_v1" in self.subspace:
        mu = self.subspace["mel_mean_v1"]
        sig = np.maximum(self.subspace["mel_std_v1"], 1e-5)
        features = (mel_40_full - mu[None, :]) / sig[None, :]
      else:
        features = mel_40_full
      return (
          features.astype(np.float32),
          frame_embs_10ms.astype(np.float32),
          0.010,
      )

    feat_v2_full = self._fe_v2(wav).numpy()
    if self.subspace is not None and "mel_mean_v2" in self.subspace:
      mu = self.subspace["mel_mean_v2"]
      sig = np.maximum(self.subspace["mel_std_v2"], 1e-5)
      features = (feat_v2_full - mu[None, :]) / sig[None, :]
    else:
      features = feat_v2_full

    num_v2 = features.shape[0]
    if frame_embs_10ms.shape[0] > 0 and num_v2 > 0:
      idx_30ms = np.minimum(
          np.arange(num_v2) * 3, frame_embs_10ms.shape[0] - 1
      )
      frame_embs_30ms = frame_embs_10ms[idx_30ms]
    else:
      frame_embs_30ms = np.zeros(
          (num_v2, self.config.speaker_embedding_dim), dtype=np.float32
      )
    return (
        features.astype(np.float32),
        frame_embs_30ms.astype(np.float32),
        0.030,
    )

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
    """Runs causal Personal VAD inference on raw audio."""
    wav = self._load_audio(audio, sample_rate=sample_rate)
    features, frame_embs, frame_step_sec = self.extract_acoustic_features(
        wav, sample_rate=16000
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
      cos_scores = dataset.compute_frame_cosine_similarity(
          frame_embs, spk_emb
      ).reshape(-1)
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
    sample_times = np.arange(len(wav), dtype=np.float32) / 16000.0
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
    is_enrollment_less = bool(enrollment_audio_or_embedding is None)
    summary = {
        "num_frames": int(num_frames),
        "frame_step_ms": round(frame_step_sec * 1000.0, 1),
        "frame_step_sec": round(float(frame_step_sec), 4),
        "enrollment_less": is_enrollment_less,
        "duration_sec": round(float(len(wav)) / 16000.0, 3),
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
        "probabilities": probs,
        "features": features,
        "frame_times_sec": frame_times,
        "frame_step_sec": float(frame_step_sec),
        "enrollment_less": is_enrollment_less,
        "target_speech_mask": target_mask,
        "target_mask": target_mask,
        "filtered_audio": filtered_audio,
        "speaker_embedding": spk_emb,
        "cosine_scores": cos_scores,
        "summary": summary,
    }

  def infer_from_files(
      self,
      audio_path: Union[str, np.ndarray],
      enrollment_path: Optional[Union[str, np.ndarray]] = None,
      enroll_path: Optional[Union[str, np.ndarray]] = None,
      enrollment_less: bool = False,
      use_tflite: bool = False,
      quantized: bool = True,
      threshold: float = 0.5,
      output_filtered_wav: Optional[str] = None,
  ) -> dict[str, Any]:
    """Runs Personal VAD inference directly from audio file paths or arrays."""
    del quantized  # Handled automatically by TFLite runner path selection.
    enroll_arg = None if enrollment_less else (enrollment_path or enroll_path)
    res = self.predict_audio(
        audio=audio_path,
        enrollment_audio_or_embedding=enroll_arg,
        use_tflite=use_tflite,
        threshold=threshold,
    )
    if output_filtered_wav:
      out_dir = os.path.dirname(output_filtered_wav)
      if out_dir:
        os.makedirs(out_dir, exist_ok=True)
      sf.write(output_filtered_wav, res["filtered_audio"], 16000)
    return res

  # Alias for backwards compatibility with any camel-cased typo calls
  infer_From_files = infer_from_files


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
