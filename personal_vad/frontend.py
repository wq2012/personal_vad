"""Acoustic feature extraction frontend for Personal VAD 1.0 and 2.0.

Supports:
- Personal VAD 1.0 (arXiv:1908.04284): 40-dimensional log-Mel filterbank
  energies with 25ms window and 10ms frame step.
- Personal VAD 2.0 (arXiv:2204.03793): 128-dimensional log-Mel filterbank
  energies with 32ms window and 10ms frame step, stacked with 3 left context
  frames and subsampled by a factor of 3 (producing 512-dimensional features at
  a 30ms frame rate).
- Optional Lingvo `MelAsrFrontend` backend (`lingvo.tasks.asr.frontend`).
- Mean/StdDev feature normalization and `concat_meanstd`.
"""

from typing import Optional
import numpy as np
import tensorflow as tf
from . import configs


def concat_meanstd(
    mean_1: np.ndarray,
    std_1: np.ndarray,
    mean_2: Optional[np.ndarray] = None,
    std_2: Optional[np.ndarray] = None,
    dim_2: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
  """Concatenates mean and standard deviation vectors for composite features.

  When conditioning on L2-normalized speaker embeddings (`ET` mode), the
  acoustic feature mean/std are concatenated with zero-mean and unit-variance
  statistics (or explicit `mean_2`, `std_2` if provided).

  Args:
    mean_1: 1-D NumPy array of acoustic feature means.
    std_1: 1-D NumPy array of acoustic feature standard deviations.
    mean_2: Optional 1-D NumPy array of secondary feature means. Defaults to
      zeros of length `dim_2`.
    std_2: Optional 1-D NumPy array of secondary feature standard deviations.
      Defaults to ones of length `dim_2`.
    dim_2: Dimension of the secondary feature when `mean_2` / `std_2` are
      omitted (default 256 for d-vector, or 1 for cosine score).

  Returns:
    Tuple `(concatenated_mean, concatenated_std)` of float32 1-D NumPy arrays.
  """
  mean_1 = np.asarray(mean_1, dtype=np.float32).reshape(-1)
  std_1 = np.asarray(std_1, dtype=np.float32).reshape(-1)
  if mean_1.shape != std_1.shape:
    raise ValueError(
        f"mean_1 shape {mean_1.shape} does not match std_1 shape {std_1.shape}"
    )
  if mean_2 is None:
    mean_2 = np.zeros((dim_2,), dtype=np.float32)
  else:
    mean_2 = np.asarray(mean_2, dtype=np.float32).reshape(-1)
  if std_2 is None:
    std_2 = np.ones((dim_2,), dtype=np.float32)
  else:
    std_2 = np.asarray(std_2, dtype=np.float32).reshape(-1)
  if mean_2.shape != std_2.shape:
    raise ValueError(
        f"mean_2 shape {mean_2.shape} does not match std_2 shape {std_2.shape}"
    )
  return (
      np.concatenate([mean_1, mean_2], axis=0),
      np.concatenate([std_1, std_2], axis=0),
  )


def stack_and_subsample_frames(
    features: tf.Tensor,
    left_context: int = 0,
    right_context: int = 0,
    frame_stride: int = 1,
) -> tf.Tensor:
  """Stacks neighboring context frames and subsamples along the time axis.

  Matches Lingvo `MelAsrFrontend` frame stacking and subsampling behavior.

  Args:
    features: Float32 tensor of shape `[batch, num_frames, feature_dim]`.
    left_context: Number of past frames to stack before the current frame.
    right_context: Number of future frames to stack after the current frame.
    frame_stride: Temporal stride for subsampling stacked frames.

  Returns:
    Float32 tensor of shape `[batch, out_frames, feature_dim * context_size]`.
  """
  if left_context == 0 and right_context == 0 and frame_stride == 1:
    return features

  context_size = 1 + left_context + right_context
  padded = tf.pad(
      features,
      paddings=[[0, 0], [left_context, right_context], [0, 0]],
      mode="CONSTANT",
  )
  num_frames = tf.shape(features)[1]
  slices = []
  for offset in range(context_size):
    slices.append(padded[:, offset:offset + num_frames, :])
  stacked = tf.concat(slices, axis=-1)
  if frame_stride > 1:
    stacked = stacked[:, ::frame_stride, :]
  return stacked


class LogMelFrontend(tf.keras.layers.Layer):
  """Extracts log-Mel filterbank features from 16kHz audio waveforms.

  Supports both pure TensorFlow `tf.signal` computation (default, TFLite
  compatible) and optional Lingvo `MelAsrFrontend` when `use_lingvo_frontend` is
  enabled in `FrontendConfig`.
  """

  def __init__(
      self,
      config: Optional[configs.FrontendConfig] = None,
      mean: Optional[np.ndarray] = None,
      std: Optional[np.ndarray] = None,
      name: str = "log_mel_frontend",
  ):
    super().__init__(trainable=False, name=name)
    self.config = config or configs.FrontendConfig.pvad_v1()
    self.frame_length = int(
        round(self.config.sample_rate * self.config.frame_size_ms / 1000.0)
    )
    self.frame_step = int(
        round(self.config.sample_rate * self.config.frame_step_ms / 1000.0)
    )
    # Next power of 2 for FFT length (512 for 16kHz with 25ms or 32ms window).
    fft_length = 1
    while fft_length < self.frame_length:
      fft_length *= 2
    self.fft_length = fft_length

    num_spectrogram_bins = self.fft_length // 2 + 1
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=self.config.num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=self.config.sample_rate,
        lower_edge_hertz=self.config.lower_edge_hertz,
        upper_edge_hertz=self.config.upper_edge_hertz,
        dtype=tf.float32,
    )
    self._mel_weight_matrix = tf.constant(mel_matrix, dtype=tf.float32)

    if mean is not None and std is not None:
      self._mean = tf.constant(mean, dtype=tf.float32)
      self._std = tf.constant(np.maximum(std, 1e-8), dtype=tf.float32)
    else:
      self._mean = None
      self._std = None

    self._lingvo_frontend = None
    if self.config.use_lingvo_frontend:
      self._init_lingvo_frontend()

  def _init_lingvo_frontend(self) -> None:
    """Initializes Lingvo `MelAsrFrontend`."""
    try:
      from lingvo.core import py_utils
      from lingvo.tasks.asr import frontend as asr_frontend
    except ImportError as exc:
      raise ImportError(
          "Lingvo is required when use_lingvo_frontend=True."
      ) from exc

    p = asr_frontend.MelAsrFrontend.Params()
    p.name = "frontend"
    p.sample_rate = float(self.config.sample_rate)
    p.frame_size_ms = float(self.config.frame_size_ms)
    p.frame_step_ms = float(self.config.frame_step_ms)
    p.num_bins = int(self.config.num_mel_bins)
    p.lower_edge_hertz = float(self.config.lower_edge_hertz)
    p.upper_edge_hertz = float(self.config.upper_edge_hertz)
    p.preemph = float(self.config.preemph)
    p.noise_scale = 0.0
    p.pad_end = False
    p.stack_left_context = int(self.config.stack_left_context)
    p.stack_right_context = int(self.config.stack_right_context)
    p.frame_stride = int(self.config.frame_stride)
    self._lingvo_frontend = p.Instantiate()
    self._lingvo_py_utils = py_utils

  def call(self, waveforms: tf.Tensor) -> tf.Tensor:
    """Extracts log-Mel features from audio waveform tensor.

    Args:
      waveforms: Float32 tensor of shape `[batch, num_samples]` or
        `[num_samples]`.

    Returns:
      Float32 tensor of shape `[batch, num_frames, output_dim]` (or
      `[num_frames, output_dim]` if input was 1-D).
    """
    waveforms = tf.convert_to_tensor(waveforms, dtype=tf.float32)
    squeeze_batch = False
    if waveforms.shape.rank == 1:
      waveforms = tf.expand_dims(waveforms, axis=0)
      squeeze_batch = True

    if self._lingvo_frontend is not None:
      src = self._lingvo_py_utils.NestedMap()
      src.pcm = waveforms
      src.paddings = tf.zeros_like(waveforms)
      out = self._lingvo_frontend.FPropDefaultTheta(src)
      features = out.frames
      if features.shape.rank == 4:
        features = tf.squeeze(features, axis=-1)
    else:
      if self.config.preemph > 0.0:
        first_sample = waveforms[:, :1]
        diff = waveforms[:, 1:] - self.config.preemph * waveforms[:, :-1]
        waveforms = tf.concat([first_sample, diff], axis=1)

      stfts = tf.signal.stft(
          waveforms,
          frame_length=self.frame_length,
          frame_step=self.frame_step,
          fft_length=self.fft_length,
          window_fn=tf.signal.hann_window,
          pad_end=False,
      )
      magnitude_spectrograms = tf.abs(stfts)
      mel_spectrograms = tf.matmul(
          magnitude_spectrograms, self._mel_weight_matrix
      )
      log_mel = tf.math.log(mel_spectrograms + self.config.log_offset)
      features = stack_and_subsample_frames(
          log_mel,
          left_context=self.config.stack_left_context,
          right_context=self.config.stack_right_context,
          frame_stride=self.config.frame_stride,
      )

    if self._mean is not None and self._std is not None:
      features = (features - self._mean) / self._std

    if squeeze_batch:
      return tf.squeeze(features, axis=0)
    return features
