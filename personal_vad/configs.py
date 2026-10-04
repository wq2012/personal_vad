# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration dataclasses and enums for Personal VAD 1.0 and 2.0."""

import dataclasses
import enum


class FrameLabel(enum.IntEnum):
  """Frame-level ground truth labels for Personal VAD.

  Label values:
    0: SPEECH (`tss` - target speaker speech)
    1: SILENCE (`ns` - non-speech / silence)
    2: SPEECH_FROM_NON_TARGET_SPEAKER (`ntss` - non-target speaker speech)
  """

  SPEECH = 0
  SILENCE = 1
  SPEECH_FROM_NON_TARGET_SPEAKER = 2


class ConditioningMode(enum.Enum):
  """Speaker conditioning modes for Personal VAD 1.0 and 2.0.

  Personal VAD 1.0 modes (arXiv:1908.04284):
    SC: Score Combination baseline (standard 2-class VAD + speaker verification
      cosine similarity rescaled via online 5th/95th percentiles).
    ST: Score Conditioned Training (concatenates acoustic features and framewise
      cosine similarity score `[x_t, s_t]`).
    ET: Embedding Conditioned Training (concatenates acoustic features and
      target speaker d-vector `[x_t, e_target]`).
    SET: Score and Embedding Conditioned Training (`[x_t, e_target, s_t]`).

  Personal VAD 2.0 modes (arXiv:2204.03793):
    CONCAT: Input feature concatenation (`[x_t, e_target]`).
    FILM_DVECTOR: Feature-wise Linear Modulation (FiLM) using the target speaker
      d-vector `e_target` (Architecture E1).
    FILM_COS: Speaker Pre-Net predicts frame-level d-vector; cosine similarity
      with `e_target` modulates the backbone output via FiLM (Architecture E2).
    CONCAT_COS: Speaker Pre-Net cosine similarity is concatenated with the
      backbone output before the linear classifier.
    FILM_DVECTOR_COS: Both target speaker d-vector `e_target` and Speaker
      Pre-Net cosine similarity are concatenated and fed to the FiLM layer
      (Architecture E3).
  """

  SC = "sc"
  ST = "st"
  ET = "et"
  SET = "set"
  CONCAT = "concat"
  FILM_DVECTOR = "film_dvector"
  FILM_COS = "film_cos"
  CONCAT_COS = "concat_cos"
  FILM_DVECTOR_COS = "film_dvector_cos"


class BackboneType(enum.Enum):
  """Backbone neural network architectures for Personal VAD."""

  LSTM_V1 = "lstm_v1"
  LSTM_V2 = "lstm_v2"
  CONFORMER = "conformer"


class LossType(enum.Enum):
  """Training loss types for Personal VAD."""

  CROSS_ENTROPY = "cross_entropy"
  WEIGHTED_PAIRWISE = "weighted_pairwise"


@dataclasses.dataclass
class FrontendConfig:
  """Configuration for log-Mel acoustic feature extraction.

  Attributes:
    sample_rate: Audio sample rate in Hz (default 16000).
    frame_size_ms: Window length in milliseconds (25.0 for PVAD 1.0, 32.0 for
      PVAD 2.0).
    frame_step_ms: Frame hop length in milliseconds (10.0).
    num_mel_bins: Number of Mel filterbank channels (40 for PVAD 1.0, 128 for
      PVAD 2.0).
    lower_edge_hertz: Lower bound on the frequencies to be included in the Mel
      spectrum (125.0 Hz).
    upper_edge_hertz: Upper bound on the frequencies to be included in the Mel
      spectrum (7500.0 Hz).
    stack_left_context: Number of left context frames to stack (0 for PVAD 1.0,
      3 for PVAD 2.0).
    stack_right_context: Number of right context frames to stack (0).
    frame_stride: Temporal subsampling stride after frame stacking (1 for PVAD
      1.0, 3 for PVAD 2.0 resulting in 30ms frame rate).
    log_offset: Small positive offset added before taking the logarithm.
    preemph: Pre-emphasis filter coefficient (0.97 for Lingvo frontend, 0.0 to
      disable).
    use_lingvo_frontend: Whether to use Lingvo's `MelAsrFrontend` when Lingvo
      is installed.
  """

  sample_rate: int = 16000
  frame_size_ms: float = 25.0
  frame_step_ms: float = 10.0
  num_mel_bins: int = 40
  lower_edge_hertz: float = 125.0
  upper_edge_hertz: float = 7500.0
  stack_left_context: int = 0
  stack_right_context: int = 0
  frame_stride: int = 1
  log_offset: float = 1e-6
  preemph: float = 0.97
  use_lingvo_frontend: bool = False

  @property
  def output_dim(self) -> int:
    """Returns the feature dimension after frame stacking."""
    return self.num_mel_bins * (
        1 + self.stack_left_context + self.stack_right_context
    )

  @property
  def output_frame_step_ms(self) -> float:
    """Returns the effective output frame step in milliseconds."""
    return self.frame_step_ms * self.frame_stride

  @classmethod
  def pvad_v1(cls) -> "FrontendConfig":
    """Returns the Personal VAD 1.0 frontend configuration (40-dim, 10ms)."""
    return cls(
        sample_rate=16000,
        frame_size_ms=25.0,
        frame_step_ms=10.0,
        num_mel_bins=40,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0,
        stack_left_context=0,
        stack_right_context=0,
        frame_stride=1,
        use_lingvo_frontend=False,
    )

  @classmethod
  def pvad_v2(cls, use_lingvo_frontend: bool = False) -> "FrontendConfig":
    """Returns the Personal VAD 2.0 frontend configuration (512-dim, 30ms)."""
    return cls(
        sample_rate=16000,
        frame_size_ms=32.0,
        frame_step_ms=10.0,
        num_mel_bins=128,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0,
        stack_left_context=3,
        stack_right_context=0,
        frame_stride=3,
        use_lingvo_frontend=use_lingvo_frontend,
    )


@dataclasses.dataclass
class LossConfig:
  """Configuration for Personal VAD training loss.

  Attributes:
    loss_type: Either `LossType.WEIGHTED_PAIRWISE` or `LossType.CROSS_ENTROPY`.
    num_classes: Number of output classes (3 for Personal VAD, 2 for standard
      VAD).
    weight_01: Pairwise weight between class 0 (`tss`) and class 1 (`ns`).
    weight_02: Pairwise weight between class 0 (`tss`) and class 2 (`ntss`).
    weight_12: Pairwise weight between class 1 (`ns`) and class 2 (`ntss`).
    symmetric_weights: Whether pairwise weights are symmetric (`w(i,j)=w(j,i)`).
    label_smoothing: Optional label smoothing factor in `[0, 1)`.
  """

  loss_type: LossType = LossType.WEIGHTED_PAIRWISE
  num_classes: int = 3
  weight_01: float = 1.0
  weight_02: float = 1.0
  weight_12: float = 0.1
  symmetric_weights: bool = True
  label_smoothing: float = 0.0


@dataclasses.dataclass
class ModelConfig:
  """Configuration for Personal VAD 1.0 and 2.0 neural network models.

  Attributes:
    backbone: Neural network backbone (`LSTM_V1`, `LSTM_V2`, or `CONFORMER`).
    conditioning_mode: Speaker conditioning mode (`ConditioningMode`).
    num_classes: Number of output classes (3 for Personal VAD, 2 for SC
      standard VAD).
    feature_dim: Acoustic input feature dimension (40 for PVAD 1.0, 512 for PVAD
      2.0).
    speaker_embedding_dim: Target speaker embedding (d-vector) dimension (256).
    lstm_num_layers: Number of stacked unidirectional LSTM layers.
    lstm_units: Number of hidden units per LSTM layer.
    fc_units: Number of units in the fully-connected hidden layer (for LSTM_V1
      post-LSTM FC or LSTM_V2 pre-LSTM FC).
    use_layer_norm: Whether to apply Layer Normalization inside LSTM layers.
    conformer_num_layers: Number of Conformer blocks in the main backbone (4 in
      PVAD 2.0).
    conformer_dim: Model dimension of the Conformer blocks (64 in PVAD 2.0).
    conformer_num_heads: Number of self-attention heads (8 in PVAD 2.0).
    conformer_left_context: Number of left context frames for causal attention
      (31 in PVAD 2.0, giving a 32-frame window).
    conformer_right_context: Number of right context frames (0 for causal
      streaming).
    conformer_kernel_size: Depthwise convolution kernel size (7 in PVAD 2.0).
    conformer_ffn_multiplier: Feed-forward expansion factor (8 in PVAD 2.0).
    dropout_rate: Dropout probability applied during training.
    prenet_num_layers: Number of Conformer blocks in the Speaker Pre-Net (2 in
      PVAD 2.0 E2/E3).
    prenet_dim: Model dimension of the Speaker Pre-Net (64 in PVAD 2.0).
    prenet_output_dim: Output embedding dimension of the Speaker Pre-Net (256).
    film_has_bias: Whether linear projections in FiLM layer include bias terms.
    film_apply_residual: Whether FiLM layer adds a residual connection.
  """

  backbone: BackboneType = BackboneType.LSTM_V1
  conditioning_mode: ConditioningMode = ConditioningMode.ET
  num_classes: int = 3
  feature_dim: int = 40
  speaker_embedding_dim: int = 256
  lstm_num_layers: int = 2
  lstm_units: int = 64
  fc_units: int = 64
  use_layer_norm: bool = False
  conformer_num_layers: int = 4
  conformer_dim: int = 64
  conformer_num_heads: int = 8
  conformer_left_context: int = 31
  conformer_right_context: int = 0
  conformer_kernel_size: int = 7
  conformer_ffn_multiplier: int = 8
  dropout_rate: float = 0.0
  prenet_num_layers: int = 2
  prenet_dim: int = 64
  prenet_output_dim: int = 256
  film_has_bias: bool = False
  film_apply_residual: bool = False

  @classmethod
  def pvad_v1(
      cls, conditioning_mode: ConditioningMode = ConditioningMode.ET
  ) -> "ModelConfig":
    """Returns Personal VAD 1.0 configuration (~130K parameters for ET)."""
    num_classes = 2 if conditioning_mode == ConditioningMode.SC else 3
    return cls(
        backbone=BackboneType.LSTM_V1,
        conditioning_mode=conditioning_mode,
        num_classes=num_classes,
        feature_dim=40,
        speaker_embedding_dim=256,
        lstm_num_layers=2,
        lstm_units=64,
        fc_units=64,
        use_layer_norm=False,
    )

  @classmethod
  def standard_vad_v1(cls) -> "ModelConfig":
    """Returns a 2-class standard VAD configuration (used in SC baseline)."""
    return cls.pvad_v1(conditioning_mode=ConditioningMode.SC)

  @classmethod
  def pvad_v2_conformer(
      cls, conditioning_mode: ConditioningMode = ConditioningMode.FILM_DVECTOR
  ) -> "ModelConfig":
    """Returns Personal VAD 2.0 streaming Conformer configuration."""
    return cls(
        backbone=BackboneType.CONFORMER,
        conditioning_mode=conditioning_mode,
        num_classes=3,
        feature_dim=512,
        speaker_embedding_dim=256,
        conformer_num_layers=4,
        conformer_dim=64,
        conformer_num_heads=8,
        conformer_left_context=31,
        conformer_right_context=0,
        conformer_kernel_size=7,
        conformer_ffn_multiplier=8,
        prenet_num_layers=2,
        prenet_dim=64,
        prenet_output_dim=256,
        film_has_bias=False,
        film_apply_residual=False,
    )

  @classmethod
  def pvad_v2_lstm(
      cls, conditioning_mode: ConditioningMode = ConditioningMode.CONCAT
  ) -> "ModelConfig":
    """Returns Personal VAD 2.0 3-layer LayerNorm LSTM baseline config."""
    return cls(
        backbone=BackboneType.LSTM_V2,
        conditioning_mode=conditioning_mode,
        num_classes=3,
        feature_dim=512,
        speaker_embedding_dim=256,
        lstm_num_layers=3,
        lstm_units=256,
        fc_units=256,
        use_layer_norm=True,
    )


@dataclasses.dataclass
class DatasetConfig:
  """Configuration for multi-speaker Personal VAD dataset generation.

  Attributes:
    min_utterances: Minimum number of utterances to concatenate per sequence
      (default 1).
    max_utterances: Maximum number of utterances to concatenate per sequence
      (default 3).
    enrollment_less_prob: Probability `p_0` of replacing the target speaker
      d-vector with a zero vector and mapping `ntss` (2) labels to `tss` (0)
      for joint enrolled and enrollment-less training (Algorithm 1 in PVAD 2.0,
      default 0.2).
    convert_for_vad_baseline: If True, converts all `ntss` (2) labels to `tss`
      (0) for standard 2-class VAD training.
    true_speaker_prob: Probability of selecting the true speaker embedding in
      single-speaker utterance mode.
    batch_size: Mini-batch size for `tf.data.Dataset`.
    max_frames: Maximum number of frames per sequence when padding/truncating.
  """

  min_utterances: int = 1
  max_utterances: int = 3
  enrollment_less_prob: float = 0.2
  convert_for_vad_baseline: bool = False
  true_speaker_prob: float = 0.5
  batch_size: int = 16
  max_frames: int = 1600
