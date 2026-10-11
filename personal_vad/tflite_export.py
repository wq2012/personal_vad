"""TFLite export, 8-bit quantization, and stateful streaming runner.

Supports:
- Personal VAD 2.0 Section 2.1 (arXiv:2204.03793): exporting Personal VAD
  models to TensorFlow Lite format with optional 8-bit dynamic-range
  post-training quantization.
- Explicit stateful streaming carry-over across chunks in TFLite (`state_in` ->
  `probs, state_out`) so long sequences preserve recurrent (LSTM) and
  attention/convolution (Conformer) state across chunk boundaries.
"""

import os
from typing import Optional, Union
import numpy as np
import tensorflow as tf
from . import configs
from . import model as model_lib


def _ensure_model_built(
    pvad_model: model_lib.PersonalVadModel, sequence_length: int
) -> None:
  """Runs a dummy forward pass to ensure all Keras variables are created."""
  cfg = pvad_model.config
  dummy_features = tf.zeros(
      [1, sequence_length, cfg.feature_dim], dtype=tf.float32
  )
  dummy_spk = tf.zeros([1, cfg.speaker_embedding_dim], dtype=tf.float32)
  dummy_cos = tf.zeros([1, sequence_length, 1], dtype=tf.float32)
  _ = pvad_model(
      {
          "features": dummy_features,
          "speaker_embedding": dummy_spk,
          "cosine_score": dummy_cos,
      },
      training=False,
  )


def export_to_tflite(
    pvad_model: model_lib.PersonalVadModel,
    output_path: Optional[str] = None,
    quantize: bool = True,
    sequence_length: int = 32,
) -> bytes:
  """Exports a `PersonalVadModel` to stateful streaming TensorFlow Lite format.

  Args:
    pvad_model: Trained or initialized `PersonalVadModel` instance.
    output_path: Optional file path where the `.tflite` model will be written.
    quantize: If True, applies 8-bit dynamic range post-training quantization
      (`tf.lite.Optimize.DEFAULT`), reducing model size by ~75% as described in
      Personal VAD 2.0 Section 2.1.
    sequence_length: Number of frames per inference chunk in the exported TFLite
      signature (default 32).

  Returns:
    Serialized `.tflite` flatbuffer bytes.
  """
  if sequence_length <= 0:
    raise ValueError(f"sequence_length must be positive, got {sequence_length}")

  _ensure_model_built(pvad_model, sequence_length=sequence_length)
  cfg = pvad_model.config
  needs_cosine = cfg.conditioning_mode in (
      configs.ConditioningMode.ST,
      configs.ConditioningMode.SET,
  )
  needs_speaker = cfg.conditioning_mode != configs.ConditioningMode.SC
  state_shape = pvad_model.streaming_state_shape(batch_size=1)

  def _forward_stateful(
      features: tf.Tensor,
      speaker_embedding: Optional[tf.Tensor],
      cosine_score: Optional[tf.Tensor],
      state_tensor: tf.Tensor,
  ) -> tuple[tf.Tensor, tf.Tensor]:
    state_dict = pvad_model.unpack_streaming_state(state_tensor)
    _, probs, new_state_dict = pvad_model.stream_step(
        features,
        speaker_embedding=speaker_embedding,
        state=state_dict,
        cosine_score=cosine_score,
    )
    new_state_tensor = pvad_model.pack_streaming_state(new_state_dict)
    return probs, new_state_tensor

  feat_spec = tf.TensorSpec(
      shape=[1, sequence_length, cfg.feature_dim],
      dtype=tf.float32,
      name="features",
  )
  spk_spec = tf.TensorSpec(
      shape=[1, cfg.speaker_embedding_dim],
      dtype=tf.float32,
      name="speaker_embedding",
  )
  cos_spec = tf.TensorSpec(
      shape=[1, sequence_length, 1],
      dtype=tf.float32,
      name="cosine_score",
  )
  state_spec = tf.TensorSpec(
      shape=state_shape,
      dtype=tf.float32,
      name="state_in",
  )

  if not needs_speaker:

    @tf.function(input_signature=[feat_spec, state_spec])
    def serve_sc(
        features: tf.Tensor, state_in: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
      return _forward_stateful(features, None, None, state_in)

    concrete_func = serve_sc.get_concrete_function()
  elif needs_cosine and cfg.conditioning_mode == configs.ConditioningMode.ST:

    @tf.function(input_signature=[feat_spec, cos_spec, state_spec])
    def serve_st(
        features: tf.Tensor, cosine_score: tf.Tensor, state_in: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
      return _forward_stateful(features, None, cosine_score, state_in)

    concrete_func = serve_st.get_concrete_function()
  elif needs_cosine and cfg.conditioning_mode == configs.ConditioningMode.SET:

    @tf.function(input_signature=[feat_spec, spk_spec, cos_spec, state_spec])
    def serve_set(
        features: tf.Tensor,
        speaker_embedding: tf.Tensor,
        cosine_score: tf.Tensor,
        state_in: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
      return _forward_stateful(
          features, speaker_embedding, cosine_score, state_in
      )

    concrete_func = serve_set.get_concrete_function()
  else:

    @tf.function(input_signature=[feat_spec, spk_spec, state_spec])
    def serve_default(
        features: tf.Tensor, speaker_embedding: tf.Tensor, state_in: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
      return _forward_stateful(features, speaker_embedding, None, state_in)

    concrete_func = serve_default.get_concrete_function()

  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [concrete_func], pvad_model
  )
  converter.experimental_lower_to_saved_model = False
  if quantize:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

  tflite_bytes = converter.convert()
  if output_path is not None:
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
      os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "wb") as f:
      f.write(tflite_bytes)
  return tflite_bytes


class TFLitePersonalVadRunner:
  """Runs stateful streaming Personal VAD inference using a TFLite model."""

  def __init__(self, model_source: Union[str, bytes]):
    """Initializes the TFLite interpreter.

    Args:
      model_source: Path to a `.tflite` file or raw `.tflite` flatbuffer bytes.
    """
    if isinstance(model_source, (bytes, bytearray)):
      self.interpreter = tf.lite.Interpreter(model_content=bytes(model_source))
    elif isinstance(model_source, str):
      self.interpreter = tf.lite.Interpreter(model_path=model_source)
    else:
      raise TypeError(
          "Expected str path or bytes for model_source, got "
          f"{type(model_source)}"
      )
    self.interpreter.allocate_tensors()
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()

    # Identify input tensor indices by shape/name.
    self._features_idx = 0
    self._speaker_idx = None
    self._cosine_idx = None
    self._state_in_idx = None
    for idx, detail in enumerate(self.input_details):
      name = detail["name"].lower()
      shape = tuple(detail["shape"])
      if len(shape) == 4 or "state" in name:
        self._state_in_idx = idx
      elif "feature" in name or (len(shape) == 3 and shape[-1] > 1):
        self._features_idx = idx
      elif "speaker" in name or len(shape) == 2:
        self._speaker_idx = idx
      elif "cosine" in name or (len(shape) == 3 and shape[-1] == 1):
        self._cosine_idx = idx

    # Identify output tensor indices (3-D probs vs 4-D state_out).
    self._probs_out_idx = 0
    self._state_out_idx = None
    for idx, detail in enumerate(self.output_details):
      shape = tuple(detail["shape"])
      if len(shape) == 4:
        self._state_out_idx = idx
      elif len(shape) == 3:
        self._probs_out_idx = idx

    feat_shape = self.input_details[self._features_idx]["shape"]
    self.chunk_length = int(feat_shape[1])
    self.feature_dim = int(feat_shape[2])
    self._state: Optional[np.ndarray] = None
    self._base_state: Optional[np.ndarray] = None
    self._pending_features: list[np.ndarray] = []
    self._pending_cos: list[float] = []
    self.reset_state()

  def reset_state(self) -> Optional[np.ndarray]:
    """Resets the internal streaming state to zeros and returns it."""
    if self._state_in_idx is not None:
      state_shape = tuple(self.input_details[self._state_in_idx]["shape"])
      self._state = np.zeros(state_shape, dtype=np.float32)
      self._base_state = self._state.copy()
    else:
      self._state = None
      self._base_state = None
    self._pending_features = []
    self._pending_cos = []
    return self._state

  def predict_chunk(
      self,
      features_chunk: np.ndarray,
      speaker_embedding: Optional[np.ndarray] = None,
      cosine_chunk: Optional[np.ndarray] = None,
      state: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Runs inference on a single chunk and updates streaming state."""
    feat = np.asarray(features_chunk, dtype=np.float32)
    if feat.ndim == 2:
      feat = np.expand_dims(feat, axis=0)
    self.interpreter.set_tensor(
        self.input_details[self._features_idx]["index"], feat
    )
    if self._speaker_idx is not None:
      if speaker_embedding is None:
        raise ValueError("TFLite model requires speaker_embedding input.")
      spk = np.asarray(speaker_embedding, dtype=np.float32).reshape(1, -1)
      self.interpreter.set_tensor(
          self.input_details[self._speaker_idx]["index"], spk
      )
    if self._cosine_idx is not None:
      if cosine_chunk is None:
        raise ValueError("TFLite model requires cosine_score input.")
      cos = np.asarray(cosine_chunk, dtype=np.float32).reshape(
          1, self.chunk_length, 1
      )
      self.interpreter.set_tensor(
          self.input_details[self._cosine_idx]["index"], cos
      )
    if self._state_in_idx is not None:
      active_state = state if state is not None else self._state
      if active_state is None:
        active_state = self.reset_state()
      self.interpreter.set_tensor(
          self.input_details[self._state_in_idx]["index"],
          np.asarray(active_state, dtype=np.float32),
      )

    self.interpreter.invoke()
    output = self.interpreter.get_tensor(
        self.output_details[self._probs_out_idx]["index"]
    )
    if self._state_out_idx is not None:
      self._state = self.interpreter.get_tensor(
          self.output_details[self._state_out_idx]["index"]
      ).copy()
    return np.squeeze(output, axis=0)

  def step_frame(
      self,
      feature_frame: np.ndarray,
      speaker_embedding: Optional[np.ndarray] = None,
      cosine_score: Optional[Union[float, np.ndarray]] = None,
  ) -> np.ndarray:
    """Executes a single causal streaming frame step `[num_classes]`.

    Uses causal block-state checkpointing so that stepping 1 frame at a time
    through a fixed-chunk TFLite flatbuffer produces bit-for-bit identical
    posteriors to full-chunk streaming without padding state pollution.
    """
    frame = np.asarray(feature_frame, dtype=np.float32).reshape(
        self.feature_dim
    )
    self._pending_features.append(frame)
    if cosine_score is not None:
      cos_val = float(np.asarray(cosine_score, dtype=np.float32).reshape(-1)[0])
      self._pending_cos.append(cos_val)
    elif self._cosine_idx is not None:
      raise ValueError("TFLite model requires cosine_score input.")

    cur_len = len(self._pending_features)
    chunk = np.zeros((self.chunk_length, self.feature_dim), dtype=np.float32)
    chunk[:cur_len] = np.stack(self._pending_features, axis=0)

    cos_chunk = None
    if self._cosine_idx is not None:
      cos_chunk = np.zeros((self.chunk_length, 1), dtype=np.float32)
      cos_chunk[:cur_len, 0] = np.asarray(self._pending_cos, dtype=np.float32)

    base = (
        self._base_state.copy()
        if self._base_state is not None
        else self._state
    )
    chunk_out = self.predict_chunk(
        chunk,
        speaker_embedding=speaker_embedding,
        cosine_chunk=cos_chunk,
        state=base,
    )
    frame_out = chunk_out[cur_len - 1].copy()

    if cur_len == self.chunk_length:
      if self._state is not None:
        self._base_state = self._state.copy()
      self._pending_features = []
      self._pending_cos = []
    return frame_out

  def predict(
      self,
      features: np.ndarray,
      speaker_embedding: Optional[np.ndarray] = None,
      cosine_score: Optional[np.ndarray] = None,
      reset_state: bool = True,
      chunk_size: Optional[int] = None,
      stateful: bool = True,
  ) -> np.ndarray:
    """Runs stateful streaming inference over an arbitrary-length sequence.

    Initializes streaming state to zeros (when `reset_state=True`), evaluates
    chunk by chunk while carrying recurrent/Conformer state across chunk
    boundaries, and trims the final chunk's padding back to `num_frames`.

    Args:
      features: Float32 array of shape `[num_frames, feature_dim]`.
      speaker_embedding: Optional float32 array of shape `[emb_dim]`.
      cosine_score: Optional float32 array of shape `[num_frames, 1]` or
        `[num_frames]`.
      reset_state: Whether to reset streaming state before the first chunk.
      chunk_size: Optional streaming chunk size in frames. Defaults to the
        flatbuffer's `self.chunk_length`.
      stateful: Whether to carry recurrent/Conformer state across chunks.

    Returns:
      Float32 array of shape `[num_frames, num_classes]` containing frame-level
      posterior probabilities.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
      raise ValueError(
          "Expected 2-D features [num_frames, feature_dim], got "
          f"{features.shape}"
      )
    num_frames = features.shape[0]
    if num_frames == 0:
      raise ValueError("features must have at least 1 frame.")

    if reset_state:
      self.reset_state()

    if not stateful:
      step = (
          int(chunk_size)
          if chunk_size is not None and chunk_size > 0
          else self.chunk_length
      )
      step = min(step, self.chunk_length)
      outputs = []
      for start in range(0, num_frames, step):
        self.reset_state()
        end = min(start + step, num_frames)
        chunk = features[start:end]
        pad_len = self.chunk_length - chunk.shape[0]
        if pad_len > 0:
          chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode="constant")
        cos_chunk = None
        if cosine_score is not None:
          cos_arr = np.asarray(cosine_score, dtype=np.float32).reshape(-1, 1)
          cos_chunk = cos_arr[start:end]
          if pad_len > 0:
            cos_chunk = np.pad(
                cos_chunk, ((0, pad_len), (0, 0)), mode="constant"
            )
        chunk_out = self.predict_chunk(
            chunk, speaker_embedding=speaker_embedding, cosine_chunk=cos_chunk
        )
        outputs.append(chunk_out[: end - start])
      return np.concatenate(outputs, axis=0)

    if (
        chunk_size is not None
        and chunk_size != self.chunk_length
        and chunk_size > 0
    ):
      cos_arr = (
          np.asarray(cosine_score, dtype=np.float32).reshape(-1)
          if cosine_score is not None
          else None
      )
      frame_outputs = []
      for t in range(num_frames):
        c_t = cos_arr[t] if cos_arr is not None else None
        frame_outputs.append(
            self.step_frame(
                features[t],
                speaker_embedding=speaker_embedding,
                cosine_score=c_t,
            )
        )
      return np.stack(frame_outputs, axis=0)

    outputs = []
    for start in range(0, num_frames, self.chunk_length):
      end = min(start + self.chunk_length, num_frames)
      chunk = features[start:end]
      pad_len = self.chunk_length - chunk.shape[0]
      if pad_len > 0:
        chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode="constant")

      cos_chunk = None
      if cosine_score is not None:
        cos_arr = np.asarray(cosine_score, dtype=np.float32).reshape(-1, 1)
        cos_chunk = cos_arr[start:end]
        if pad_len > 0:
          cos_chunk = np.pad(cos_chunk, ((0, pad_len), (0, 0)), mode="constant")

      base = (
          self._base_state.copy()
          if self._base_state is not None
          else self._state
      )
      chunk_out = self.predict_chunk(
          chunk,
          speaker_embedding=speaker_embedding,
          cosine_chunk=cos_chunk,
          state=base,
      )
      if pad_len == 0 and self._state is not None:
        self._base_state = self._state.copy()
      outputs.append(chunk_out[: end - start])

    return np.concatenate(outputs, axis=0)
