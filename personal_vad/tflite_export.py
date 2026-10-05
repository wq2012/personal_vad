"""TFLite export, 8-bit quantization, and inference runner for Personal VAD.

Supports:
- Personal VAD 2.0 Section 2.1 (arXiv:2204.03793): exporting Personal VAD
  models to TensorFlow Lite format with optional 8-bit dynamic-range
  post-training quantization.
- Running TFLite inference over arbitrary-length acoustic feature sequences.
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
  """Exports a `PersonalVadModel` to TensorFlow Lite format.

  Args:
    pvad_model: Trained or initialized `PersonalVadModel` instance.
    output_path: Optional file path where the `.tflite` model will be written.
    quantize: If True, applies 8-bit dynamic range post-training quantization
      (`tf.lite.Optimize.DEFAULT`), reducing model size by ~75% as described in
      Personal VAD 2.0 Section 2.1.
    sequence_length: Number of frames per inference call in the exported TFLite
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

  def _forward_unrolled(
      features: tf.Tensor,
      speaker_embedding: Optional[tf.Tensor],
      cosine_score: Optional[tf.Tensor],
  ) -> tf.Tensor:
    if cfg.backbone == configs.BackboneType.CONFORMER:
      return pvad_model.predict_proba(
          features,
          speaker_embedding=speaker_embedding,
          cosine_score=cosine_score,
      )
    # For LSTM backbones, run `stream_step` on the chunk so `lstm.cell` is
    # unrolled into pure TFLITE_BUILTINS without TensorList ops.
    state = pvad_model.init_streaming_state(batch_size=1)
    _, probs, _ = pvad_model.stream_step(
        features,
        speaker_embedding=speaker_embedding,
        state=state,
        cosine_score=cosine_score,
    )
    return probs

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

  if not needs_speaker:

    @tf.function(input_signature=[feat_spec])
    def serve_sc(features: tf.Tensor) -> tf.Tensor:
      return _forward_unrolled(features, None, None)

    concrete_func = serve_sc.get_concrete_function()
  elif needs_cosine and cfg.conditioning_mode == configs.ConditioningMode.ST:

    @tf.function(input_signature=[feat_spec, cos_spec])
    def serve_st(features: tf.Tensor, cosine_score: tf.Tensor) -> tf.Tensor:
      return _forward_unrolled(features, None, cosine_score)

    concrete_func = serve_st.get_concrete_function()
  elif needs_cosine and cfg.conditioning_mode == configs.ConditioningMode.SET:

    @tf.function(input_signature=[feat_spec, spk_spec, cos_spec])
    def serve_set(
        features: tf.Tensor,
        speaker_embedding: tf.Tensor,
        cosine_score: tf.Tensor,
    ) -> tf.Tensor:
      return _forward_unrolled(features, speaker_embedding, cosine_score)

    concrete_func = serve_set.get_concrete_function()
  else:

    @tf.function(input_signature=[feat_spec, spk_spec])
    def serve_default(
        features: tf.Tensor, speaker_embedding: tf.Tensor
    ) -> tf.Tensor:
      return _forward_unrolled(features, speaker_embedding, None)

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
  """Runs Personal VAD inference using a TensorFlow Lite model."""

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

    # Identify input tensor indices by name or shape.
    self._features_idx = 0
    self._speaker_idx = None
    self._cosine_idx = None
    for idx, detail in enumerate(self.input_details):
      name = detail["name"].lower()
      shape = tuple(detail["shape"])
      if "feature" in name or (len(shape) == 3 and shape[-1] > 1):
        self._features_idx = idx
      elif "speaker" in name or len(shape) == 2:
        self._speaker_idx = idx
      elif "cosine" in name or (len(shape) == 3 and shape[-1] == 1):
        self._cosine_idx = idx

    feat_shape = self.input_details[self._features_idx]["shape"]
    self.chunk_length = int(feat_shape[1])
    self.feature_dim = int(feat_shape[2])

  def predict_chunk(
      self,
      features_chunk: np.ndarray,
      speaker_embedding: Optional[np.ndarray] = None,
      cosine_chunk: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Runs inference on a single chunk of length `self.chunk_length`."""
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

    self.interpreter.invoke()
    output = self.interpreter.get_tensor(self.output_details[0]["index"])
    return np.squeeze(output, axis=0)

  def predict(
      self,
      features: np.ndarray,
      speaker_embedding: Optional[np.ndarray] = None,
      cosine_score: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Runs inference on an arbitrary-length sequence of acoustic features.

    Pads the sequence to a multiple of `self.chunk_length`, evaluates chunk by
    chunk, and trims the output back to `num_frames`.

    Args:
      features: Float32 array of shape `[num_frames, feature_dim]`.
      speaker_embedding: Optional float32 array of shape `[emb_dim]`.
      cosine_score: Optional float32 array of shape `[num_frames, 1]` or
        `[num_frames]`.

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

      chunk_out = self.predict_chunk(
          chunk, speaker_embedding=speaker_embedding, cosine_chunk=cos_chunk
      )
      outputs.append(chunk_out[: end - start])

    return np.concatenate(outputs, axis=0)
