"""Personal VAD 1.0 and 2.0 neural network models.

Implements:
- Personal VAD 1.0 (arXiv:1908.04284): 2-layer unidirectional LSTM (64 units) +
  64-unit ReLU FC + 3-class linear output, supporting `ET`, `ST`, `SET`, and
  `SC` conditioning modes.
- Personal VAD 2.0 (arXiv:2204.03793):
  - 4-layer causal streaming Conformer (`model_dim=64`, `num_heads=8`,
    `left_context=31`, `right_context=0`, `kernel_size=7`, `ffn_multiplier=8`).
  - 3-layer Layer-Normalized LSTM (`256` units) with 256-unit ReLU input
    projection.
  - Speaker conditioning via input concatenation (`CONCAT`), Feature-wise Linear
    Modulation on target speaker d-vector (`FILM_DVECTOR`, E1), Speaker Pre-Net
    cosine similarity FiLM (`FILM_COS`, E2), Speaker Pre-Net cosine similarity
    concatenation (`CONCAT_COS`), and combined d-vector + cosine similarity FiLM
    (`FILM_DVECTOR_COS`, E3).
"""

from collections.abc import Mapping
from typing import Any, Optional, Union
import tensorflow as tf
from . import configs
from . import layers


class PersonalVadModel(tf.keras.Model):
  """Unified Keras model for Personal VAD 1.0 and Personal VAD 2.0."""

  def __init__(
      self,
      config: Optional[configs.ModelConfig] = None,
      name: str = "personal_vad_model",
  ):
    super().__init__(name=name)
    self.config = config or configs.ModelConfig.pvad_v1()
    self.input_proj: Optional[tf.keras.layers.Layer] = None
    self.post_fc: Optional[tf.keras.layers.Layer] = None
    self.speaker_prenet: Optional[layers.SpeakerPreNet] = None
    self.film_layer: Optional[layers.FeatureWiseModulationLayer] = None
    self.lstm_layers: list[tf.keras.layers.LSTM] = []
    self.lstm_norms: list[Optional[tf.keras.layers.LayerNormalization]] = []
    self.conformer_blocks: list[layers.ConformerBlock] = []
    self._build_submodules()

  def _uses_input_concat(self) -> bool:
    return self.config.conditioning_mode in (
        configs.ConditioningMode.ET,
        configs.ConditioningMode.ST,
        configs.ConditioningMode.SET,
        configs.ConditioningMode.CONCAT,
    )

  def _uses_prenet(self) -> bool:
    return self.config.conditioning_mode in (
        configs.ConditioningMode.FILM_COS,
        configs.ConditioningMode.CONCAT_COS,
        configs.ConditioningMode.FILM_DVECTOR_COS,
    )

  def _uses_film(self) -> bool:
    return self.config.conditioning_mode in (
        configs.ConditioningMode.FILM_DVECTOR,
        configs.ConditioningMode.FILM_COS,
        configs.ConditioningMode.FILM_DVECTOR_COS,
    )

  def _build_submodules(self) -> None:
    """Instantiates backbone, speaker Pre-Net, FiLM, and classifier layers."""
    cfg = self.config

    if cfg.backbone == configs.BackboneType.LSTM_V1:
      self.input_proj = None
      self.lstm_layers = [
          tf.keras.layers.LSTM(
              cfg.lstm_units,
              return_sequences=True,
              return_state=True,
              name=f"lstm_{i}",
          )
          for i in range(cfg.lstm_num_layers)
      ]
      self.lstm_norms = []
      self.post_fc = tf.keras.layers.Dense(
          cfg.fc_units, activation="relu", name="post_fc"
      )
      backbone_out_dim = cfg.fc_units

    elif cfg.backbone == configs.BackboneType.LSTM_V2:
      self.input_proj = tf.keras.layers.Dense(
          cfg.fc_units, activation="relu", name="input_fc"
      )
      self.lstm_layers = [
          tf.keras.layers.LSTM(
              cfg.lstm_units,
              return_sequences=True,
              return_state=True,
              name=f"lstm_{i}",
          )
          for i in range(cfg.lstm_num_layers)
      ]
      self.lstm_norms = [
          tf.keras.layers.LayerNormalization(
              epsilon=1e-6, name=f"lstm_ln_{i}"
          )
          if cfg.use_layer_norm
          else None
          for i in range(cfg.lstm_num_layers)
      ]
      self.post_fc = None
      backbone_out_dim = cfg.lstm_units

    elif cfg.backbone == configs.BackboneType.CONFORMER:
      self.input_proj = tf.keras.layers.Dense(
          cfg.conformer_dim, name="conformer_input_proj"
      )
      self.conformer_blocks = [
          layers.ConformerBlock(
              model_dim=cfg.conformer_dim,
              num_heads=cfg.conformer_num_heads,
              left_context=cfg.conformer_left_context,
              right_context=cfg.conformer_right_context,
              kernel_size=cfg.conformer_kernel_size,
              ffn_multiplier=cfg.conformer_ffn_multiplier,
              dropout_rate=cfg.dropout_rate,
              name=f"conformer_block_{i}",
          )
          for i in range(cfg.conformer_num_layers)
      ]
      self.post_fc = None
      backbone_out_dim = cfg.conformer_dim
    else:
      raise ValueError(f"Unsupported backbone: {cfg.backbone}")

    if self._uses_prenet():
      self.speaker_prenet = layers.SpeakerPreNet(
          num_layers=cfg.prenet_num_layers,
          model_dim=cfg.prenet_dim,
          output_dim=cfg.prenet_output_dim,
          num_heads=cfg.conformer_num_heads,
          left_context=cfg.conformer_left_context,
          right_context=cfg.conformer_right_context,
          kernel_size=cfg.conformer_kernel_size,
          ffn_multiplier=cfg.conformer_ffn_multiplier,
          dropout_rate=cfg.dropout_rate,
          name="speaker_prenet",
      )
    else:
      self.speaker_prenet = None

    if self._uses_film():
      self.film_layer = layers.FeatureWiseModulationLayer(
          feature_dim=backbone_out_dim,
          has_bias=cfg.film_has_bias,
          apply_residual=cfg.film_apply_residual,
          name="film_layer",
      )
    else:
      self.film_layer = None

    self.classifier = tf.keras.layers.Dense(
        cfg.num_classes, name="classifier"
    )

  def _unpack_inputs(
      self,
      inputs: Union[
          Mapping[str, tf.Tensor],
          tuple[tf.Tensor, ...],
          list[tf.Tensor],
          tf.Tensor,
      ],
      speaker_embedding: Optional[tf.Tensor] = None,
      cosine_score: Optional[tf.Tensor] = None,
  ) -> tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
    """Normalizes model inputs into `(features, spk_emb, cos_score)`."""
    if isinstance(inputs, Mapping):
      features = tf.convert_to_tensor(inputs["features"], dtype=tf.float32)
      spk_emb = inputs.get("speaker_embedding", speaker_embedding)
      if spk_emb is not None:
        spk_emb = tf.convert_to_tensor(spk_emb, dtype=tf.float32)
      cos_score = inputs.get("cosine_score", cosine_score)
      if cos_score is not None:
        cos_score = tf.convert_to_tensor(cos_score, dtype=tf.float32)
      return features, spk_emb, cos_score

    if isinstance(inputs, (tuple, list)):
      features = tf.convert_to_tensor(inputs[0], dtype=tf.float32)
      spk_emb = (
          tf.convert_to_tensor(inputs[1], dtype=tf.float32)
          if len(inputs) > 1 and inputs[1] is not None
          else speaker_embedding
      )
      cos_score = (
          tf.convert_to_tensor(inputs[2], dtype=tf.float32)
          if len(inputs) > 2 and inputs[2] is not None
          else cosine_score
      )
      return features, spk_emb, cos_score

    features = tf.convert_to_tensor(inputs, dtype=tf.float32)
    if speaker_embedding is not None:
      speaker_embedding = tf.convert_to_tensor(
          speaker_embedding, dtype=tf.float32
      )
    if cosine_score is not None:
      cosine_score = tf.convert_to_tensor(cosine_score, dtype=tf.float32)
    return features, speaker_embedding, cosine_score

  def _prepare_backbone_input(
      self,
      features: tf.Tensor,
      speaker_embedding: Optional[tf.Tensor],
      cosine_score: Optional[tf.Tensor],
  ) -> tf.Tensor:
    """Constructs the input tensor to the backbone network."""
    mode = self.config.conditioning_mode
    seq_len = tf.shape(features)[1]

    if mode == configs.ConditioningMode.SC or not self._uses_input_concat():
      return features

    parts = [features]
    if mode in (
        configs.ConditioningMode.ET,
        configs.ConditioningMode.SET,
        configs.ConditioningMode.CONCAT,
    ):
      if speaker_embedding is None:
        raise ValueError(
            f"speaker_embedding is required for conditioning mode {mode.value}"
        )
      norm_emb = layers.safe_l2_normalize(speaker_embedding, axis=-1)
      if norm_emb.shape.rank == 2:
        tiled_emb = tf.tile(tf.expand_dims(norm_emb, axis=1), [1, seq_len, 1])
      else:
        tiled_emb = norm_emb
      parts.append(tiled_emb)

    if mode in (configs.ConditioningMode.ST, configs.ConditioningMode.SET):
      if cosine_score is None:
        raise ValueError(
            f"cosine_score is required for conditioning mode {mode.value}"
        )
      if cosine_score.shape.rank == 2:
        cosine_score = tf.expand_dims(cosine_score, axis=-1)
      parts.append(cosine_score)

    return tf.concat(parts, axis=-1)

  def _apply_post_backbone_conditioning(
      self,
      hidden: tf.Tensor,
      raw_features: tf.Tensor,
      speaker_embedding: Optional[tf.Tensor],
      training: bool = False,
  ) -> tf.Tensor:
    """Applies Speaker Pre-Net and/or FiLM modulation after the backbone."""
    mode = self.config.conditioning_mode
    if mode in (
        configs.ConditioningMode.SC,
        configs.ConditioningMode.ST,
        configs.ConditioningMode.ET,
        configs.ConditioningMode.SET,
        configs.ConditioningMode.CONCAT,
    ):
      return hidden

    if speaker_embedding is None:
      raise ValueError(
          f"speaker_embedding is required for conditioning mode {mode.value}"
      )
    norm_emb = layers.safe_l2_normalize(speaker_embedding, axis=-1)

    if mode == configs.ConditioningMode.FILM_DVECTOR:
      assert self.film_layer is not None
      return self.film_layer(hidden, norm_emb)

    # Modes using SpeakerPreNet: FILM_COS, CONCAT_COS, FILM_DVECTOR_COS
    assert self.speaker_prenet is not None
    cos_sim = self.speaker_prenet(
        raw_features, norm_emb, training=training
    )
    assert isinstance(cos_sim, tf.Tensor)
    if mode == configs.ConditioningMode.FILM_COS:
      assert self.film_layer is not None
      return self.film_layer(hidden, cos_sim)
    if mode == configs.ConditioningMode.CONCAT_COS:
      return tf.concat([hidden, cos_sim], axis=-1)
    if mode == configs.ConditioningMode.FILM_DVECTOR_COS:
      assert self.film_layer is not None
      seq_len = tf.shape(hidden)[1]
      if norm_emb.shape.rank == 2:
        tiled_emb = tf.tile(tf.expand_dims(norm_emb, axis=1), [1, seq_len, 1])
      else:
        tiled_emb = norm_emb
      modulator = tf.concat([tiled_emb, cos_sim], axis=-1)
      return self.film_layer(hidden, modulator)

    raise ValueError(f"Unhandled conditioning mode: {mode}")

  def call(
      self,
      inputs: Any,
      training: bool = False,
      mask: Any = None,
      speaker_embedding: Optional[tf.Tensor] = None,
      cosine_score: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    """Computes frame-level Personal VAD logits `[batch, time, num_classes]`.

    Args:
      inputs: Either a dictionary with keys `"features"`, `"speaker_embedding"`,
        and optional `"cosine_score"`, a tuple `(features, speaker_embedding[,
        cosine_score])`, or the `features` tensor `[batch, time, feature_dim]`.
      training: Boolean indicating whether the call is in training mode.
      mask: Unused Keras mask argument for signature compatibility.
      speaker_embedding: Optional target speaker d-vector `[batch, emb_dim]`
        when `inputs` is the raw `features` tensor.
      cosine_score: Optional frame-level cosine similarity `[batch, time, 1]`.

    Returns:
      Float32 tensor of shape `[batch, time, num_classes]` containing
      unnormalized frame-level logits.
    """
    del mask
    features, spk_emb, cos_score = self._unpack_inputs(
        inputs, speaker_embedding=speaker_embedding, cosine_score=cosine_score
    )
    x = self._prepare_backbone_input(features, spk_emb, cos_score)

    if self.config.backbone == configs.BackboneType.LSTM_V1:
      for lstm in self.lstm_layers:
        x, _, _ = lstm(x, training=training)
      assert self.post_fc is not None
      x = self.post_fc(x)
    elif self.config.backbone == configs.BackboneType.LSTM_V2:
      assert self.input_proj is not None
      x = self.input_proj(x)
      for lstm, norm in zip(self.lstm_layers, self.lstm_norms):
        x, _, _ = lstm(x, training=training)
        if norm is not None:
          x = norm(x)
    elif self.config.backbone == configs.BackboneType.CONFORMER:
      assert self.input_proj is not None
      x = self.input_proj(x)
      for block in self.conformer_blocks:
        x = block(x, training=training)

    x = self._apply_post_backbone_conditioning(
        hidden=x,
        raw_features=features,
        speaker_embedding=spk_emb,
        training=training,
    )
    return self.classifier(x)

  def predict_proba(
      self,
      inputs: Union[
          Mapping[str, tf.Tensor],
          tuple[tf.Tensor, ...],
          list[tf.Tensor],
          tf.Tensor,
      ],
      speaker_embedding: Optional[tf.Tensor] = None,
      cosine_score: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    """Computes frame-level posterior probabilities."""
    logits = self(
        inputs,
        training=False,
        speaker_embedding=speaker_embedding,
        cosine_score=cosine_score,
    )
    return tf.nn.softmax(logits, axis=-1)

  def init_streaming_state(self, batch_size: int = 1) -> dict[str, Any]:
    """Returns initial recurrent/attention/conv state for streaming."""
    state: dict[str, Any] = {}
    cfg = self.config
    if cfg.backbone in (
        configs.BackboneType.LSTM_V1,
        configs.BackboneType.LSTM_V2,
    ):
      state["lstm"] = [
          (
              tf.zeros([batch_size, cfg.lstm_units], dtype=tf.float32),
              tf.zeros([batch_size, cfg.lstm_units], dtype=tf.float32),
          )
          for _ in range(cfg.lstm_num_layers)
      ]
    elif cfg.backbone == configs.BackboneType.CONFORMER:
      state["conformer"] = [
          block.init_cache(batch_size) for block in self.conformer_blocks
      ]

    if self.speaker_prenet is not None:
      state["prenet"] = self.speaker_prenet.init_cache(batch_size)

    return state

  def stream_step(
      self,
      features: tf.Tensor,
      speaker_embedding: Optional[tf.Tensor],
      state: dict[str, Any],
      cosine_score: Optional[tf.Tensor] = None,
  ) -> tuple[tf.Tensor, tf.Tensor, dict[str, Any]]:
    """Executes one streaming step over a frame or chunk of frames.

    Args:
      features: Float32 tensor `[batch, step_frames, feature_dim]`.
      speaker_embedding: Optional float32 tensor `[batch, emb_dim]`.
      state: State dictionary created by `init_streaming_state`.
      cosine_score: Optional float32 tensor `[batch, step_frames, 1]`.

    Returns:
      Tuple `(logits, probs, new_state)`.
    """
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    if speaker_embedding is not None:
      speaker_embedding = tf.convert_to_tensor(
          speaker_embedding, dtype=tf.float32
      )
    if cosine_score is not None:
      cosine_score = tf.convert_to_tensor(cosine_score, dtype=tf.float32)

    x = self._prepare_backbone_input(features, speaker_embedding, cosine_score)
    new_state: dict[str, Any] = {}

    if self.config.backbone == configs.BackboneType.LSTM_V1:
      new_lstm_states = []
      for lstm, prev_state in zip(self.lstm_layers, state["lstm"]):
        x, h, c = lstm(x, initial_state=prev_state, training=False)
        new_lstm_states.append((h, c))
      new_state["lstm"] = new_lstm_states
      assert self.post_fc is not None
      x = self.post_fc(x)

    elif self.config.backbone == configs.BackboneType.LSTM_V2:
      assert self.input_proj is not None
      x = self.input_proj(x)
      new_lstm_states = []
      for lstm, norm, prev_state in zip(
          self.lstm_layers, self.lstm_norms, state["lstm"]
      ):
        x, h, c = lstm(x, initial_state=prev_state, training=False)
        if norm is not None:
          x = norm(x)
        new_lstm_states.append((h, c))
      new_state["lstm"] = new_lstm_states

    elif self.config.backbone == configs.BackboneType.CONFORMER:
      assert self.input_proj is not None
      x = self.input_proj(x)
      new_conformer_caches = []
      for block, (attn_c, valid_c, conv_c) in zip(
          self.conformer_blocks, state["conformer"]
      ):
        x, new_attn_c, new_valid_c, new_conv_c = block.stream_step(
            x, attn_c, valid_c, conv_c
        )
        new_conformer_caches.append((new_attn_c, new_valid_c, new_conv_c))
      new_state["conformer"] = new_conformer_caches

    mode = self.config.conditioning_mode
    if mode == configs.ConditioningMode.FILM_DVECTOR:
      assert speaker_embedding is not None
      assert self.film_layer is not None
      norm_emb = layers.safe_l2_normalize(speaker_embedding, axis=-1)
      x = self.film_layer(x, norm_emb)
    elif mode in (
        configs.ConditioningMode.FILM_COS,
        configs.ConditioningMode.CONCAT_COS,
        configs.ConditioningMode.FILM_DVECTOR_COS,
    ):
      assert speaker_embedding is not None
      assert self.speaker_prenet is not None
      norm_emb = layers.safe_l2_normalize(speaker_embedding, axis=-1)
      cos_sim, new_prenet_caches = self.speaker_prenet.stream_step(
          features, norm_emb, state["prenet"]
      )
      new_state["prenet"] = new_prenet_caches
      if mode == configs.ConditioningMode.FILM_COS:
        assert self.film_layer is not None
        x = self.film_layer(x, cos_sim)
      elif mode == configs.ConditioningMode.CONCAT_COS:
        x = tf.concat([x, cos_sim], axis=-1)
      elif mode == configs.ConditioningMode.FILM_DVECTOR_COS:
        assert self.film_layer is not None
        seq_len = tf.shape(x)[1]
        tiled_emb = tf.tile(tf.expand_dims(norm_emb, axis=1), [1, seq_len, 1])
        modulator = tf.concat([tiled_emb, cos_sim], axis=-1)
        x = self.film_layer(x, modulator)

    logits = self.classifier(x)
    probs = tf.nn.softmax(logits, axis=-1)
    return logits, probs, new_state
