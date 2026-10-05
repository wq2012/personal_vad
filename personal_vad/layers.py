"""Neural network layers for Personal VAD 1.0 and 2.0.

Implements:
- `FeatureWiseModulationLayer` (FiLM) for speaker conditioning (Personal VAD 2.0
  Section 2.2, arXiv:2204.03793).
- Causal streaming `ConformerBlock` (Macaron FFN + local causal Multi-Head
  Self-Attention with left context + causal depthwise Convolution module).
- `SpeakerPreNet` (2-layer Conformer network predicting frame-level speaker
  embeddings and cosine similarity with the enrolled target speaker d-vector)
  from Section 2.2.2 of Personal VAD 2.0 (arXiv:2204.03793).
"""

from typing import Union
import tensorflow as tf


def safe_l2_normalize(
    x: tf.Tensor, axis: int = -1, epsilon: float = 1e-12
) -> tf.Tensor:
  """L2-normalizes a tensor while preserving zero vectors without NaNs.

  When enrollment-less conditioning replaces the target speaker d-vector with
  all zeros, standard normalization without zero-guarding or small epsilon can
  amplify numerical noise. This helper returns exact zeros when the norm is
  below `epsilon`.

  Args:
    x: Input float tensor.
    axis: Dimension along which to normalize.
    epsilon: Minimum squared norm threshold.

  Returns:
    L2-normalized tensor of the same shape and dtype as `x`.
  """
  x = tf.convert_to_tensor(x, dtype=tf.float32)
  sq_sum = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)
  inv_norm = tf.math.rsqrt(tf.maximum(sq_sum, epsilon))
  is_nonzero = tf.cast(sq_sum > epsilon, tf.float32)
  return x * inv_norm * is_nonzero


class FeatureWiseModulationLayer(tf.keras.layers.Layer):
  """Feature-wise Linear Modulation (FiLM) layer.

  Applies affine modulation conditioned on `modulator`:
    `scale = W_gamma * modulator (+ b_gamma)`
    `bias = W_beta * modulator (+ b_beta)`
    `output = feature * scale + bias (+ feature if apply_residual)`
  """

  def __init__(
      self,
      feature_dim: int,
      has_bias: bool = False,
      apply_residual: bool = False,
      name: str = "film_layer",
  ):
    super().__init__(name=name)
    self.feature_dim = feature_dim
    self.has_bias = has_bias
    self.apply_residual = apply_residual
    self.fc_scale = tf.keras.layers.Dense(
        feature_dim, use_bias=has_bias, name="fc_scale"
    )
    self.fc_bias = tf.keras.layers.Dense(
        feature_dim, use_bias=has_bias, name="fc_bias"
    )

  def call(self, feature: tf.Tensor, modulator: tf.Tensor) -> tf.Tensor:
    """Applies FiLM modulation to `feature` conditioned on `modulator`.

    Args:
      feature: Float32 tensor of shape `[batch, time, feature_dim]` or `[batch,
        feature_dim]`.
      modulator: Float32 tensor of shape `[batch, modulator_dim]` (utterance-
        level) or `[batch, time, modulator_dim]` (frame-level).

    Returns:
      Modulated feature tensor of the same shape as `feature`.
    """
    feature = tf.convert_to_tensor(feature, dtype=tf.float32)
    modulator = tf.convert_to_tensor(modulator, dtype=tf.float32)

    scale = self.fc_scale(modulator)
    bias = self.fc_bias(modulator)

    if feature.shape.rank == 3 and scale.shape.rank == 2:
      scale = tf.expand_dims(scale, axis=1)
      bias = tf.expand_dims(bias, axis=1)

    output = feature * scale + bias
    if self.apply_residual:
      output = output + feature
    return output


class ConformerFeedForward(tf.keras.layers.Layer):
  """Macaron-style Feed-Forward module for Conformer blocks."""

  def __init__(
      self,
      model_dim: int,
      expansion_factor: int = 8,
      dropout_rate: float = 0.0,
      residual_weight: float = 0.5,
      name: str = "ffn",
  ):
    super().__init__(name=name)
    self.model_dim = model_dim
    self.expansion_factor = expansion_factor
    self.dropout_rate = dropout_rate
    self.residual_weight = residual_weight
    inner_dim = model_dim * expansion_factor

    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="ln"
    )
    self.dense1 = tf.keras.layers.Dense(
        inner_dim, activation=tf.nn.swish, name="dense1"
    )
    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.dense2 = tf.keras.layers.Dense(model_dim, name="dense2")
    self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    x = self.layer_norm(inputs)
    x = self.dense1(x)
    x = self.dropout1(x, training=training)
    x = self.dense2(x)
    x = self.dropout2(x, training=training)
    return inputs + self.residual_weight * x


class CausalMultiHeadSelfAttention(tf.keras.layers.Layer):
  """Causal local Multi-Head Self-Attention with bounded left context.

  Matches the streaming Conformer attention configuration in Personal VAD 2.0
  (`num_heads=8`, `left_context=31`, `right_context=0`), where each frame `t`
  attends only to frames in `[max(0, t - left_context), t + right_context]`.
  Supports both full-sequence execution and stateful streaming execution.
  """

  def __init__(
      self,
      model_dim: int = 64,
      num_heads: int = 8,
      left_context: int = 31,
      right_context: int = 0,
      dropout_rate: float = 0.0,
      name: str = "causal_mhsa",
  ):
    super().__init__(name=name)
    if model_dim % num_heads != 0:
      raise ValueError(
          f"model_dim ({model_dim}) must be divisible by num_heads "
          f"({num_heads})."
      )
    self.model_dim = model_dim
    self.num_heads = num_heads
    self.head_dim = model_dim // num_heads
    self.left_context = left_context
    self.right_context = right_context
    self.dropout_rate = dropout_rate

    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="ln"
    )
    self.q_proj = tf.keras.layers.Dense(model_dim, name="q_proj")
    self.k_proj = tf.keras.layers.Dense(model_dim, name="k_proj")
    self.v_proj = tf.keras.layers.Dense(model_dim, name="v_proj")
    self.out_proj = tf.keras.layers.Dense(model_dim, name="out_proj")
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    reshaped = tf.reshape(
        x, [batch_size, seq_len, self.num_heads, self.head_dim]
    )
    return tf.transpose(reshaped, perm=[0, 2, 1, 3])

  def _combine_heads(self, x: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[2]
    transposed = tf.transpose(x, perm=[0, 2, 1, 3])
    return tf.reshape(transposed, [batch_size, seq_len, self.model_dim])

  def _local_causal_mask(self, seq_len: tf.Tensor) -> tf.Tensor:
    """Builds a `[1, 1, seq_len, seq_len]` boolean mask for local attention."""
    idx = tf.range(seq_len)
    # diff[i, j] = i - j (query index minus key index)
    diff = tf.expand_dims(idx, 1) - tf.expand_dims(idx, 0)
    valid = tf.logical_and(
        diff >= -self.right_context, diff <= self.left_context
    )
    return tf.reshape(valid, [1, 1, seq_len, seq_len])

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Computes local causal multi-head self-attention over a sequence.

    Args:
      inputs: Float32 tensor of shape `[batch, time, model_dim]`.
      training: Whether in training mode.

    Returns:
      Float32 tensor of shape `[batch, time, model_dim]`.
    """
    normed = self.layer_norm(inputs)
    q = self._split_heads(self.q_proj(normed))
    k = self._split_heads(self.k_proj(normed))
    v = self._split_heads(self.v_proj(normed))

    scale = tf.math.rsqrt(tf.cast(self.head_dim, tf.float32))
    logits = tf.matmul(q, k, transpose_b=True) * scale

    seq_len = tf.shape(inputs)[1]
    mask = self._local_causal_mask(seq_len)
    neg_inf = tf.constant(-1e9, dtype=tf.float32)
    masked_logits = tf.where(mask, logits, neg_inf)

    weights = tf.nn.softmax(masked_logits, axis=-1)
    weights = self.dropout(weights, training=training)
    context = tf.matmul(weights, v)
    out = self.out_proj(self._combine_heads(context))
    out = self.dropout(out, training=training)
    return inputs + out

  def init_cache(self, batch_size: int = 1) -> tuple[tf.Tensor, tf.Tensor]:
    """Returns initial `(cache_frames, valid_count)` for streaming inference."""
    cache = tf.zeros(
        [batch_size, self.left_context, self.model_dim], dtype=tf.float32
    )
    valid_count = tf.zeros([batch_size, 1], dtype=tf.int32)
    return cache, valid_count

  def stream_step(
      self,
      inputs: tf.Tensor,
      cache: tf.Tensor,
      valid_count: tf.Tensor,
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Executes one streaming step given new frame(s) and history cache.

    Args:
      inputs: Float32 tensor of shape `[batch, step_frames, model_dim]`.
      cache: Float32 tensor of shape `[batch, left_context, model_dim]` storing
        the most recent `left_context` normalized input frames.
      valid_count: Int32 tensor of shape `[batch, 1]` tracking how many frames
        in `cache` are populated from actual past frames.

    Returns:
      Tuple `(output, new_cache, new_valid_count)`.
    """
    normed = self.layer_norm(inputs)
    step_len = tf.shape(inputs)[1]
    if self.left_context > 0:
      full_kv_normed = tf.concat([cache, normed], axis=1)
      new_cache = full_kv_normed[:, -self.left_context:, :]
    else:
      full_kv_normed = normed
      new_cache = cache

    q = self._split_heads(self.q_proj(normed))
    k = self._split_heads(self.k_proj(full_kv_normed))
    v = self._split_heads(self.v_proj(full_kv_normed))

    scale = tf.math.rsqrt(tf.cast(self.head_dim, tf.float32))
    logits = tf.matmul(q, k, transpose_b=True) * scale

    # Build validity mask over the [left_context + step_len] keys for each query
    total_kv_len = self.left_context + step_len
    q_pos = tf.range(self.left_context, total_kv_len)  # [step_len]
    k_pos = tf.range(total_kv_len)  # [total_kv_len]
    diff = tf.expand_dims(q_pos, 1) - tf.expand_dims(k_pos, 0)
    window_valid = tf.logical_and(diff >= 0, diff <= self.left_context)
    window_valid = tf.reshape(window_valid, [1, 1, step_len, total_kv_len])

    # Mask out unpopulated initial cache slots (k_pos < left_context - valid)
    min_valid_k = self.left_context - tf.reshape(valid_count, [-1, 1, 1, 1])
    k_pos_4d = tf.reshape(k_pos, [1, 1, 1, total_kv_len])
    history_valid = k_pos_4d >= min_valid_k
    full_mask = tf.logical_and(window_valid, history_valid)

    neg_inf = tf.constant(-1e9, dtype=tf.float32)
    masked_logits = tf.where(full_mask, logits, neg_inf)
    weights = tf.nn.softmax(masked_logits, axis=-1)
    context = tf.matmul(weights, v)
    out = self.out_proj(self._combine_heads(context))
    new_valid_count = tf.minimum(
        valid_count + step_len, self.left_context
    )
    return inputs + out, new_cache, new_valid_count


class CausalConvModule(tf.keras.layers.Layer):
  """Causal depthwise convolution module for streaming Conformer blocks."""

  def __init__(
      self,
      model_dim: int = 64,
      kernel_size: int = 7,
      dropout_rate: float = 0.0,
      name: str = "causal_conv",
  ):
    super().__init__(name=name)
    self.model_dim = model_dim
    self.kernel_size = kernel_size
    self.dropout_rate = dropout_rate

    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="ln"
    )
    self.pointwise_conv1 = tf.keras.layers.Dense(
        model_dim * 2, name="pointwise_conv1"
    )
    self.depthwise_conv = tf.keras.layers.DepthwiseConv1D(
        kernel_size=kernel_size,
        strides=1,
        padding="valid",
        name="depthwise_conv",
    )
    self.conv_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="conv_ln"
    )
    self.pointwise_conv2 = tf.keras.layers.Dense(
        model_dim, name="pointwise_conv2"
    )
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def _glu(self, x: tf.Tensor) -> tf.Tensor:
    a, b = tf.split(x, num_or_size_splits=2, axis=-1)
    return a * tf.nn.sigmoid(b)

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    normed = self.layer_norm(inputs)
    pw1 = self.pointwise_conv1(normed)
    gated = self._glu(pw1)
    if self.kernel_size > 1:
      padded = tf.pad(
          gated,
          paddings=[[0, 0], [self.kernel_size - 1, 0], [0, 0]],
          mode="CONSTANT",
      )
    else:
      padded = gated
    dw = self.depthwise_conv(padded)
    dw = tf.nn.swish(self.conv_norm(dw))
    pw2 = self.pointwise_conv2(dw)
    pw2 = self.dropout(pw2, training=training)
    return inputs + pw2

  def init_cache(self, batch_size: int = 1) -> tf.Tensor:
    """Returns initial depthwise conv cache `[batch, kernel_size - 1, dim]`."""
    cache_len = max(self.kernel_size - 1, 0)
    return tf.zeros([batch_size, cache_len, self.model_dim], dtype=tf.float32)

  def stream_step(
      self, inputs: tf.Tensor, conv_cache: tf.Tensor
  ) -> tuple[tf.Tensor, tf.Tensor]:
    """Executes one causal convolution streaming step with explicit cache."""
    normed = self.layer_norm(inputs)
    pw1 = self.pointwise_conv1(normed)
    gated = self._glu(pw1)
    if self.kernel_size > 1:
      padded = tf.concat([conv_cache, gated], axis=1)
      new_cache = padded[:, -(self.kernel_size - 1):, :]
    else:
      padded = gated
      new_cache = conv_cache
    dw = self.depthwise_conv(padded)
    dw = tf.nn.swish(self.conv_norm(dw))
    pw2 = self.pointwise_conv2(dw)
    return inputs + pw2, new_cache


class ConformerBlock(tf.keras.layers.Layer):
  """Causal streaming Conformer block for Personal VAD 2.0.

  Consists of:
  1. First half-step Macaron Feed-Forward (`ffn1`)
  2. Causal local Multi-Head Self-Attention (`mhsa`)
  3. Causal Depthwise Convolution module (`conv`)
  4. Second half-step Macaron Feed-Forward (`ffn2`)
  5. Final LayerNormalization (`final_ln`)
  """

  def __init__(
      self,
      model_dim: int = 64,
      num_heads: int = 8,
      left_context: int = 31,
      right_context: int = 0,
      kernel_size: int = 7,
      ffn_multiplier: int = 8,
      dropout_rate: float = 0.0,
      name: str = "conformer_block",
  ):
    super().__init__(name=name)
    self.model_dim = model_dim
    self.ffn1 = ConformerFeedForward(
        model_dim=model_dim,
        expansion_factor=ffn_multiplier,
        dropout_rate=dropout_rate,
        name="ffn1",
    )
    self.mhsa = CausalMultiHeadSelfAttention(
        model_dim=model_dim,
        num_heads=num_heads,
        left_context=left_context,
        right_context=right_context,
        dropout_rate=dropout_rate,
        name="mhsa",
    )
    self.conv = CausalConvModule(
        model_dim=model_dim,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        name="conv",
    )
    self.ffn2 = ConformerFeedForward(
        model_dim=model_dim,
        expansion_factor=ffn_multiplier,
        dropout_rate=dropout_rate,
        name="ffn2",
    )
    self.final_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="final_ln"
    )

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    x = self.ffn1(inputs, training=training)
    x = self.mhsa(x, training=training)
    x = self.conv(x, training=training)
    x = self.ffn2(x, training=training)
    return self.final_ln(x)

  def init_cache(
      self, batch_size: int = 1
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Initializes `(attn_cache, attn_valid_count, conv_cache)`."""
    attn_cache, attn_valid_count = self.mhsa.init_cache(batch_size)
    conv_cache = self.conv.init_cache(batch_size)
    return attn_cache, attn_valid_count, conv_cache

  def stream_step(
      self,
      inputs: tf.Tensor,
      attn_cache: tf.Tensor,
      attn_valid_count: tf.Tensor,
      conv_cache: tf.Tensor,
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Executes one streaming step through the Conformer block."""
    x = self.ffn1(inputs, training=False)
    x, new_attn_cache, new_valid_count = self.mhsa.stream_step(
        x, attn_cache, attn_valid_count
    )
    x, new_conv_cache = self.conv.stream_step(x, conv_cache)
    x = self.ffn2(x, training=False)
    x = self.final_ln(x)
    return x, new_attn_cache, new_valid_count, new_conv_cache


class SpeakerPreNet(tf.keras.layers.Layer):
  """Speaker Pre-Net for Personal VAD 2.0 (Architectures E2 and E3).

  Reproduces the Speaker Pre-Net in Section 2.2.2 of Personal VAD 2.0
  (arXiv:2204.03793):
  - Projects acoustic features `x` to `prenet_dim` (64).
  - Applies `num_layers` (2) causal Conformer blocks.
  - Projects to `output_dim` (256) to form frame-level speaker embeddings
    `e_prenet`.
  - Computes frame-level cosine similarity `s_t = cos(e_prenet, e_target)` of
    shape `[batch, time, 1]`.
  """

  def __init__(
      self,
      num_layers: int = 2,
      model_dim: int = 64,
      output_dim: int = 256,
      num_heads: int = 8,
      left_context: int = 31,
      right_context: int = 0,
      kernel_size: int = 7,
      ffn_multiplier: int = 8,
      dropout_rate: float = 0.0,
      name: str = "speaker_prenet",
  ):
    super().__init__(name=name)
    self.num_layers = num_layers
    self.model_dim = model_dim
    self.output_dim = output_dim

    self.input_proj = tf.keras.layers.Dense(model_dim, name="input_proj")
    self.blocks = [
        ConformerBlock(
            model_dim=model_dim,
            num_heads=num_heads,
            left_context=left_context,
            right_context=right_context,
            kernel_size=kernel_size,
            ffn_multiplier=ffn_multiplier,
            dropout_rate=dropout_rate,
            name=f"block_{i}",
        )
        for i in range(num_layers)
    ]
    self.output_proj = tf.keras.layers.Dense(output_dim, name="output_proj")

  def compute_cosine_similarity(
      self, prenet_embeddings: tf.Tensor, target_embedding: tf.Tensor
  ) -> tf.Tensor:
    """Computes frame-level cosine similarity `[batch, time, 1]`."""
    norm_prenet = safe_l2_normalize(prenet_embeddings, axis=-1)
    norm_target = safe_l2_normalize(target_embedding, axis=-1)
    if norm_target.shape.rank == 2:
      norm_target = tf.expand_dims(norm_target, axis=1)
    return tf.reduce_sum(norm_prenet * norm_target, axis=-1, keepdims=True)

  def call(
      self,
      features: tf.Tensor,
      target_embedding: tf.Tensor,
      training: bool = False,
      return_embeddings: bool = False,
  ) -> Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
    """Computes frame-level cosine similarity (and optionally prenet d-vectors).

    Args:
      features: Acoustic features of shape `[batch, time, feature_dim]`.
      target_embedding: Target speaker d-vector of shape `[batch, emb_dim]`.
      training: Whether in training mode.
      return_embeddings: If True, returns `(cosine_sim, prenet_embeddings)`.

    Returns:
      Cosine similarity tensor of shape `[batch, time, 1]`, or a tuple of
      `(cosine_sim, prenet_embeddings)` if `return_embeddings=True`.
    """
    x = self.input_proj(features)
    for block in self.blocks:
      x = block(x, training=training)
    prenet_embeddings = self.output_proj(x)
    cosine_sim = self.compute_cosine_similarity(
        prenet_embeddings, target_embedding
    )
    if return_embeddings:
      return cosine_sim, prenet_embeddings
    return cosine_sim

  def init_cache(
      self, batch_size: int = 1
  ) -> list[tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """Initializes streaming caches for all Conformer blocks in the Pre-Net."""
    return [block.init_cache(batch_size) for block in self.blocks]

  def stream_step(
      self,
      features: tf.Tensor,
      target_embedding: tf.Tensor,
      caches: list[tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
  ) -> tuple[tf.Tensor, list[tuple[tf.Tensor, tf.Tensor, tf.Tensor]]]:
    """Executes one streaming step for the Speaker Pre-Net."""
    x = self.input_proj(features)
    new_caches = []
    for block, (attn_c, valid_c, conv_c) in zip(self.blocks, caches):
      x, new_attn_c, new_valid_c, new_conv_c = block.stream_step(
          x, attn_c, valid_c, conv_c
      )
      new_caches.append((new_attn_c, new_valid_c, new_conv_c))
    prenet_embeddings = self.output_proj(x)
    cosine_sim = self.compute_cosine_similarity(
        prenet_embeddings, target_embedding
    )
    return cosine_sim, new_caches
