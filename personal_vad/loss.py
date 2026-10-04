"""Loss functions for Personal VAD, including Weighted Pairwise Loss.

Implements the weighted pairwise loss defined in Section 2.3 of Personal VAD
(arXiv:1908.04284).
"""

from collections.abc import Mapping
from typing import Optional
import tensorflow as tf


def normalize_weights(
    weights: Mapping[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
  """Normalizes pairwise weights so that their sum equals 1.0.

  If all weights are 0.0, returns a copy with 0.0 values unchanged.

  Args:
    weights: Mapping from `(class1, class2)` integer pairs to non-negative float
      weights.

  Returns:
    A new dictionary with normalized weights.

  Raises:
    ValueError: If any weight is negative.
  """
  total = 0.0
  for pair, weight in weights.items():
    if weight < 0.0:
      raise ValueError(f"Weight for pair {pair} must be non-negative: {weight}")
    total += float(weight)

  if total <= 0.0:
    return {pair: 0.0 for pair in weights}
  return {pair: float(weight) / total for pair, weight in weights.items()}


class WeightedPairwiseLoss(tf.keras.losses.Loss):
  """Weighted pairwise cross-entropy loss for Personal VAD.

  In Personal VAD, distinguishing between non-speech (`ns`, class 1) and
  non-target speaker speech (`ntss`, class 2) is less critical than accurately
  detecting target speaker speech (`tss`, class 0) against both `ns` and `ntss`.
  Additionally, training data often has fewer `<tss, ntss>` boundaries than
  `<tss, ns>` boundaries.

  For each frame with ground-truth label `y` and predicted logit vector `z`:
    `L_pair(z, y) = (1 / (C - 1)) * sum_{k != y} w(k, y) * CE([z_y, z_k], 0)`
  where `CE([z_y, z_k], 0) = -log(exp(z_y) / (exp(z_y) + exp(z_k)))`.
  """

  def __init__(
      self,
      weights: Optional[Mapping[tuple[int, int], float]] = None,
      num_classes: int = 3,
      symmetric_weights: bool = True,
      label_smoothing: float = 0.0,
      name: str = "weighted_pairwise_loss",
  ):
    """Initializes WeightedPairwiseLoss.

    Args:
      weights: Mapping from `(class1, class2)` tuple to float weight. For
        symmetric weights, keys must satisfy `0 <= class1 < class2 <
        num_classes`, and all `num_classes * (num_classes - 1) // 2` pairs must
        be present. For asymmetric weights, all
        `num_classes * (num_classes - 1)` ordered pairs `class1 != class2` must
        be present. If `None` and `num_classes == 3`, defaults to
        `{(0, 1): 1.0, (0, 2): 1.0, (1, 2): 0.1}` from the Personal VAD paper.
      num_classes: Total number of classes (default 3).
      symmetric_weights: Whether `w(i, j) == w(j, i)` (default True).
      label_smoothing: Float in `[0, 1)`. When > 0, smooths the binary pairwise
        targets `[1, 0]` to
        `[1 - 0.5 * label_smoothing, 0.5 * label_smoothing]`.
      name: Name of the loss instance.

    Raises:
      ValueError: If `num_classes < 2`, `label_smoothing` is out of bounds, or
        `weights` does not match the expected pairs for `num_classes`.
    """
    super().__init__(name=name)
    if num_classes < 2:
      raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    if not 0.0 <= label_smoothing < 1.0:
      raise ValueError(
          f"label_smoothing must be in [0, 1), got {label_smoothing}"
      )

    self.num_classes = num_classes
    self.symmetric_weights = symmetric_weights
    self.label_smoothing = label_smoothing

    if weights is None:
      if num_classes == 3:
        raw_weights = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 0.1}
        if not symmetric_weights:
          raw_weights = {
              (0, 1): 1.0,
              (1, 0): 1.0,
              (0, 2): 1.0,
              (2, 0): 1.0,
              (1, 2): 0.1,
              (2, 1): 0.1,
          }
      elif num_classes == 2:
        raw_weights = {(0, 1): 1.0}
        if not symmetric_weights:
          raw_weights[(1, 0)] = 1.0
      else:
        raise ValueError(
            "Explicit weights dictionary is required when num_classes > 3."
        )
    else:
      raw_weights = dict(weights)

    self._validate_weights(raw_weights)
    self.weights = normalize_weights(raw_weights)

    # Prebuild a [num_classes, num_classes] weight lookup table where entry
    # [y, k] is the pairwise weight for ground-truth class y against alternative
    # class k, and diagonal entries are 0.0.
    weight_matrix = [
        [0.0 for _ in range(num_classes)] for _ in range(num_classes)
    ]
    for class1 in range(num_classes):
      for class2 in range(num_classes):
        if class1 == class2:
          weight_matrix[class1][class2] = 0.0
        elif self.symmetric_weights:
          pair = (min(class1, class2), max(class1, class2))
          weight_matrix[class1][class2] = self.weights[pair]
        else:
          weight_matrix[class1][class2] = self.weights[(class1, class2)]
    self._weight_matrix = tf.constant(weight_matrix, dtype=tf.float32)

  def _validate_weights(
      self, weights: Mapping[tuple[int, int], float]
  ) -> None:
    """Validates keys of the pairwise weight dictionary."""
    if self.symmetric_weights:
      expected_num_pairs = self.num_classes * (self.num_classes - 1) // 2
    else:
      expected_num_pairs = self.num_classes * (self.num_classes - 1)

    if len(weights) != expected_num_pairs:
      raise ValueError(
          f"Expected {expected_num_pairs} weight pairs for "
          f"num_classes={self.num_classes} (symmetric={self.symmetric_weights})"
          f", but got {len(weights)}."
      )

    for pair in weights:
      if len(pair) != 2:
        raise ValueError(f"Weight key must be a 2-tuple, got {pair}")
      c1, c2 = pair
      if not (0 <= c1 < self.num_classes and 0 <= c2 < self.num_classes):
        raise ValueError(
            f"Classes in {pair} must be in [0, {self.num_classes - 1}]."
        )
      if c1 == c2:
        raise ValueError(f"Self-pair {pair} is not allowed.")
      if self.symmetric_weights and c1 >= c2:
        raise ValueError(
            f"For symmetric weights, keys must satisfy c1 < c2, got {pair}."
        )

  def get_pairwise_weight(
      self, class1: int, class2: tf.Tensor
  ) -> tf.Tensor:
    """Returns pairwise weights between `class2` (label) and `class1` (alt).

    Args:
      class1: An integer alternative class index in `[0, num_classes)`.
      class2: A 1-D integer tensor of ground-truth class labels.

    Returns:
      A 1-D float32 tensor of the same shape as `class2` containing
      `w(class2[i], class1)` (with `0.0` wherever `class2[i] == class1`).
    """
    class2_int = tf.cast(class2, tf.int32)
    col = self._weight_matrix[:, class1]
    return tf.gather(col, class2_int)

  def compute_per_frame_loss(
      self, logits: tf.Tensor, labels: tf.Tensor
  ) -> tf.Tensor:
    """Computes the vectorized per-frame weighted pairwise loss tensor.

    Args:
      logits: Float32 tensor of shape `[..., num_classes]`.
      labels: Integer tensor of shape `[...]` with values in
        `[0, num_classes - 1]`.

    Returns:
      Float32 tensor of shape `[...]` containing the weighted pairwise loss for
      each frame.
    """
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.int32)

    orig_shape = tf.shape(labels)
    flat_logits = tf.reshape(logits, [-1, self.num_classes])
    flat_labels = tf.reshape(labels, [-1])
    num_elements = tf.shape(flat_labels)[0]

    # Gather ground-truth class logit z_{n, y_n} for every frame: shape [N].
    true_label_logits = tf.gather(flat_logits, flat_labels, batch_dims=1)

    # Construct all [N, C, 2] pairwise logit pairs [z_{n, y_n}, z_{n, k}] at
    # once via tensor broadcasting.
    true_col = tf.tile(
        tf.reshape(true_label_logits, [-1, 1, 1]), [1, self.num_classes, 1]
    )
    alt_col = tf.expand_dims(flat_logits, axis=-1)
    pairwise_logits = tf.concat([true_col, alt_col], axis=-1)

    if self.label_smoothing > 0.0:
      smooth_pos = 1.0 - 0.5 * self.label_smoothing
      smooth_neg = 0.5 * self.label_smoothing
      target_probs = tf.constant([smooth_pos, smooth_neg], dtype=tf.float32)
      target_probs = tf.broadcast_to(target_probs, tf.shape(pairwise_logits))
      pair_ce = tf.nn.softmax_cross_entropy_with_logits(
          labels=target_probs, logits=pairwise_logits
      )
    else:
      zero_labels = tf.zeros(
          [num_elements, self.num_classes], dtype=tf.int32
      )
      pair_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=zero_labels, logits=pairwise_logits
      )

    # Look up pairwise weights w(y_n, k) of shape [N, C]; diagonal entries
    # (k == y_n) are 0.0 in self._weight_matrix.
    pair_weights = tf.gather(self._weight_matrix, flat_labels)
    total_loss = tf.reduce_sum(pair_ce * pair_weights, axis=-1) / float(
        self.num_classes - 1
    )
    return tf.reshape(total_loss, orig_shape)

  def compute_loss(
      self,
      logits: tf.Tensor,
      labels: tf.Tensor,
      mask: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    """Computes the scalar mean weighted pairwise loss.

    Args:
      logits: Float tensor of shape `[..., num_classes]`.
      labels: Integer tensor of shape `[...]`.
      mask: Optional float/boolean mask tensor of shape `[...]` (1.0 for valid
        frames, 0.0 for padded frames).

    Returns:
      Scalar float32 loss tensor.
    """
    per_frame_loss = self.compute_per_frame_loss(logits=logits, labels=labels)
    if mask is not None:
      mask_float = tf.cast(mask, tf.float32)
      masked_loss = per_frame_loss * mask_float
      denom = tf.maximum(tf.reduce_sum(mask_float), 1.0)
      return tf.reduce_sum(masked_loss) / denom
    return tf.reduce_mean(per_frame_loss)

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Keras `Loss.call` interface (`y_true` = labels, `y_pred` = logits)."""
    # If y_true has a trailing singleton dimension [..., 1], squeeze it.
    if (
        y_true.shape.rank is not None
        and y_pred.shape.rank is not None
        and y_true.shape.rank == y_pred.shape.rank
        and y_true.shape[-1] == 1
    ):
      y_true = tf.squeeze(y_true, axis=-1)
    return self.compute_per_frame_loss(logits=y_pred, labels=y_true)


class CrossEntropyLoss(tf.keras.losses.Loss):
  """Standard multi-class cross-entropy loss with optional label smoothing."""

  def __init__(
      self,
      num_classes: int = 3,
      label_smoothing: float = 0.0,
      name: str = "cross_entropy_loss",
  ):
    super().__init__(name=name)
    self.num_classes = num_classes
    self.label_smoothing = label_smoothing

  def compute_per_frame_loss(
      self, logits: tf.Tensor, labels: tf.Tensor
  ) -> tf.Tensor:
    """Computes per-frame cross-entropy loss from logits."""
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.int32)
    if self.label_smoothing > 0.0:
      one_hot = tf.one_hot(labels, depth=self.num_classes, dtype=tf.float32)
      smooth_positives = 1.0 - self.label_smoothing
      smooth_negatives = self.label_smoothing / float(self.num_classes)
      smoothed_labels = one_hot * smooth_positives + smooth_negatives
      return tf.nn.softmax_cross_entropy_with_logits(
          labels=smoothed_labels, logits=logits
      )
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits
    )

  def compute_loss(
      self,
      logits: tf.Tensor,
      labels: tf.Tensor,
      mask: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    """Computes scalar mean cross-entropy loss with optional frame mask."""
    per_frame_loss = self.compute_per_frame_loss(logits=logits, labels=labels)
    if mask is not None:
      mask_float = tf.cast(mask, tf.float32)
      denom = tf.maximum(tf.reduce_sum(mask_float), 1.0)
      return tf.reduce_sum(per_frame_loss * mask_float) / denom
    return tf.reduce_mean(per_frame_loss)

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    if (
        y_true.shape.rank is not None
        and y_pred.shape.rank is not None
        and y_true.shape.rank == y_pred.shape.rank
        and y_true.shape[-1] == 1
    ):
      y_true = tf.squeeze(y_true, axis=-1)
    return self.compute_per_frame_loss(logits=y_pred, labels=y_true)
