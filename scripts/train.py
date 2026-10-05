"""CLI script to train Personal VAD 1.0 and 2.0 models."""

import argparse
from collections.abc import Sequence
import dataclasses
import json
import os
import sys
from typing import Any, Optional, Union
import tensorflow as tf

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import configs  # noqa: E402
from personal_vad import dataset  # noqa: E402
from personal_vad import loss as loss_lib  # noqa: E402
from personal_vad import model as model_lib  # noqa: E402
from scripts import prepare_dataset  # noqa: E402


def build_model_config(
    backbone: str,
    conditioning_mode: str,
    feature_dim: Optional[int] = None,
    speaker_embedding_dim: int = 256,
) -> configs.ModelConfig:
  """Constructs a `ModelConfig` from CLI string identifiers."""
  cond = configs.ConditioningMode(conditioning_mode)
  if backbone == "lstm_v1":
    cfg = configs.ModelConfig.pvad_v1(conditioning_mode=cond)
  elif backbone == "lstm_v2":
    cfg = configs.ModelConfig.pvad_v2_lstm(conditioning_mode=cond)
  elif backbone == "conformer":
    cfg = configs.ModelConfig.pvad_v2_conformer(conditioning_mode=cond)
  else:
    raise ValueError(f"Unsupported backbone: {backbone}")

  if feature_dim is not None:
    cfg.feature_dim = feature_dim
  cfg.speaker_embedding_dim = speaker_embedding_dim
  return cfg


def build_loss(loss_config: configs.LossConfig) -> tf.keras.losses.Loss:
  """Constructs the training loss function from `LossConfig`."""
  if loss_config.loss_type == configs.LossType.WEIGHTED_PAIRWISE:
    if loss_config.num_classes == 3:
      weights = {
          (0, 1): loss_config.weight_01,
          (0, 2): loss_config.weight_02,
          (1, 2): loss_config.weight_12,
      }
    else:
      weights = None
    return loss_lib.WeightedPairwiseLoss(
        weights=weights,
        num_classes=loss_config.num_classes,
        symmetric_weights=loss_config.symmetric_weights,
        label_smoothing=loss_config.label_smoothing,
    )
  return loss_lib.CrossEntropyLoss(
      num_classes=loss_config.num_classes,
      label_smoothing=loss_config.label_smoothing,
  )


def train_model(
    pvad_model: model_lib.PersonalVadModel,
    train_dataset: tf.data.Dataset,
    loss_fn: Union[loss_lib.WeightedPairwiseLoss, loss_lib.CrossEntropyLoss],
    epochs: int = 5,
    learning_rate: float = 5e-5,
    checkpoint_dir: Optional[str] = None,
) -> list[float]:
  """Runs the training loop and returns a list of per-epoch mean losses.

  Args:
    pvad_model: `PersonalVadModel` to train.
    train_dataset: Batched `tf.data.Dataset` yielding `(inputs, labels, mask)`.
    loss_fn: `WeightedPairwiseLoss` or `CrossEntropyLoss` instance.
    epochs: Number of training epochs.
    learning_rate: Learning rate for Adam optimizer (`5e-5` in PVAD 1.0).
    checkpoint_dir: Optional directory to save model weights and config JSON.

  Returns:
    List of measured mean training loss values for each completed epoch.
  """
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  epoch_losses: list[float] = []
  uses_prenet = pvad_model.speaker_prenet is not None
  for b_in, _, _ in train_dataset.take(1):
    _ = pvad_model(b_in, training=False)

  @tf.function
  def _train_step(
      batch_inputs: dict[str, tf.Tensor],
      batch_labels: tf.Tensor,
      batch_mask: tf.Tensor,
  ) -> tf.Tensor:
    with tf.GradientTape() as tape:
      logits = pvad_model(batch_inputs, training=True)
      loss_val = loss_fn.compute_loss(
          logits=logits, labels=batch_labels, mask=batch_mask
      )
      if uses_prenet and pvad_model.speaker_prenet is not None:
        cos_sim = tf.squeeze(
            pvad_model.speaker_prenet(
                batch_inputs["features"],
                batch_inputs["speaker_embedding"],
                training=True,
            ),
            axis=-1,
        )
        target_cos = tf.where(
            tf.equal(batch_labels, 0),
            1.0,
            tf.where(tf.equal(batch_labels, 2), -0.2, 0.0),
        )
        speech_mask = (
            tf.cast(tf.not_equal(batch_labels, 1), tf.float32) * batch_mask
        )
        emb_sq = tf.reduce_sum(
            tf.square(batch_inputs["speaker_embedding"]),
            axis=-1,
            keepdims=True,
        )
        emb_nonzero = tf.cast(emb_sq > 1e-6, tf.float32)
        speech_mask = speech_mask * emb_nonzero
        prenet_loss = tf.reduce_sum(
            tf.square(cos_sim - target_cos) * speech_mask
        ) / tf.maximum(tf.reduce_sum(speech_mask), 1.0)
        loss_val = loss_val + 0.8 * prenet_loss
    grads = tape.gradient(loss_val, pvad_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, pvad_model.trainable_variables))
    return loss_val

  for _ in range(epochs):
    batch_losses = []
    for batch_inputs, batch_labels, batch_mask in train_dataset:
      loss_val = _train_step(batch_inputs, batch_labels, batch_mask)
      batch_losses.append(float(loss_val.numpy()))

    mean_epoch_loss = (
        sum(batch_losses) / float(len(batch_losses)) if batch_losses else 0.0
    )
    epoch_losses.append(mean_epoch_loss)

  if checkpoint_dir is not None:
    save_checkpoint(pvad_model, checkpoint_dir)

  return epoch_losses


def save_checkpoint(
    pvad_model: model_lib.PersonalVadModel, checkpoint_dir: str
) -> str:
  """Saves model weights (`.weights.h5` and `.safetensors`) and config JSON."""
  os.makedirs(checkpoint_dir, exist_ok=True)
  cfg_dict: dict[str, Any] = dataclasses.asdict(pvad_model.config)
  cfg_dict["backbone"] = pvad_model.config.backbone.value
  cfg_dict["conditioning_mode"] = pvad_model.config.conditioning_mode.value

  config_path = os.path.join(checkpoint_dir, "model_config.json")
  with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg_dict, f, indent=2)

  weights_path = os.path.join(checkpoint_dir, "model.weights.h5")
  pvad_model.save_weights(weights_path)

  try:
    from safetensors import numpy as st_np  # noqa: E402

    weights_list = pvad_model.get_weights()
    tensor_dict = {
        f"weight_{i:03d}": w for i, w in enumerate(weights_list)
    }
    if tensor_dict:
      st_np.save_file(
          tensor_dict, os.path.join(checkpoint_dir, "model.safetensors")
      )
  except ImportError:
    pass

  return weights_path


def load_checkpoint(checkpoint_dir: str) -> model_lib.PersonalVadModel:
  """Loads a `PersonalVadModel` and its weights from `checkpoint_dir`."""
  config_path = os.path.join(checkpoint_dir, "model_config.json")
  with open(config_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

  raw["backbone"] = configs.BackboneType(raw["backbone"])
  raw["conditioning_mode"] = configs.ConditioningMode(raw["conditioning_mode"])
  cfg = configs.ModelConfig(**raw)
  pvad_model = model_lib.PersonalVadModel(config=cfg)

  # Build variables with a 1-frame dummy input before loading weights
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
  weights_path = os.path.join(checkpoint_dir, "model.weights.h5")
  safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
  if os.path.exists(weights_path):
    pvad_model.load_weights(weights_path)
  elif os.path.exists(safetensors_path):
    from safetensors import numpy as st_np  # noqa: E402

    loaded = st_np.load_file(safetensors_path)
    ordered = [loaded[f"weight_{i:03d}"] for i in range(len(loaded))]
    pvad_model.set_weights(ordered)
  else:
    raise FileNotFoundError(
        f"Neither model.weights.h5 nor model.safetensors found in "
        f"{checkpoint_dir}"
    )
  return pvad_model


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(description="Train a Personal VAD model.")
  parser.add_argument(
      "--train_npz",
      type=str,
      required=True,
      help="Path to training dataset .npz file.",
  )
  parser.add_argument(
      "--checkpoint_dir",
      type=str,
      required=True,
      help="Directory to save trained model weights and config.",
  )
  parser.add_argument(
      "--backbone",
      type=str,
      choices=["lstm_v1", "lstm_v2", "conformer"],
      default="lstm_v1",
      help="Backbone neural architecture.",
  )
  parser.add_argument(
      "--conditioning_mode",
      type=str,
      choices=[m.value for m in configs.ConditioningMode],
      default="et",
      help="Speaker conditioning mode.",
  )
  parser.add_argument(
      "--loss_type",
      type=str,
      choices=["weighted_pairwise", "cross_entropy"],
      default="weighted_pairwise",
      help="Training loss function.",
  )
  parser.add_argument(
      "--epochs",
      type=int,
      default=5,
      help="Number of training epochs.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=8,
      help="Mini-batch size.",
  )
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=5e-5,
      help="Adam optimizer learning rate.",
  )
  parser.add_argument(
      "--enrollment_less_prob",
      type=float,
      default=0.2,
      help="Probability p_0 of enrollment-less conditioning during training.",
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
  tf.random.set_seed(args.seed)
  utterances = prepare_dataset.load_utterances_from_npz(args.train_npz)
  feature_dim = utterances[0].features.shape[1]
  emb_dim = utterances[0].speaker_embedding.shape[0]

  model_cfg = build_model_config(
      backbone=args.backbone,
      conditioning_mode=args.conditioning_mode,
      feature_dim=feature_dim,
      speaker_embedding_dim=emb_dim,
  )
  pvad_model = model_lib.PersonalVadModel(config=model_cfg)

  loss_cfg = configs.LossConfig(
      loss_type=configs.LossType(args.loss_type),
      num_classes=model_cfg.num_classes,
  )
  loss_fn = build_loss(loss_cfg)

  train_ds = dataset.create_tf_dataset(
      utterances,
      batch_size=args.batch_size,
      shuffle=True,
      enrollment_less_prob=args.enrollment_less_prob,
      seed=args.seed,
  )
  train_model(
      pvad_model=pvad_model,
      train_dataset=train_ds,
      loss_fn=loss_fn,
      epochs=args.epochs,
      learning_rate=args.learning_rate,
      checkpoint_dir=args.checkpoint_dir,
  )


if __name__ == "__main__":
  main()
