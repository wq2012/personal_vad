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


def warm_start_speaker_conditioning(
    pvad_model: model_lib.PersonalVadModel,
    subspace: Optional[dict[str, Any]] = None,
) -> None:
  """Initializes speaker-conditioning weights from linear acoustic-subspace.

  When `subspace` (fitted on training speakers only) contains `w_x_v1`,
  `w_x_v2`, and `w_e_32`, initializes the standard neural weights of
  `pvad_model` so gradient descent starts in the speaker-discriminative basin
  using 100% pure log-Mel acoustic features.
  """
  if subspace is None or "w_e_32" not in subspace:
    return

  cfg = pvad_model.config
  mode = cfg.conditioning_mode
  backbone = cfg.backbone
  if mode == configs.ConditioningMode.SC:
    return

  import numpy as np  # noqa: E402

  dummy = {
      "features": tf.zeros([1, 4, cfg.feature_dim], dtype=tf.float32),
      "speaker_embedding": tf.zeros(
          [1, cfg.speaker_embedding_dim], dtype=tf.float32
      ),
      "cosine_score": tf.zeros([1, 4, 1], dtype=tf.float32),
  }
  _ = pvad_model(dummy, training=False)

  we = np.asarray(subspace["w_e_32"], dtype=np.float32)
  we_n = we / (np.std(we, axis=0, keepdims=True) + 1e-5)

  if backbone == configs.BackboneType.LSTM_V1 and "w_x_v1" in subspace:
    wx = np.asarray(subspace["w_x_v1"], dtype=np.float32)
    wx_n = wx / (np.std(wx, axis=0, keepdims=True) + 1e-5) * 0.14
    we_v1 = we_n * 0.28
    k0, rk0, b0 = pvad_model.lstm_layers[0].get_weights()
    k1, rk1, b1 = pvad_model.lstm_layers[1].get_weights()

    if mode in (configs.ConditioningMode.ET, configs.ConditioningMode.SET):
      n_pairs = 24
      for k in range(n_pairs):
        u_pos = 2 * k
        u_neg = 2 * k + 1
        k0[:, u_pos] = 0.0
        k0[:, u_neg] = 0.0
        b0[u_pos] = -1.0
        b0[u_neg] = -1.0
        k0[:, 64 + u_pos] = 0.0
        k0[:, 64 + u_neg] = 0.0
        b0[64 + u_pos] = 2.4
        b0[64 + u_neg] = 2.4
        rk0[:, 64 + u_pos] = 0.0
        rk0[:, 64 + u_neg] = 0.0
        rk0[:, 128 + u_pos] = 0.0
        rk0[:, 128 + u_neg] = 0.0
        k0[:, 128 + u_pos] = 0.0
        k0[:, 128 + u_neg] = 0.0
        k0[:40, 128 + u_pos] = wx_n[:, k]
        k0[:40, 128 + u_neg] = -wx_n[:, k]
        b0[128 + u_pos] = 0.0
        b0[128 + u_neg] = 0.0
        k0[:, 192 + u_pos] = 0.0
        k0[:, 192 + u_neg] = 0.0
        k0[40:296, 192 + u_pos] = we_v1[:, k]
        k0[40:296, 192 + u_neg] = -we_v1[:, k]
        b0[192 + u_pos] = 0.0
        b0[192 + u_neg] = 0.0
        rk0[:, 192 + u_pos] = 0.0
        rk0[:, 192 + u_neg] = 0.0
        k1[u_pos, 128 + k] = 2.5
        k1[u_neg, 128 + k] = 2.5
        b1[64 + k] = 2.0

      if mode == configs.ConditioningMode.SET and k0.shape[0] >= 297:
        k0[296, 128 + 48] = 2.0
        b0[64 + 48] = 2.2
        k1[48, 128 + 24] = 2.0
      pvad_model.lstm_layers[0].set_weights([k0, rk0, b0])
      pvad_model.lstm_layers[1].set_weights([k1, rk1, b1])

    elif mode == configs.ConditioningMode.ST and k0.shape[0] >= 41:
      f_biases = [0.8, 1.8, 2.6, 3.2]
      for k, fb in enumerate(f_biases):
        k0[40, 128 + k] = 2.0
        b0[64 + k] = fb
        k1[k, 128 + k] = 2.0
        b1[64 + k] = 1.8
      pvad_model.lstm_layers[0].set_weights([k0, rk0, b0])
      pvad_model.lstm_layers[1].set_weights([k1, rk1, b1])

  elif backbone == configs.BackboneType.LSTM_V2 and "w_x_v2" in subspace:
    if mode == configs.ConditioningMode.CONCAT and pvad_model.input_proj:
      wx = np.asarray(subspace["w_x_v2"], dtype=np.float32)
      wx_n = wx / (np.std(wx, axis=0, keepdims=True) + 1e-5) * 0.045
      we_v2 = we_n * 0.18
      w_in, b_in = pvad_model.input_proj.get_weights()
      w_in[:512, 0:32] = wx_n
      w_in[512:768, 0:32] = we_v2
      w_in[:512, 32:64] = -wx_n
      w_in[512:768, 32:64] = -we_v2
      w_in[:512, 64:96] = wx_n
      w_in[512:768, 64:96] = -we_v2
      w_in[:512, 96:128] = -wx_n
      w_in[512:768, 96:128] = we_v2
      w_in[512:768, 128:] = 0.0
      b_in[:128] = 0.0
      pvad_model.input_proj.set_weights([w_in, b_in])

      k0, rk0, b0 = pvad_model.lstm_layers[0].get_weights()
      for k in range(32):
        k0[:, 512 + k] = 0.0
        k0[k, 512 + k] = 1.2
        k0[32 + k, 512 + k] = 1.2
        k0[64 + k, 512 + k] = -1.2
        k0[96 + k, 512 + k] = -1.2
        b0[256 + k] = 2.2
      pvad_model.lstm_layers[0].set_weights([k0, rk0, b0])

  elif backbone == configs.BackboneType.CONFORMER and "w_x_v2" in subspace:
    wx = np.asarray(subspace["w_x_v2"], dtype=np.float32)
    wx_n = wx / (np.std(wx, axis=0, keepdims=True) + 1e-5) * 0.05
    we_v2 = we_n * 0.22

    if mode == configs.ConditioningMode.CONCAT and pvad_model.input_proj:
      w_in, b_in = pvad_model.input_proj.get_weights()
      w_in[:512, 0:24] = wx_n[:, :24]
      w_in[512:768, 0:24] = 0.0
      w_in[:512, 24:48] = 0.0
      w_in[512:768, 24:48] = we_v2[:, :24]
      w_in[512:768, 48:64] = 0.0
      b_in[:48] = 0.0
      pvad_model.input_proj.set_weights([w_in, b_in])

      for ffn in (
          pvad_model.conformer_blocks[0].ffn1,
          pvad_model.conformer_blocks[0].ffn2,
      ):
        w1, b1 = ffn.dense1.get_weights()
        w2, b2 = ffn.dense2.get_weights()
        for k in range(24):
          w1[:, k] = 0.0
          w1[:, 24 + k] = 0.0
          w1[:, 48 + k] = 0.0
          w1[:, 72 + k] = 0.0
          w1[k, k] = 1.2
          w1[24 + k, k] = 1.2
          w1[k, 24 + k] = -1.2
          w1[24 + k, 24 + k] = -1.2
          w1[k, 48 + k] = 1.2
          w1[24 + k, 48 + k] = -1.2
          w1[k, 72 + k] = -1.2
          w1[24 + k, 72 + k] = 1.2
          b1[k] = 0.0
          b1[24 + k] = 0.0
          b1[48 + k] = 0.0
          b1[72 + k] = 0.0
          w2[k, k] = 1.2
          w2[24 + k, k] = 1.2
          w2[48 + k, k] = -1.2
          w2[72 + k, k] = -1.2
        ffn.dense1.set_weights([w1, b1])
        ffn.dense2.set_weights([w2, b2])

    elif (
        mode == configs.ConditioningMode.FILM_DVECTOR
        and pvad_model.input_proj is not None
        and pvad_model.film_layer is not None
    ):
      w_in, b_in = pvad_model.input_proj.get_weights()
      w_in[:, :32] = wx_n[:, :32]
      b_in[:32] = 0.0
      pvad_model.input_proj.set_weights([w_in, b_in])

      sc_weights = pvad_model.film_layer.fc_scale.get_weights()
      bi_weights = pvad_model.film_layer.fc_bias.get_weights()
      w_sc = sc_weights[0]
      w_sc[:, :32] = we_v2[:, :32] * 1.6
      w_sc[:, 32:] = 0.0
      if len(sc_weights) > 1:
        b_sc = sc_weights[1]
        b_sc[:32] = 0.0
        b_sc[32:] = 1.0
        pvad_model.film_layer.fc_scale.set_weights([w_sc, b_sc])
      else:
        pvad_model.film_layer.fc_scale.set_weights([w_sc])

      w_bi = np.zeros_like(bi_weights[0])
      if len(bi_weights) > 1:
        pvad_model.film_layer.fc_bias.set_weights(
            [w_bi, np.zeros_like(bi_weights[1])]
        )
      else:
        pvad_model.film_layer.fc_bias.set_weights([w_bi])

      w_cl, b_cl = pvad_model.classifier.get_weights()
      w_cl[:32, 0] = 0.45
      w_cl[:32, 1] = 0.0
      w_cl[:32, 2] = -0.45
      pvad_model.classifier.set_weights([w_cl, b_cl])

    elif (
        mode
        in (
            configs.ConditioningMode.FILM_COS,
            configs.ConditioningMode.CONCAT_COS,
            configs.ConditioningMode.FILM_DVECTOR_COS,
        )
        and pvad_model.speaker_prenet is not None
    ):
      w_pin, b_pin = pvad_model.speaker_prenet.input_proj.get_weights()
      w_pin[:, :32] = wx_n[:, :32] * 1.2
      b_pin[:32] = 0.0
      pvad_model.speaker_prenet.input_proj.set_weights([w_pin, b_pin])

      w_pout, b_pout = pvad_model.speaker_prenet.output_proj.get_weights()
      we_unit = we / (np.linalg.norm(we, axis=0, keepdims=True) + 1e-5)
      w_pout[:32, :] = we_unit.T * 1.5
      w_pout[32:, :] *= 0.05
      b_pout[:] = 0.0
      pvad_model.speaker_prenet.output_proj.set_weights([w_pout, b_pout])

      if (
          mode == configs.ConditioningMode.FILM_COS
          and pvad_model.film_layer is not None
      ):
        sc_w = pvad_model.film_layer.fc_scale.get_weights()
        bi_w = pvad_model.film_layer.fc_bias.get_weights()
        w_sc = np.zeros_like(sc_w[0])
        w_bi = np.zeros_like(bi_w[0])
        w_bi[0, 0] = 2.2
        w_bi[0, 1] = -2.2
        if len(sc_w) > 1:
          pvad_model.film_layer.fc_scale.set_weights(
              [w_sc, np.ones_like(sc_w[1])]
          )
          pvad_model.film_layer.fc_bias.set_weights(
              [w_bi, np.zeros_like(bi_w[1])]
          )
        else:
          pvad_model.film_layer.fc_scale.set_weights([w_sc])
          pvad_model.film_layer.fc_bias.set_weights([w_bi])
        w_cl, b_cl = pvad_model.classifier.get_weights()
        w_cl[0, 0] = 1.0
        w_cl[0, 1] = 0.0
        w_cl[0, 2] = -1.0
        w_cl[1, 0] = -1.0
        w_cl[1, 1] = 0.0
        w_cl[1, 2] = 1.0
        pvad_model.classifier.set_weights([w_cl, b_cl])

      elif (
          mode == configs.ConditioningMode.FILM_DVECTOR_COS
          and pvad_model.film_layer is not None
          and pvad_model.input_proj is not None
      ):
        w_in, b_in = pvad_model.input_proj.get_weights()
        w_in[:, 2:26] = wx_n[:, :24]
        b_in[2:26] = 0.0
        pvad_model.input_proj.set_weights([w_in, b_in])

        sc_w = pvad_model.film_layer.fc_scale.get_weights()
        bi_w = pvad_model.film_layer.fc_bias.get_weights()
        w_sc = np.zeros_like(sc_w[0])
        w_sc[:256, 2:26] = we_v2[:, :24] * 1.5
        w_bi = np.zeros_like(bi_w[0])
        w_bi[256, 0] = 2.2
        w_bi[256, 1] = -2.2
        w_bi[255, 26] = 2.5
        if len(sc_w) > 1:
          b_sc = np.ones_like(sc_w[1])
          b_sc[2:26] = 0.0
          b_bi = np.zeros_like(bi_w[1])
          b_bi[26] = -1.0
          pvad_model.film_layer.fc_scale.set_weights([w_sc, b_sc])
          pvad_model.film_layer.fc_bias.set_weights([w_bi, b_bi])
        else:
          pvad_model.film_layer.fc_scale.set_weights([w_sc])
          pvad_model.film_layer.fc_bias.set_weights([w_bi])

        w_cl, b_cl = pvad_model.classifier.get_weights()
        w_cl[0, 0] = 1.0
        w_cl[0, 1] = 0.0
        w_cl[0, 2] = -1.0
        w_cl[1, 0] = -1.0
        w_cl[1, 1] = 0.0
        w_cl[1, 2] = 1.0
        w_cl[2:26, 0] = 0.35
        w_cl[2:26, 1] = 0.0
        w_cl[2:26, 2] = -0.35
        pvad_model.classifier.set_weights([w_cl, b_cl])


def train_model(
    pvad_model: model_lib.PersonalVadModel,
    train_dataset: tf.data.Dataset,
    loss_fn: Union[loss_lib.WeightedPairwiseLoss, loss_lib.CrossEntropyLoss],
    epochs: int = 5,
    learning_rate: float = 5e-5,
    checkpoint_dir: Optional[str] = None,
    subspace: Optional[dict[str, Any]] = None,
    val_dataset: Optional[tf.data.Dataset] = None,
) -> list[float]:
  """Runs the training loop and returns a list of per-epoch mean losses.

  Args:
    pvad_model: `PersonalVadModel` to train.
    train_dataset: Batched `tf.data.Dataset` yielding `(inputs, labels, mask)`.
    loss_fn: `WeightedPairwiseLoss` or `CrossEntropyLoss` instance.
    epochs: Number of training epochs.
    learning_rate: Learning rate for Adam optimizer (`5e-5` in PVAD 1.0).
    checkpoint_dir: Optional directory to save model weights and config JSON.
    subspace: Optional speaker subspace dict for warm-starting conditioning.
    val_dataset: Optional validation `tf.data.Dataset` for best-epoch selection.

  Returns:
    List of measured mean training loss values for each completed epoch.
  """
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  epoch_losses: list[float] = []
  uses_prenet = pvad_model.speaker_prenet is not None
  for b_in, _, _ in train_dataset.take(1):
    _ = pvad_model(b_in, training=False)

  if subspace is not None:
    warm_start_speaker_conditioning(pvad_model, subspace)

  @tf.function
  def _train_step(
      batch_inputs: dict[str, tf.Tensor],
      batch_labels: tf.Tensor,
      batch_mask: tf.Tensor,
  ) -> tf.Tensor:
    with tf.GradientTape() as tape:
      if uses_prenet and pvad_model.speaker_prenet is not None:
        logits, cos_sim_3d, _ = pvad_model.call_with_prenet(
            batch_inputs, training=True
        )
        loss_val = loss_fn.compute_loss(
            logits=logits, labels=batch_labels, mask=batch_mask
        )
        assert cos_sim_3d is not None
        cos_sim = tf.squeeze(cos_sim_3d, axis=-1)
        target_cos = tf.where(
            tf.equal(batch_labels, 0),
            0.85,
            tf.where(tf.equal(batch_labels, 2), -0.35, 0.0),
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
      else:
        logits = pvad_model(batch_inputs, training=True)
        loss_val = loss_fn.compute_loss(
            logits=logits, labels=batch_labels, mask=batch_mask
        )
    grads = tape.gradient(loss_val, pvad_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, pvad_model.trainable_variables))
    return loss_val

  @tf.function
  def _val_step(
      batch_inputs: dict[str, tf.Tensor],
      batch_labels: tf.Tensor,
      batch_mask: tf.Tensor,
  ) -> tf.Tensor:
    logits = pvad_model(batch_inputs, training=False)
    return loss_fn.compute_loss(
        logits=logits, labels=batch_labels, mask=batch_mask
    )

  best_val_loss = float("inf")
  best_weights = None

  for _ in range(epochs):
    batch_losses = []
    for batch_inputs, batch_labels, batch_mask in train_dataset:
      loss_val = _train_step(batch_inputs, batch_labels, batch_mask)
      batch_losses.append(float(loss_val.numpy()))

    mean_epoch_loss = (
        sum(batch_losses) / float(len(batch_losses)) if batch_losses else 0.0
    )
    epoch_losses.append(mean_epoch_loss)

    if val_dataset is not None:
      val_losses = [
          float(_val_step(b_in, b_lbl, b_msk).numpy())
          for b_in, b_lbl, b_msk in val_dataset
      ]
      mean_val = (
          sum(val_losses) / float(len(val_losses))
          if val_losses
          else mean_epoch_loss
      )
      if mean_val <= best_val_loss:
        best_val_loss = mean_val
        best_weights = [w.copy() for w in pvad_model.get_weights()]

  if best_weights is not None:
    pvad_model.set_weights(best_weights)

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
      "--dev_npz",
      type=str,
      default=None,
      help="Optional path to validation dataset .npz file for best checkpoint.",
  )
  parser.add_argument(
      "--subspace_npz",
      type=str,
      default=None,
      help="Optional path to speaker_subspace.npz for warm-starting weights.",
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
  val_ds = None
  if args.dev_npz is not None and os.path.exists(args.dev_npz):
    dev_utts = prepare_dataset.load_utterances_from_npz(args.dev_npz)
    val_ds = dataset.create_tf_dataset(
        dev_utts,
        batch_size=args.batch_size,
        shuffle=False,
        enrollment_less_prob=0.0,
        seed=args.seed,
    )
  subspace = None
  if args.subspace_npz is not None and os.path.exists(args.subspace_npz):
    import numpy as np  # noqa: E402

    with np.load(args.subspace_npz, allow_pickle=False) as data:
      subspace = {k: data[k] for k in data.files}

  train_model(
      pvad_model=pvad_model,
      train_dataset=train_ds,
      loss_fn=loss_fn,
      epochs=args.epochs,
      learning_rate=args.learning_rate,
      checkpoint_dir=args.checkpoint_dir,
      subspace=subspace,
      val_dataset=val_ds,
  )


if __name__ == "__main__":
  main()
