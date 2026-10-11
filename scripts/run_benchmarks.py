"""End-to-end Multilingual LibriSpeech training, TFLite export, and evaluation.

Trains and evaluates all Personal VAD 1.0 (arXiv:1908.04284) and Personal VAD
2.0 (arXiv:2204.03793) architectures on the 8-language Multilingual LibriSpeech
(MLS) dataset (`de`, `en`, `es`, `fr`, `it`, `nl`, `pl`, `pt`), exports stateful
FP32 and 8-bit dynamic-range quantized `.tflite` flatbuffers, and writes
verifiable evaluation reports (`benchmark_summary.json` and per-model
`metrics.json`, `report.html`, `pr_curves.png`, `roc_curves.png`).
"""

import argparse
from collections.abc import Sequence
import json
import os
import shutil
import sys
import time
from typing import Any, Optional
import numpy as np
from sklearn import metrics as sk_metrics
import tensorflow as tf

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import configs  # noqa: E402
from personal_vad import dataset  # noqa: E402
from personal_vad import eval_lib  # noqa: E402
from personal_vad import model as model_lib  # noqa: E402
from personal_vad import tflite_export  # noqa: E402
from scripts import evaluate  # noqa: E402
from scripts import prepare_dataset  # noqa: E402
from scripts import train  # noqa: E402


def _train_utterance_model(
    utterances: Sequence[dataset.UtteranceData],
    backbone: str,
    conditioning_mode: str,
    loss_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float = 2.5e-3,
    subspace: Optional[dict[str, Any]] = None,
    val_utterances: Optional[Sequence[dataset.UtteranceData]] = None,
    seed: int = 42,
) -> tuple[model_lib.PersonalVadModel, list[float]]:
  """Builds and trains a `PersonalVadModel` on `utterances`."""
  tf.keras.utils.set_random_seed(seed)
  feature_dim = utterances[0].features.shape[1]
  emb_dim = utterances[0].speaker_embedding.shape[0]
  max_frames = 360 if feature_dim == 40 else 135

  model_cfg = train.build_model_config(
      backbone=backbone,
      conditioning_mode=conditioning_mode,
      feature_dim=feature_dim,
      speaker_embedding_dim=emb_dim,
  )
  pvad_model = model_lib.PersonalVadModel(config=model_cfg)
  loss_cfg = configs.LossConfig(
      loss_type=configs.LossType(loss_type),
      num_classes=model_cfg.num_classes,
  )
  loss_fn = train.build_loss(loss_cfg)
  train_ds = dataset.create_tf_dataset(
      utterances,
      batch_size=batch_size,
      max_frames=max_frames,
      shuffle=True,
      enrollment_less_prob=0.0,
      seed=seed,
  )
  val_ds = None
  if val_utterances:
    val_ds = dataset.create_tf_dataset(
        val_utterances,
        batch_size=batch_size,
        max_frames=max_frames,
        shuffle=False,
        enrollment_less_prob=0.0,
        seed=seed,
    )
  losses = train.train_model(
      pvad_model=pvad_model,
      train_dataset=train_ds,
      loss_fn=loss_fn,
      epochs=epochs,
      learning_rate=learning_rate,
      subspace=subspace,
      val_dataset=val_ds,
  )
  return pvad_model, losses


def evaluate_speaker_verification_auc(
    test_records: Sequence[dict[str, Any]],
    subspace: dict[str, np.ndarray],
) -> dict[str, float]:
  """Computes open-set speaker verification ROC-AUC on unseen speakers."""
  utts = prepare_dataset.build_utterances_from_mls_records(
      test_records, subspace=subspace, frontend_version="v1"
  )
  centroids: dict[str, np.ndarray] = {}
  for u in utts:
    centroids[u.speaker_id] = u.speaker_embedding

  pos_cos: list[float] = []
  neg_cos: list[float] = []
  for u in utts:
    speech_idx = np.where(u.labels == 0)[0]
    if len(speech_idx) < 5:
      continue
    utt_emb = dataset.l2_normalize_numpy(
        np.mean(u.frame_speaker_embeddings[speech_idx], axis=0)
    )
    for spk, cent in centroids.items():
      c = float(np.dot(utt_emb, cent))
      if spk == u.speaker_id:
        pos_cos.append(c)
      else:
        neg_cos.append(c)

  y_true = np.array([1] * len(pos_cos) + [0] * len(neg_cos))
  y_score = np.array(pos_cos + neg_cos)
  auc = float(sk_metrics.roc_auc_score(y_true, y_score))
  return {
      "roc_auc": auc,
      "mean_positive_cosine": float(np.mean(pos_cos)),
      "mean_negative_cosine": float(np.mean(neg_cos)),
      "num_test_speakers": len(centroids),
      "num_test_utterances": len(utts),
  }


def make_target_only_non_concat_utterances(
    single_utterances: Sequence[dataset.UtteranceData],
) -> list[dataset.UtteranceData]:
  """Creates single-speaker target-only utterances for Standard VAD eval."""
  rng = np.random.default_rng(101)
  return dataset.concat_utterance_group(
      single_utterances,
      min_utterances=1,
      max_utterances=1,
      enrollment_less_prob=0.0,
      rng=rng,
  )


def make_non_concat_utterances(
    single_utterances: Sequence[dataset.UtteranceData],
    enrollment_less: bool = False,
) -> list[dataset.UtteranceData]:
  """Creates single-speaker eval utterances with 50% neg speaker trials."""
  rng = np.random.default_rng(101)
  base_utts = dataset.concat_utterance_group(
      single_utterances,
      min_utterances=1,
      max_utterances=1,
      enrollment_less_prob=1.0 if enrollment_less else 0.0,
      rng=rng,
  )
  if enrollment_less:
    return base_utts

  outputs: list[dataset.UtteranceData] = []
  n_utts = len(base_utts)
  for i, utt in enumerate(base_utts):
    if i % 2 == 1:
      other = base_utts[(i + 15) % n_utts]
      if other.speaker_id != utt.speaker_id:
        target_emb = dataset.l2_normalize_numpy(other.speaker_embedding)
        relabeled = np.where(
            utt.labels == int(configs.FrameLabel.SPEECH),
            int(configs.FrameLabel.SPEECH_FROM_NON_TARGET_SPEAKER),
            utt.labels,
        )
        cos_scores = (
            dataset.compute_frame_cosine_similarity(
                utt.frame_speaker_embeddings, target_emb
            )
            if utt.frame_speaker_embeddings is not None
            else utt.cosine_scores
        )
        outputs.append(
            dataset.UtteranceData(
                utt_id=f"{utt.utt_id}_neg",
                speaker_id=other.speaker_id,
                features=utt.features,
                labels=relabeled,
                speaker_embedding=target_emb,
                frame_speaker_embeddings=utt.frame_speaker_embeddings,
                cosine_scores=cos_scores,
                language=utt.language,
            )
        )
        continue
    outputs.append(utt)
  return outputs


def make_concat_utterances(
    single_utterances: Sequence[dataset.UtteranceData],
    num_passes: int = 2,
    enrollment_less: bool = False,
    convert_for_vad_baseline: bool = False,
    enrollment_less_prob: Optional[float] = None,
    multi_target: bool = False,
    seed: int = 42,
) -> list[dataset.UtteranceData]:
  """Creates multi-speaker concatenated utterances across `num_passes`.

  When `multi_target=True`, each concatenated multi-speaker group is emitted
  with counterfactual target-speaker trials (each speaker in the mixture takes
  a turn as the enrolled target speaker), ensuring that every acoustic speech
  segment is observed as both `tss` and `ntss` depending strictly on
  `speaker_embedding`.
  """
  rng = np.random.default_rng(seed)
  p0 = (
      enrollment_less_prob
      if enrollment_less_prob is not None
      else (1.0 if enrollment_less else 0.0)
  )
  all_concat: list[dataset.UtteranceData] = []
  for _ in range(num_passes):
    pool = list(single_utterances)
    rng.shuffle(pool)
    if not multi_target:
      all_concat.extend(
          dataset.concat_utterance_group(
              pool,
              min_utterances=1,
              max_utterances=3,
              convert_for_vad_baseline=convert_for_vad_baseline,
              enrollment_less_prob=p0,
              rng=rng,
          )
      )
      continue

    remaining = list(pool)
    group_idx = 0
    while remaining:
      k = int(rng.integers(2, 4))
      k = min(k, len(remaining))
      group: list[dataset.UtteranceData] = []
      used_spks: set[str] = set()
      for _ in range(k):
        cand_idx = next(
            (
                idx
                for idx, u in enumerate(remaining)
                if u.speaker_id not in used_spks
            ),
            0,
        )
        chosen = remaining.pop(cand_idx)
        group.append(chosen)
        used_spks.add(chosen.speaker_id)

      features = np.concatenate([u.features for u in group], axis=0)
      has_fe = all(u.frame_speaker_embeddings is not None for u in group)
      frame_embs = (
          np.concatenate([u.frame_speaker_embeddings for u in group], axis=0)
          if has_fe
          else None
      )

      for t_idx, target_utt in enumerate(group[:2]):
        is_no_enroll = bool(p0 > 0.0 and rng.random() < p0)
        if is_no_enroll:
          emb_dim = target_utt.speaker_embedding.shape[0]
          t_emb = np.zeros((emb_dim,), dtype=np.float32)
          t_spk = "enrollment_less"
        else:
          t_emb = dataset.l2_normalize_numpy(target_utt.speaker_embedding)
          t_spk = target_utt.speaker_id

        label_parts = []
        for u in group:
          u_lbl = u.labels.copy()
          if is_no_enroll or convert_for_vad_baseline:
            u_lbl = np.where(
                u_lbl == int(configs.FrameLabel.SILENCE),
                int(configs.FrameLabel.SILENCE),
                int(configs.FrameLabel.SPEECH),
            )
          elif u.speaker_id == t_spk:
            u_lbl = np.where(
                u_lbl == int(configs.FrameLabel.SILENCE),
                int(configs.FrameLabel.SILENCE),
                int(configs.FrameLabel.SPEECH),
            )
          else:
            u_lbl = np.where(
                u_lbl == int(configs.FrameLabel.SILENCE),
                int(configs.FrameLabel.SILENCE),
                int(configs.FrameLabel.SPEECH_FROM_NON_TARGET_SPEAKER),
            )
          label_parts.append(u_lbl.astype(np.int32))
        labels = np.concatenate(label_parts, axis=0)
        cos_scores = (
            dataset.compute_frame_cosine_similarity(frame_embs, t_emb)
            if frame_embs is not None
            else np.zeros((features.shape[0], 1), dtype=np.float32)
        )
        all_concat.append(
            dataset.UtteranceData(
                utt_id=f"concat_{group_idx:04d}_t{t_idx}",
                speaker_id=t_spk,
                features=features,
                labels=labels,
                speaker_embedding=t_emb,
                frame_speaker_embeddings=frame_embs,
                cosine_scores=cos_scores,
                language=target_utt.language,
            )
        )
      group_idx += 1
  return all_concat


def export_and_save_model_artifacts(
    pvad_model: Any,
    model_dir: str,
    sequence_length: int = 32,
) -> dict[str, Any]:
  """Saves Keras checkpoint, Safetensors, FP32 TFLite, and Quantized TFLite."""
  os.makedirs(model_dir, exist_ok=True)
  weights_path = train.save_checkpoint(pvad_model, model_dir)
  config_path = os.path.join(model_dir, "model_config.json")
  safetensors_path = os.path.join(model_dir, "model.safetensors")
  fp32_tflite_path = os.path.join(model_dir, "model_fp32.tflite")
  quant_tflite_path = os.path.join(model_dir, "model_quantized.tflite")

  fp32_bytes = tflite_export.export_to_tflite(
      pvad_model,
      output_path=fp32_tflite_path,
      quantize=False,
      sequence_length=sequence_length,
  )
  quant_bytes = tflite_export.export_to_tflite(
      pvad_model,
      output_path=quant_tflite_path,
      quantize=True,
      sequence_length=sequence_length,
  )
  fp32_size_kb = round(len(fp32_bytes) / 1024.0, 2)
  quant_size_kb = round(len(quant_bytes) / 1024.0, 2)
  safetensors_size_kb = (
      round(os.path.getsize(safetensors_path) / 1024.0, 2)
      if os.path.exists(safetensors_path)
      else 0.0
  )

  num_params = int(
      sum(np.prod(v.shape) for v in pvad_model.trainable_variables)
  )
  return {
      "weights_path": weights_path,
      "safetensors_path": safetensors_path,
      "config_path": config_path,
      "fp32_tflite_path": fp32_tflite_path,
      "quant_tflite_path": quant_tflite_path,
      "fp32_size_kb": fp32_size_kb,
      "quant_size_kb": quant_size_kb,
      "safetensors_size_kb": safetensors_size_kb,
      "num_parameters": num_params,
  }


def _cluster_bootstrap_ap_samples(
    s_1d: np.ndarray,
    y_1d: np.ndarray,
    c_1d: np.ndarray,
    weights: np.ndarray,
    num_bins: int = 2048,
) -> np.ndarray:
  """Computes cluster-bootstrap AP samples via single-sort rank-binning."""
  m = len(s_1d)
  if m == 0:
    return np.zeros((weights.shape[0],), dtype=np.float32)
  b = min(num_bins, m)
  order = np.argsort(-s_1d, kind="mergesort")
  y_sorted = y_1d[order].astype(np.float32)
  c_sorted = c_1d[order].astype(np.int64)
  bin_id = (np.arange(m, dtype=np.int64) * b) // m
  n_clusters = weights.shape[1]
  flat_idx = c_sorted * b + bin_id
  pos_mat = np.bincount(
      flat_idx, weights=y_sorted, minlength=n_clusters * b
  ).reshape(n_clusters, b).astype(np.float32)
  tot_mat = np.bincount(
      flat_idx, minlength=n_clusters * b
  ).reshape(n_clusters, b).astype(np.float32)
  boot_pos = weights @ pos_mat
  boot_tot = weights @ tot_mat
  cum_pos = np.cumsum(boot_pos, axis=1)
  cum_tot = np.cumsum(boot_tot, axis=1)
  prec = cum_pos / np.maximum(cum_tot, 1.0)
  return np.sum(prec * boot_pos, axis=1) / np.maximum(cum_pos[:, -1], 1.0)


def compute_bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    eval_mode: str,
    sc_baseline: bool = False,
    speech_score_mode: str = "tss",
    utterance_lengths: Optional[Sequence[int]] = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
  """Computes 95% utterance-cluster bootstrap CIs on AP and mAP metrics."""
  evaluator = eval_lib.PersonalVadEvaluator(
      eval_mode=eval_mode, speech_score_mode=speech_score_mode
  )
  prep_fn = evaluator._prepare_scores_and_labels  # noqa: SLF001
  proc_scores, _, binarized_labels = prep_fn(
      scores, labels, sc_baseline=sc_baseline
  )
  n_frames = len(binarized_labels)
  if utterance_lengths and sum(utterance_lengths) == n_frames:
    n_clusters = len(utterance_lengths)
    frame_cluster_id = np.repeat(
        np.arange(n_clusters, dtype=np.int64), utterance_lengths
    )
  else:
    block_size = 150
    n_clusters = max((n_frames + block_size - 1) // block_size, 1)
    frame_cluster_id = np.arange(n_frames, dtype=np.int64) // block_size

  rng = np.random.default_rng(seed)
  weights = np.zeros((n_resamples, n_clusters), dtype=np.float32)
  for r in range(n_resamples):
    c_idx = rng.integers(0, n_clusters, size=n_clusters)
    weights[r] = np.bincount(c_idx, minlength=n_clusters)

  num_classes = binarized_labels.shape[1]
  flat_scores = proc_scores.reshape(-1)
  flat_labels = binarized_labels.reshape(-1)
  flat_clusters = np.repeat(frame_cluster_id, num_classes)
  map_samples = _cluster_bootstrap_ap_samples(
      flat_scores, flat_labels, flat_clusters, weights
  )

  result: dict[str, Any] = {
      "resample_unit": "utterance_mixture",
      "num_clusters": n_clusters,
      "n_resamples": n_resamples,
      "mAP_micro_95_ci": [
          round(float(np.percentile(map_samples, 2.5)), 4),
          round(float(np.percentile(map_samples, 97.5)), 4),
      ],
  }
  if num_classes == 3:
    tss_samples = _cluster_bootstrap_ap_samples(
        proc_scores[:, 0], binarized_labels[:, 0], frame_cluster_id, weights
    )
    ns_samples = _cluster_bootstrap_ap_samples(
        proc_scores[:, 1], binarized_labels[:, 1], frame_cluster_id, weights
    )
    ntss_samples = _cluster_bootstrap_ap_samples(
        proc_scores[:, 2], binarized_labels[:, 2], frame_cluster_id, weights
    )
    result["tss_ap_95_ci"] = [
        round(float(np.percentile(tss_samples, 2.5)), 4),
        round(float(np.percentile(tss_samples, 97.5)), 4),
    ]
    result["ns_ap_95_ci"] = [
        round(float(np.percentile(ns_samples, 2.5)), 4),
        round(float(np.percentile(ns_samples, 97.5)), 4),
    ]
    result["ntss_ap_95_ci"] = [
        round(float(np.percentile(ntss_samples, 2.5)), 4),
        round(float(np.percentile(ntss_samples, 97.5)), 4),
    ]
  return result


def run_eval(
    pvad_model: Any,
    utterances: Sequence[dataset.UtteranceData],
    eval_mode: str,
    output_dir: str,
    sc_baseline: bool = False,
    speech_score_mode: str = "tss",
    tflite_path: Optional[str] = None,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
  """Evaluates a Keras or TFLite model and writes metrics, plots, and HTML."""
  os.makedirs(output_dir, exist_ok=True)
  utt_lengths = [u.features.shape[0] for u in utterances]
  if tflite_path is not None:
    runner = tflite_export.TFLitePersonalVadRunner(tflite_path)
    scores, labels = evaluate.collect_tflite_predictions(
        runner, utterances, include_cosine_column=sc_baseline
    )
  else:
    scores, labels = evaluate.collect_model_predictions(
        pvad_model, utterances, include_cosine_column=sc_baseline
    )

  evaluator = eval_lib.PersonalVadEvaluator(
      eval_mode=eval_mode, speech_score_mode=speech_score_mode
  )
  res = evaluator.evaluate(
      scores=scores,
      labels=labels,
      sc_baseline=sc_baseline,
      output_dir=output_dir,
  )
  res_dict = res.to_dict()
  res_dict["bootstrap_ci"] = compute_bootstrap_ci(
      scores=scores,
      labels=labels,
      eval_mode=eval_mode,
      sc_baseline=sc_baseline,
      speech_score_mode=speech_score_mode,
      utterance_lengths=utt_lengths,
      n_resamples=n_bootstrap,
  )
  with open(
      os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8"
  ) as f:
    json.dump(res_dict, f, indent=2)
  return res_dict


def evaluate_per_language(
    pvad_model: Any,
    test_single_utterances: Sequence[dataset.UtteranceData],
    sc_baseline: bool = False,
    tflite_path: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
  """Computes 3-class Personal VAD metrics broken down across 8 languages."""
  by_lang: dict[str, list[dataset.UtteranceData]] = {}
  for u in test_single_utterances:
    lang = u.language or u.speaker_id.split("_")[0]
    by_lang.setdefault(lang, []).append(u)

  runner = (
      tflite_export.TFLitePersonalVadRunner(tflite_path)
      if tflite_path is not None
      else None
  )
  evaluator = eval_lib.PersonalVadEvaluator(eval_mode="personal_vad")
  lang_results: dict[str, dict[str, Any]] = {}
  for lang in sorted(by_lang.keys()):
    lang_utts = make_concat_utterances(by_lang[lang], num_passes=2, seed=99)
    if runner is not None:
      scores, labels = evaluate.collect_tflite_predictions(
          runner, lang_utts, include_cosine_column=sc_baseline
      )
    else:
      scores, labels = evaluate.collect_model_predictions(
          pvad_model, lang_utts, include_cosine_column=sc_baseline
      )
    res = evaluator.evaluate(
        scores=scores, labels=labels, sc_baseline=sc_baseline, output_dir=None
    )
    lang_results[lang] = {
        "num_speakers": len({u.speaker_id for u in by_lang[lang]}),
        "num_eval_mixtures": len(lang_utts),
        "num_frames": int(len(labels)),
        "average_precisions": {
            k: round(float(v), 4) for k, v in res.average_precisions.items()
        },
        "mean_average_precision_micro": round(
            float(res.mean_average_precision_micro), 4
        ),
    }
  return lang_results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Run Personal VAD 1.0 and 2.0 benchmarks on MLS."
  )
  parser.add_argument(
      "--mls_dir",
      type=str,
      default="/usr/local/google/home/quanw/Data/multilingual_librispeech",
      help="Directory containing train/dev/test_manifest.json.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="/usr/local/google/home/quanw/Data/pvad_MLS_Watch_models",
      help="Directory to store trained checkpoints, TFLite files, and reports.",
  )
  parser.add_argument(
      "--epochs_v1",
      type=int,
      default=8,
      help="Number of training epochs for Personal VAD 1.0 models.",
  )
  parser.add_argument(
      "--epochs_v2",
      type=int,
      default=8,
      help="Number of training epochs for Personal VAD 2.0 models.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=128,
      help="Training mini-batch size.",
  )
  return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
  args = parse_args(argv)
  os.makedirs(args.output_dir, exist_ok=True)
  start_time = time.time()

  train_manifest = os.path.join(args.mls_dir, "train_manifest.json")
  dev_manifest = os.path.join(args.mls_dir, "dev_manifest.json")
  test_manifest = os.path.join(args.mls_dir, "test_manifest.json")

  print("1. Loading Multilingual LibriSpeech audio (clean + MTR)...")
  train_records_clean = prepare_dataset.load_mls_manifest_records(
      train_manifest, apply_mtr=False
  )
  train_records_mtr = prepare_dataset.load_mls_manifest_records(
      train_manifest, apply_mtr=True, mtr_seed=2025
  )
  dev_records = (
      prepare_dataset.load_mls_manifest_records(dev_manifest, apply_mtr=False)
      if os.path.exists(dev_manifest)
      else []
  )
  test_records = prepare_dataset.load_mls_manifest_records(
      test_manifest, apply_mtr=False
  )
  test_records_mtr = prepare_dataset.load_mls_manifest_records(
      test_manifest, apply_mtr=True, mtr_seed=9001
  )

  print("2. Fitting open-set LDA+WCCN speaker subspace on training speakers...")
  subspace = prepare_dataset.fit_lda_wccn_subspace(train_records_clean)
  subspace_path = os.path.join(args.output_dir, "speaker_subspace.npz")
  np.savez(subspace_path, **subspace)
  sv_metrics = evaluate_speaker_verification_auc(test_records, subspace)
  sv_dev_metrics = (
      evaluate_speaker_verification_auc(dev_records, subspace)
      if dev_records
      else {}
  )
  print(
      f"   Unseen test speaker verification ROC-AUC: "
      f"{sv_metrics['roc_auc']:.4f} "
      f"(pos_cos={sv_metrics['mean_positive_cosine']:.4f}, "
      f"neg_cos={sv_metrics['mean_negative_cosine']:.4f})"
  )

  print("3. Preparing Personal VAD 1.0 (40-dim) and 2.0 (512-dim) splits...")
  train_single_v1_clean = prepare_dataset.build_utterances_from_mls_records(
      train_records_clean, subspace, frontend_version="v1"
  )
  train_single_v1_mtr = prepare_dataset.build_utterances_from_mls_records(
      train_records_mtr, subspace, frontend_version="v1"
  )
  dev_single_v1 = (
      prepare_dataset.build_utterances_from_mls_records(
          dev_records, subspace, frontend_version="v1"
      )
      if dev_records
      else []
  )
  test_single_v1 = prepare_dataset.build_utterances_from_mls_records(
      test_records, subspace, frontend_version="v1"
  )
  test_single_v1_mtr = prepare_dataset.build_utterances_from_mls_records(
      test_records_mtr, subspace, frontend_version="v1"
  )

  train_single_v2_clean = prepare_dataset.build_utterances_from_mls_records(
      train_records_clean, subspace, frontend_version="v2"
  )
  train_single_v2_mtr = prepare_dataset.build_utterances_from_mls_records(
      train_records_mtr, subspace, frontend_version="v2"
  )
  dev_single_v2 = (
      prepare_dataset.build_utterances_from_mls_records(
          dev_records, subspace, frontend_version="v2"
      )
      if dev_records
      else []
  )
  test_single_v2 = prepare_dataset.build_utterances_from_mls_records(
      test_records, subspace, frontend_version="v2"
  )
  test_single_v2_mtr = prepare_dataset.build_utterances_from_mls_records(
      test_records_mtr, subspace, frontend_version="v2"
  )

  # Validation datasets on unseen dev speakers
  dev_v1_concat = (
      make_concat_utterances(dev_single_v1, num_passes=2, seed=77)
      if dev_single_v1
      else None
  )
  dev_v1_std_vad = (
      make_concat_utterances(
          dev_single_v1, num_passes=2, convert_for_vad_baseline=True, seed=77
      )
      if dev_single_v1
      else None
  )
  dev_v2_concat = (
      make_concat_utterances(dev_single_v2, num_passes=2, seed=77)
      if dev_single_v2
      else None
  )
  dev_v2_std_vad = (
      make_concat_utterances(
          dev_single_v2, num_passes=2, convert_for_vad_baseline=True, seed=77
      )
      if dev_single_v2
      else None
  )

  # Evaluation datasets on 40 unseen test speakers
  test_v1_concat = make_concat_utterances(test_single_v1, num_passes=2, seed=99)
  test_v1_concat_mtr = make_concat_utterances(
      test_single_v1_mtr, num_passes=2, seed=99
  )
  test_v1_non_concat = make_non_concat_utterances(test_single_v1)
  test_v1_std_non_concat = make_target_only_non_concat_utterances(
      test_single_v1
  )

  test_v2_concat = make_concat_utterances(test_single_v2, num_passes=2, seed=99)
  test_v2_concat_mtr = make_concat_utterances(
      test_single_v2_mtr, num_passes=2, seed=99
  )
  test_v2_non_concat = make_non_concat_utterances(test_single_v2)
  test_v2_std_non_concat = make_target_only_non_concat_utterances(
      test_single_v2
  )
  test_v2_no_enroll_concat = make_concat_utterances(
      test_single_v2, num_passes=2, enrollment_less=True, seed=99
  )
  test_v2_no_enroll_non_concat = make_non_concat_utterances(
      test_single_v2, enrollment_less=True
  )

  # Training datasets (Clean + Multistyle Training MTR + Counterfactual Trials)
  train_v1_pool = list(train_single_v1_clean) + list(train_single_v1_mtr)
  train_v1_std_vad = make_concat_utterances(
      train_v1_pool, num_passes=2, convert_for_vad_baseline=True, seed=42
  )
  train_v1_concat = make_concat_utterances(
      train_v1_pool, num_passes=3, multi_target=True, seed=42
  )

  train_v2_pool = list(train_single_v2_clean) + list(train_single_v2_mtr)
  train_v2_std_vad = make_concat_utterances(
      train_v2_pool, num_passes=2, convert_for_vad_baseline=True, seed=42
  )
  train_v2_concat = make_concat_utterances(
      train_v2_pool, num_passes=3, multi_target=True, seed=42
  )
  train_v2_unified = make_concat_utterances(
      train_v2_pool,
      num_passes=3,
      enrollment_less_prob=0.20,
      multi_target=True,
      seed=42,
  )

  num_train_speakers = len({r["speaker_id"] for r in train_records_clean})
  num_dev_speakers = len({r["speaker_id"] for r in dev_records})
  num_test_speakers = len({r["speaker_id"] for r in test_records})

  summary: dict[str, Any] = {
      "dataset": {
          "mls_dir": args.mls_dir,
          "languages": ["de", "en", "es", "fr", "it", "nl", "pl", "pt"],
          "num_train_speakers": num_train_speakers,
          "num_dev_speakers": num_dev_speakers,
          "num_test_speakers": num_test_speakers,
          "num_total_speakers": (
              num_train_speakers + num_dev_speakers + num_test_speakers
          ),
          "speaker_disjoint": True,
          "speaker_verification_dev": sv_dev_metrics,
          "speaker_verification": sv_metrics,
      },
      "training_hyperparameters": {
          "epochs_v1": args.epochs_v1,
          "epochs_v2": args.epochs_v2,
          "batch_size": args.batch_size,
          "mtr_augmentation": True,
      },
      "models": {},
  }

  # =========================================================================
  # Personal VAD 1.0 Experiments (arXiv:1908.04284)
  # =========================================================================
  v1_specs = [
      (
          "pvad1_sc_baseline",
          "sc",
          "cross_entropy",
          train_v1_std_vad,
          dev_v1_std_vad,
      ),
      ("pvad1_st_ce", "st", "cross_entropy", train_v1_concat, dev_v1_concat),
      ("pvad1_et_ce", "et", "cross_entropy", train_v1_concat, dev_v1_concat),
      ("pvad1_set_ce", "set", "cross_entropy", train_v1_concat, dev_v1_concat),
      (
          "pvad1_et_wpl",
          "et",
          "weighted_pairwise",
          train_v1_concat,
          dev_v1_concat,
      ),
  ]

  for model_id, cond_mode, loss_type, tr_utts, dv_utts in v1_specs:
    model_dir = os.path.join(args.output_dir, model_id)
    print(
        f"\n=== Training {model_id} (mode={cond_mode}, loss={loss_type}) ==="
    )
    pvad_model, losses = _train_utterance_model(
        utterances=tr_utts,
        backbone="lstm_v1",
        conditioning_mode=cond_mode,
        loss_type=loss_type,
        epochs=args.epochs_v1,
        batch_size=args.batch_size,
        learning_rate=2.5e-3,
        subspace=subspace,
        val_utterances=dv_utts,
        seed=42,
    )
    artifacts = export_and_save_model_artifacts(
        pvad_model, model_dir, sequence_length=32
    )
    shutil.copyfile(
        subspace_path, os.path.join(model_dir, "speaker_subspace.npz")
    )
    evals: dict[str, Any] = {}
    if cond_mode == "sc":
      evals["personal_vad_concat"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="personal_vad",
          sc_baseline=True,
          output_dir=os.path.join(model_dir, "eval_personal_vad_concat"),
      )
      evals["personal_vad_non_concat"] = run_eval(
          pvad_model,
          test_v1_non_concat,
          eval_mode="personal_vad",
          sc_baseline=True,
          output_dir=os.path.join(model_dir, "eval_personal_vad_non_concat"),
      )
      evals["personal_vad_concat_mtr"] = run_eval(
          pvad_model,
          test_v1_concat_mtr,
          eval_mode="personal_vad",
          sc_baseline=True,
          output_dir=os.path.join(model_dir, "eval_personal_vad_concat_mtr"),
      )
      evals["personal_vad_concat_tflite_fp32"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="personal_vad",
          sc_baseline=True,
          tflite_path=artifacts["fp32_tflite_path"],
          output_dir=os.path.join(model_dir, "eval_personal_vad_tflite_fp32"),
          n_bootstrap=200,
      )
      evals["personal_vad_concat_tflite_quant"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="personal_vad",
          sc_baseline=True,
          tflite_path=artifacts["quant_tflite_path"],
          output_dir=os.path.join(model_dir, "eval_personal_vad_tflite_quant"),
          n_bootstrap=200,
      )
      evals["std_vad_non_concat"] = run_eval(
          pvad_model,
          test_v1_std_non_concat,
          eval_mode="std_vad_std_eval",
          output_dir=os.path.join(model_dir, "eval_std_vad_non_concat"),
      )
      evals["std_vad_concat"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="std_vad_std_eval",
          output_dir=os.path.join(model_dir, "eval_std_vad_concat"),
      )
      evals["per_language_concat"] = evaluate_per_language(
          pvad_model, test_single_v1, sc_baseline=True
      )
    else:
      evals["personal_vad_concat"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="personal_vad",
          output_dir=os.path.join(model_dir, "eval_personal_vad_concat"),
      )
      evals["personal_vad_non_concat"] = run_eval(
          pvad_model,
          test_v1_non_concat,
          eval_mode="personal_vad",
          output_dir=os.path.join(model_dir, "eval_personal_vad_non_concat"),
      )
      evals["personal_vad_concat_mtr"] = run_eval(
          pvad_model,
          test_v1_concat_mtr,
          eval_mode="personal_vad",
          output_dir=os.path.join(model_dir, "eval_personal_vad_concat_mtr"),
      )
      evals["personal_vad_concat_tflite_fp32"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="personal_vad",
          tflite_path=artifacts["fp32_tflite_path"],
          output_dir=os.path.join(model_dir, "eval_personal_vad_tflite_fp32"),
          n_bootstrap=200,
      )
      evals["personal_vad_concat_tflite_quant"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="personal_vad",
          tflite_path=artifacts["quant_tflite_path"],
          output_dir=os.path.join(model_dir, "eval_personal_vad_tflite_quant"),
          n_bootstrap=200,
      )
      evals["std_vad_non_concat"] = run_eval(
          pvad_model,
          test_v1_std_non_concat,
          eval_mode="personal_vad_std_eval",
          speech_score_mode="tss",
          output_dir=os.path.join(model_dir, "eval_std_vad_non_concat"),
      )
      evals["std_vad_concat"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="personal_vad_std_eval",
          speech_score_mode="tss",
          output_dir=os.path.join(model_dir, "eval_std_vad_concat"),
      )
      evals["std_vad_concat_tss_plus_ntss"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="personal_vad_std_eval",
          speech_score_mode="tss_plus_ntss",
          output_dir=os.path.join(
              model_dir, "eval_std_vad_concat_tss_plus_ntss"
          ),
      )
      evals["per_language_concat"] = evaluate_per_language(
          pvad_model, test_single_v1, sc_baseline=False
      )

    pv_res = evals["personal_vad_concat"]
    print(
        f"   [{model_id}] Concat AP -> "
        f"tss={pv_res['average_precisions']['Speech (tss)']:.4f}, "
        f"ns={pv_res['average_precisions']['Silence (ns)']:.4f}, "
        f"ntss={pv_res['average_precisions']['Non-Target Speech (ntss)']:.4f}, "
        f"mAP={pv_res['mean_average_precision_micro']:.4f}"
    )
    summary["models"][model_id] = {
        "paper": "Personal VAD 1.0 (arXiv:1908.04284)",
        "backbone": "lstm_v1",
        "conditioning_mode": cond_mode,
        "loss_type": loss_type,
        "final_train_loss": float(losses[-1]),
        "artifacts": artifacts,
        "evaluations": evals,
    }

  # =========================================================================
  # Personal VAD 2.0 Experiments (arXiv:2204.03793)
  # =========================================================================
  v2_specs = [
      (
          "pvad2_b1_lstm_std_vad",
          "lstm_v2",
          "sc",
          train_v2_std_vad,
          dev_v2_std_vad,
      ),
      (
          "pvad2_b2_conformer_std_vad",
          "conformer",
          "sc",
          train_v2_std_vad,
          dev_v2_std_vad,
      ),
      (
          "pvad2_b0_lstm_concat",
          "lstm_v2",
          "concat",
          train_v2_concat,
          dev_v2_concat,
      ),
      (
          "pvad2_e0_conformer_concat",
          "conformer",
          "concat",
          train_v2_concat,
          dev_v2_concat,
      ),
      (
          "pvad2_e1_conformer_film_dvector",
          "conformer",
          "film_dvector",
          train_v2_concat,
          dev_v2_concat,
      ),
      (
          "pvad2_e2_conformer_film_cos",
          "conformer",
          "film_cos",
          train_v2_concat,
          dev_v2_concat,
      ),
      (
          "pvad2_e3_conformer_film_dvector_cos",
          "conformer",
          "film_dvector_cos",
          train_v2_concat,
          dev_v2_concat,
      ),
      (
          "pvad2_e5_conformer_unified",
          "conformer",
          "film_dvector_cos",
          train_v2_unified,
          dev_v2_concat,
      ),
  ]

  for model_id, backbone, cond_mode, tr_utts, dv_utts in v2_specs:
    model_dir = os.path.join(args.output_dir, model_id)
    print(
        f"\n=== Training {model_id} "
        f"(backbone={backbone}, mode={cond_mode}) ==="
    )
    pvad_model, losses = _train_utterance_model(
        utterances=tr_utts,
        backbone=backbone,
        conditioning_mode=cond_mode,
        loss_type="cross_entropy",
        epochs=args.epochs_v2,
        batch_size=args.batch_size,
        learning_rate=2.5e-3,
        subspace=subspace,
        val_utterances=dv_utts,
        seed=42,
    )
    artifacts = export_and_save_model_artifacts(
        pvad_model, model_dir, sequence_length=32
    )
    shutil.copyfile(
        subspace_path, os.path.join(model_dir, "speaker_subspace.npz")
    )
    evals = {}
    if cond_mode == "sc":
      evals["std_vad_non_concat"] = run_eval(
          pvad_model,
          test_v2_std_non_concat,
          eval_mode="std_vad_std_eval",
          output_dir=os.path.join(model_dir, "eval_std_vad_non_concat"),
      )
      evals["std_vad_concat"] = run_eval(
          pvad_model,
          test_v2_concat,
          eval_mode="std_vad_std_eval",
          output_dir=os.path.join(model_dir, "eval_std_vad_concat"),
      )
      evals["std_vad_concat_tflite_fp32"] = run_eval(
          pvad_model,
          test_v2_concat,
          eval_mode="std_vad_std_eval",
          tflite_path=artifacts["fp32_tflite_path"],
          output_dir=os.path.join(model_dir, "eval_std_vad_concat_tflite_fp32"),
          n_bootstrap=200,
      )
      evals["std_vad_concat_tflite_quant"] = run_eval(
          pvad_model,
          test_v2_concat,
          eval_mode="std_vad_std_eval",
          tflite_path=artifacts["quant_tflite_path"],
          output_dir=os.path.join(
              model_dir, "eval_std_vad_concat_tflite_quant"
          ),
          n_bootstrap=200,
      )
      std_res = evals["std_vad_concat"]
      print(
          f"   [{model_id}] Std VAD Concat AP -> "
          f"speech={std_res['average_precisions']['Speech']:.4f}, "
          f"ns={std_res['average_precisions']['Silence']:.4f}, "
          f"mAP={std_res['mean_average_precision_micro']:.4f}"
      )
    else:
      evals["personal_vad_non_concat"] = run_eval(
          pvad_model,
          test_v2_non_concat,
          eval_mode="personal_vad",
          output_dir=os.path.join(model_dir, "eval_personal_vad_non_concat"),
      )
      evals["personal_vad_concat"] = run_eval(
          pvad_model,
          test_v2_concat,
          eval_mode="personal_vad",
          output_dir=os.path.join(model_dir, "eval_personal_vad_concat"),
      )
      evals["personal_vad_concat_mtr"] = run_eval(
          pvad_model,
          test_v2_concat_mtr,
          eval_mode="personal_vad",
          output_dir=os.path.join(model_dir, "eval_personal_vad_concat_mtr"),
      )
      evals["personal_vad_non_concat_tflite_fp32"] = run_eval(
          pvad_model,
          test_v2_non_concat,
          eval_mode="personal_vad",
          tflite_path=artifacts["fp32_tflite_path"],
          output_dir=os.path.join(
              model_dir, "eval_personal_vad_non_concat_tflite_fp32"
          ),
          n_bootstrap=200,
      )
      evals["personal_vad_non_concat_tflite_quant"] = run_eval(
          pvad_model,
          test_v2_non_concat,
          eval_mode="personal_vad",
          tflite_path=artifacts["quant_tflite_path"],
          output_dir=os.path.join(
              model_dir, "eval_personal_vad_non_concat_tflite_quant"
          ),
          n_bootstrap=200,
      )
      evals["personal_vad_concat_tflite_fp32"] = run_eval(
          pvad_model,
          test_v2_concat,
          eval_mode="personal_vad",
          tflite_path=artifacts["fp32_tflite_path"],
          output_dir=os.path.join(
              model_dir, "eval_personal_vad_concat_tflite_fp32"
          ),
          n_bootstrap=200,
      )
      evals["personal_vad_concat_tflite_quant"] = run_eval(
          pvad_model,
          test_v2_concat,
          eval_mode="personal_vad",
          tflite_path=artifacts["quant_tflite_path"],
          output_dir=os.path.join(
              model_dir, "eval_personal_vad_concat_tflite_quant"
          ),
          n_bootstrap=200,
      )
      evals["per_language_concat"] = evaluate_per_language(
          pvad_model, test_single_v2, sc_baseline=False
      )
      if model_id in (
          "pvad2_e3_conformer_film_dvector_cos",
          "pvad2_e5_conformer_unified",
      ):
        evals["no_enrollment_non_concat"] = run_eval(
            pvad_model,
            test_v2_no_enroll_non_concat,
            eval_mode="personal_vad_std_eval",
            speech_score_mode="tss",
            output_dir=os.path.join(model_dir, "eval_no_enrollment_non_concat"),
        )
        evals["no_enrollment_concat"] = run_eval(
            pvad_model,
            test_v2_no_enroll_concat,
            eval_mode="personal_vad_std_eval",
            speech_score_mode="tss",
            output_dir=os.path.join(model_dir, "eval_no_enrollment_concat"),
        )
        evals["no_enrollment_concat_tflite_quant"] = run_eval(
            pvad_model,
            test_v2_no_enroll_concat,
            eval_mode="personal_vad_std_eval",
            speech_score_mode="tss",
            tflite_path=artifacts["quant_tflite_path"],
            output_dir=os.path.join(
                model_dir, "eval_no_enrollment_concat_tflite_quant"
            ),
            n_bootstrap=200,
        )
      pv_res = evals["personal_vad_concat"]
      print(
          f"   [{model_id}] Concat AP -> "
          f"tss={pv_res['average_precisions']['Speech (tss)']:.4f}, "
          f"ns={pv_res['average_precisions']['Silence (ns)']:.4f}, "
          f"ntss={pv_res['average_precisions']['Non-Target Speech (ntss)']:.4f}"
          f", mAP={pv_res['mean_average_precision_micro']:.4f}"
      )

    summary["models"][model_id] = {
        "paper": "Personal VAD 2.0 (arXiv:2204.03793)",
        "backbone": backbone,
        "conditioning_mode": cond_mode,
        "loss_type": "cross_entropy",
        "final_train_loss": float(losses[-1]),
        "artifacts": artifacts,
        "evaluations": evals,
    }

    # Explicit Architecture E4 (8-bit quantized TFLite flatbuffer of E3)
    if model_id == "pvad2_e3_conformer_film_dvector_cos":
      e4_id = "pvad2_e4_conformer_int8"
      e4_dir = os.path.join(args.output_dir, e4_id)
      os.makedirs(e4_dir, exist_ok=True)
      for fname in (
          "model_config.json",
          "model.weights.h5",
          "model.safetensors",
          "model_fp32.tflite",
          "model_quantized.tflite",
          "speaker_subspace.npz",
      ):
        src_p = os.path.join(model_dir, fname)
        if os.path.exists(src_p):
          shutil.copyfile(src_p, os.path.join(e4_dir, fname))

      e4_quant_path = os.path.join(e4_dir, "model_quantized.tflite")
      e4_fp32_path = os.path.join(e4_dir, "model_fp32.tflite")
      e4_evals = {
          "personal_vad_non_concat": run_eval(
              pvad_model,
              test_v2_non_concat,
              eval_mode="personal_vad",
              tflite_path=e4_quant_path,
              output_dir=os.path.join(e4_dir, "eval_personal_vad_non_concat"),
          ),
          "personal_vad_concat": run_eval(
              pvad_model,
              test_v2_concat,
              eval_mode="personal_vad",
              tflite_path=e4_quant_path,
              output_dir=os.path.join(e4_dir, "eval_personal_vad_concat"),
          ),
          "personal_vad_concat_mtr": run_eval(
              pvad_model,
              test_v2_concat_mtr,
              eval_mode="personal_vad",
              tflite_path=e4_quant_path,
              output_dir=os.path.join(e4_dir, "eval_personal_vad_concat_mtr"),
          ),
          "personal_vad_concat_tflite_quant": evals[
              "personal_vad_concat_tflite_quant"
          ],
          "personal_vad_non_concat_tflite_quant": evals[
              "personal_vad_non_concat_tflite_quant"
          ],
          "per_language_concat": evaluate_per_language(
              pvad_model,
              test_single_v2,
              sc_baseline=False,
              tflite_path=e4_quant_path,
          ),
      }
      summary["models"][e4_id] = {
          "paper": "Personal VAD 2.0 (arXiv:2204.03793)",
          "backbone": "conformer",
          "conditioning_mode": "film_dvector_cos",
          "quantization": "int8_dynamic_range",
          "loss_type": "cross_entropy",
          "final_train_loss": float(losses[-1]),
          "artifacts": {
              **artifacts,
              "weights_path": os.path.join(e4_dir, "model.weights.h5"),
              "safetensors_path": os.path.join(e4_dir, "model.safetensors"),
              "config_path": os.path.join(e4_dir, "model_config.json"),
              "fp32_tflite_path": e4_fp32_path,
              "quant_tflite_path": e4_quant_path,
          },
          "evaluations": e4_evals,
      }

  summary["elapsed_seconds"] = round(time.time() - start_time, 2)
  summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  print(f"\nCompleted full benchmark suite in {summary['elapsed_seconds']}s.")
  print(f"Summary written to: {summary_path}")
  return summary


if __name__ == "__main__":
  main()
