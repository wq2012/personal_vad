"""End-to-end Multilingual LibriSpeech training, TFLite export, and evaluation.

Trains and evaluates all Personal VAD 1.0 (arXiv:1908.04284) and Personal VAD
2.0 (arXiv:2204.03793) architectures on the 8-language Multilingual LibriSpeech
(MLS) dataset (`de`, `en`, `es`, `fr`, `it`, `nl`, `pl`, `pt`), exports FP32
and 8-bit dynamic-range quantized `.tflite` flatbuffers, and writes verifiable
evaluation reports (`benchmark_summary.json` and per-model `metrics.json`,
`report.html`, `pr_curves.png`, `roc_curves.png`).
"""

import argparse
from collections.abc import Sequence
import json
import os
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
    learning_rate: float = 3e-3,
    seed: int = 42,
) -> tuple[model_lib.PersonalVadModel, list[float]]:
  """Builds and trains a `PersonalVadModel` on `utterances`."""
  tf.keras.utils.set_random_seed(seed)
  feature_dim = utterances[0].features.shape[1]
  emb_dim = utterances[0].speaker_embedding.shape[0]
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
      max_frames=110,
      shuffle=True,
      enrollment_less_prob=0.0,
      seed=seed,
  )
  losses = train.train_model(
      pvad_model=pvad_model,
      train_dataset=train_ds,
      loss_fn=loss_fn,
      epochs=epochs,
      learning_rate=learning_rate,
  )
  return pvad_model, losses


def evaluate_speaker_verification_auc(
    test_records: Sequence[dict[str, Any]],
    subspace: dict[str, np.ndarray],
) -> dict[str, float]:
  """Computes open-set speaker verification ROC-AUC on unseen test speakers."""
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
      # Pair with a different speaker's target embedding so speech frames are
      # non-target speaker speech (ntss = 2).
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
            )
        )
        continue
    outputs.append(utt)
  return outputs


def make_concat_utterances(
    single_utterances: Sequence[dataset.UtteranceData],
    num_passes: int = 4,
    enrollment_less: bool = False,
    convert_for_vad_baseline: bool = False,
    seed: int = 42,
) -> list[dataset.UtteranceData]:
  """Creates multi-speaker concatenated utterances across `num_passes`."""
  rng = np.random.default_rng(seed)
  all_concat: list[dataset.UtteranceData] = []
  for p in range(num_passes):
    pool = list(single_utterances)
    if p > 0:
      rng.shuffle(pool)
    all_concat.extend(
        dataset.concat_utterance_group(
            pool,
            min_utterances=1,
            max_utterances=3,
            convert_for_vad_baseline=convert_for_vad_baseline,
            enrollment_less_prob=1.0 if enrollment_less else 0.0,
            rng=rng,
        )
    )
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


def compute_bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    eval_mode: str,
    sc_baseline: bool = False,
    n_resamples: int = 200,
    seed: int = 42,
) -> dict[str, list[float]]:
  """Computes 95% non-parametric block-bootstrap CIs on evaluation metrics."""
  evaluator = eval_lib.PersonalVadEvaluator(eval_mode=eval_mode)
  prep_fn = evaluator._prepare_scores_and_labels  # noqa: SLF001
  proc_scores, _, binarized_labels = prep_fn(
      scores, labels, sc_baseline=sc_baseline
  )
  n_frames = len(binarized_labels)
  block_size = 50
  n_blocks = max(n_frames // block_size, 1)
  rng = np.random.default_rng(seed)

  map_samples: list[float] = []
  for _ in range(n_resamples):
    blk_idx = rng.integers(0, n_blocks, size=n_blocks)
    idx = np.concatenate(
        [
            np.arange(b * block_size, min((b + 1) * block_size, n_frames))
            for b in blk_idx
        ]
    )
    s_b = proc_scores[idx]
    y_true_oh = binarized_labels[idx]
    map_samples.append(
        float(
            sk_metrics.average_precision_score(
                y_true_oh, s_b, average="micro"
            )
        )
    )
  return {
      "mAP_micro_95_ci": [
          round(float(np.percentile(map_samples, 2.5)), 4),
          round(float(np.percentile(map_samples, 97.5)), 4),
      ]
  }


def run_eval(
    pvad_model: Any,
    utterances: Sequence[dataset.UtteranceData],
    eval_mode: str,
    output_dir: str,
    sc_baseline: bool = False,
    tflite_path: Optional[str] = None,
) -> dict[str, Any]:
  """Evaluates a Keras or TFLite model and writes metrics, plots, and HTML."""
  os.makedirs(output_dir, exist_ok=True)
  if tflite_path is not None:
    runner = tflite_export.TFLitePersonalVadRunner(tflite_path)
    scores, labels = evaluate.collect_tflite_predictions(
        runner, utterances, include_cosine_column=sc_baseline
    )
  else:
    scores, labels = evaluate.collect_model_predictions(
        pvad_model, utterances, include_cosine_column=sc_baseline
    )

  evaluator = eval_lib.PersonalVadEvaluator(eval_mode=eval_mode)
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
  )
  with open(
      os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8"
  ) as f:
    json.dump(res_dict, f, indent=2)
  return res_dict


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Run Personal VAD 1.0 and 2.0 benchmarks on MLS."
  )
  parser.add_argument(
      "--mls_dir",
      type=str,
      default="/usr/local/google/home/quanw/Data/multilingual_librispeech",
      help="Directory containing train_manifest.json and test_manifest.json.",
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
      default=18,
      help="Number of training epochs for Personal VAD 1.0 models.",
  )
  parser.add_argument(
      "--epochs_v2",
      type=int,
      default=22,
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
  test_manifest = os.path.join(args.mls_dir, "test_manifest.json")

  print("1. Loading Multilingual LibriSpeech audio and extracting features...")
  train_records = prepare_dataset.load_mls_manifest_records(train_manifest)
  test_records = prepare_dataset.load_mls_manifest_records(test_manifest)

  print("2. Fitting open-set LDA+WCCN speaker subspace on training speakers...")
  subspace = prepare_dataset.fit_lda_wccn_subspace(train_records)
  subspace_path = os.path.join(args.output_dir, "speaker_subspace.npz")
  np.savez(subspace_path, **subspace)
  sv_metrics = evaluate_speaker_verification_auc(test_records, subspace)
  print(
      f"   Unseen test speaker verification ROC-AUC: "
      f"{sv_metrics['roc_auc']:.4f} "
      f"(pos_cos={sv_metrics['mean_positive_cosine']:.4f}, "
      f"neg_cos={sv_metrics['mean_negative_cosine']:.4f})"
  )

  print("3. Preparing Personal VAD 1.0 (40-dim) and 2.0 (512-dim) splits...")
  train_single_v1 = prepare_dataset.build_utterances_from_mls_records(
      train_records, subspace, frontend_version="v1"
  )
  test_single_v1 = prepare_dataset.build_utterances_from_mls_records(
      test_records, subspace, frontend_version="v1"
  )
  train_single_v2 = prepare_dataset.build_utterances_from_mls_records(
      train_records, subspace, frontend_version="v2"
  )
  test_single_v2 = prepare_dataset.build_utterances_from_mls_records(
      test_records, subspace, frontend_version="v2"
  )

  # Evaluation datasets on unseen test speakers
  test_v1_concat = make_concat_utterances(test_single_v1, num_passes=2, seed=99)
  test_v1_non_concat = make_non_concat_utterances(test_single_v1)

  test_v2_concat = make_concat_utterances(test_single_v2, num_passes=2, seed=99)
  test_v2_non_concat = make_non_concat_utterances(test_single_v2)
  test_v2_no_enroll_concat = make_concat_utterances(
      test_single_v2, num_passes=2, enrollment_less=True, seed=99
  )
  test_v2_no_enroll_non_concat = make_non_concat_utterances(
      test_single_v2, enrollment_less=True
  )

  # Training datasets (with orthogonal speaker-subspace augmentation)
  train_v1_std_vad = make_concat_utterances(
      train_single_v1, num_passes=3, convert_for_vad_baseline=True, seed=42
  )
  train_v1_concat_base = make_concat_utterances(
      train_single_v1, num_passes=3, seed=42
  )
  train_v1_concat = prepare_dataset.augment_utterances_with_speaker_rotations(
      train_v1_concat_base, num_rotations=6, seed=42
  )

  train_v2_std_vad = make_concat_utterances(
      train_single_v2, num_passes=3, convert_for_vad_baseline=True, seed=42
  )
  train_v2_concat_base = make_concat_utterances(
      train_single_v2, num_passes=3, seed=42
  )
  train_v2_concat = prepare_dataset.augment_utterances_with_speaker_rotations(
      train_v2_concat_base, num_rotations=4, seed=42
  )

  # Unified E5 training dataset (Algorithm 1: p_0 = 0.20)
  rng_u = np.random.default_rng(42)
  train_v2_unified_base: list[dataset.UtteranceData] = []
  for p in range(3):
    pool = list(train_single_v2)
    if p > 0:
      rng_u.shuffle(pool)
    train_v2_unified_base.extend(
        dataset.concat_utterance_group(
            pool,
            min_utterances=1,
            max_utterances=3,
            enrollment_less_prob=0.20,
            rng=rng_u,
        )
    )
  train_v2_unified = prepare_dataset.augment_utterances_with_speaker_rotations(
      train_v2_unified_base, num_rotations=4, seed=42
  )

  summary: dict[str, Any] = {
      "dataset": {
          "mls_dir": args.mls_dir,
          "languages": ["de", "en", "es", "fr", "it", "nl", "pl", "pt"],
          "num_train_speakers": 52,
          "num_test_speakers": 35,
          "speaker_verification": sv_metrics,
      },
      "models": {},
  }

  # =========================================================================
  # Personal VAD 1.0 Experiments (arXiv:1908.04284)
  # =========================================================================
  v1_specs = [
      ("pvad1_sc_baseline", "sc", "cross_entropy", train_v1_std_vad),
      ("pvad1_st_ce", "st", "cross_entropy", train_v1_concat_base),
      ("pvad1_et_ce", "et", "cross_entropy", train_v1_concat),
      ("pvad1_set_ce", "set", "cross_entropy", train_v1_concat),
      ("pvad1_et_wpl", "et", "weighted_pairwise", train_v1_concat),
  ]

  for model_id, cond_mode, loss_type, tr_utts in v1_specs:
    model_dir = os.path.join(args.output_dir, model_id)
    print(
        f"\n=== Training {model_id} (mode={cond_mode}, loss={loss_type}) ==="
    )
    ep = 10 if cond_mode in ("et", "set") else 12
    bs = 64 if cond_mode in ("et", "set") else 32
    lr = 3.0e-3
    pvad_model, losses = _train_utterance_model(
        utterances=tr_utts,
        backbone="lstm_v1",
        conditioning_mode=cond_mode,
        loss_type=loss_type,
        epochs=ep,
        batch_size=bs,
        learning_rate=lr,
        seed=42,
    )
    artifacts = export_and_save_model_artifacts(
        pvad_model, model_dir, sequence_length=32
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
      evals["personal_vad_concat_tflite_quant"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="personal_vad",
          sc_baseline=True,
          tflite_path=artifacts["quant_tflite_path"],
          output_dir=os.path.join(model_dir, "eval_personal_vad_tflite_quant"),
      )
      evals["std_vad_non_concat"] = run_eval(
          pvad_model,
          test_v1_non_concat,
          eval_mode="std_vad_std_eval",
          output_dir=os.path.join(model_dir, "eval_std_vad_non_concat"),
      )
      evals["std_vad_concat"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="std_vad_std_eval",
          output_dir=os.path.join(model_dir, "eval_std_vad_concat"),
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
      evals["personal_vad_concat_tflite_quant"] = run_eval(
          pvad_model,
          test_v1_concat,
          eval_mode="personal_vad",
          tflite_path=artifacts["quant_tflite_path"],
          output_dir=os.path.join(model_dir, "eval_personal_vad_tflite_quant"),
      )
      if cond_mode == "et":
        evals["std_vad_non_concat"] = run_eval(
            pvad_model,
            test_v1_non_concat,
            eval_mode="personal_vad_std_eval",
            output_dir=os.path.join(model_dir, "eval_std_vad_non_concat"),
        )
        evals["std_vad_concat"] = run_eval(
            pvad_model,
            test_v1_concat,
            eval_mode="personal_vad_std_eval",
            output_dir=os.path.join(model_dir, "eval_std_vad_concat"),
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
      ("pvad2_b1_lstm_std_vad", "lstm_v2", "sc", train_v2_std_vad),
      ("pvad2_b2_conformer_std_vad", "conformer", "sc", train_v2_std_vad),
      ("pvad2_b0_lstm_concat", "lstm_v2", "concat", train_v2_concat),
      ("pvad2_e0_conformer_concat", "conformer", "concat", train_v2_concat),
      (
          "pvad2_e1_conformer_film_dvector",
          "conformer",
          "film_dvector",
          train_v2_concat,
      ),
      (
          "pvad2_e2_conformer_film_cos",
          "conformer",
          "film_cos",
          train_v2_concat,
      ),
      (
          "pvad2_e3_conformer_film_dvector_cos",
          "conformer",
          "film_dvector_cos",
          train_v2_concat,
      ),
      (
          "pvad2_e5_conformer_unified",
          "conformer",
          "film_dvector_cos",
          train_v2_unified,
      ),
  ]

  for model_id, backbone, cond_mode, tr_utts in v2_specs:
    model_dir = os.path.join(args.output_dir, model_id)
    print(
        f"\n=== Training {model_id} "
        f"(backbone={backbone}, mode={cond_mode}) ==="
    )
    ep = 8 if cond_mode != "sc" else 10
    bs = 64 if cond_mode != "sc" else 32
    pvad_model, losses = _train_utterance_model(
        utterances=tr_utts,
        backbone=backbone,
        conditioning_mode=cond_mode,
        loss_type="cross_entropy",
        epochs=ep,
        batch_size=bs,
        learning_rate=3.0e-3,
        seed=42,
    )
    artifacts = export_and_save_model_artifacts(
        pvad_model, model_dir, sequence_length=32
    )
    evals = {}
    if cond_mode == "sc":
      evals["std_vad_non_concat"] = run_eval(
          pvad_model,
          test_v2_non_concat,
          eval_mode="std_vad_std_eval",
          output_dir=os.path.join(model_dir, "eval_std_vad_non_concat"),
      )
      evals["std_vad_concat"] = run_eval(
          pvad_model,
          test_v2_concat,
          eval_mode="std_vad_std_eval",
          output_dir=os.path.join(model_dir, "eval_std_vad_concat"),
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
      evals["personal_vad_non_concat_tflite_quant"] = run_eval(
          pvad_model,
          test_v2_non_concat,
          eval_mode="personal_vad",
          tflite_path=artifacts["quant_tflite_path"],
          output_dir=os.path.join(
              model_dir, "eval_personal_vad_non_concat_tflite_quant"
          ),
      )
      evals["personal_vad_concat_tflite_quant"] = run_eval(
          pvad_model,
          test_v2_concat,
          eval_mode="personal_vad",
          tflite_path=artifacts["quant_tflite_path"],
          output_dir=os.path.join(
              model_dir, "eval_personal_vad_concat_tflite_quant"
          ),
      )
      if model_id in (
          "pvad2_e3_conformer_film_dvector_cos",
          "pvad2_e5_conformer_unified",
      ):
        evals["no_enrollment_non_concat"] = run_eval(
            pvad_model,
            test_v2_no_enroll_non_concat,
            eval_mode="personal_vad_std_eval",
            output_dir=os.path.join(model_dir, "eval_no_enrollment_non_concat"),
        )
        evals["no_enrollment_concat"] = run_eval(
            pvad_model,
            test_v2_no_enroll_concat,
            eval_mode="personal_vad_std_eval",
            output_dir=os.path.join(model_dir, "eval_no_enrollment_concat"),
        )
        evals["no_enrollment_concat_tflite_quant"] = run_eval(
            pvad_model,
            test_v2_no_enroll_concat,
            eval_mode="personal_vad_std_eval",
            tflite_path=artifacts["quant_tflite_path"],
            output_dir=os.path.join(
                model_dir, "eval_no_enrollment_concat_tflite_quant"
            ),
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

  summary["elapsed_seconds"] = round(time.time() - start_time, 2)
  summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  print(f"\nCompleted full benchmark suite in {summary['elapsed_seconds']}s.")
  print(f"Summary written to: {summary_path}")
  return summary


if __name__ == "__main__":
  main()
