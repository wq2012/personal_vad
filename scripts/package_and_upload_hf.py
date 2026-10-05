"""Packages trained Personal VAD 1.0 & 2.0 models and uploads to HF Hub."""

import argparse
from collections.abc import Sequence
import json
import os
import shutil
import sys
from typing import Any, Optional
import numpy as np
import soundfile as sf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import personal_vad  # noqa: E402

HF_MODEL_MAP: dict[str, dict[str, str]] = {
    "pvad1_sc_baseline": {
        "repo_id": "wq2012/personal-vad-v1-sc",
        "title": "Personal VAD 1.0 — Score Combination (SC Baseline)",
        "arch": (
            "2-Layer LSTM (64 units) Standard VAD + Cosine Score Combination"
        ),
        "paper_url": "https://arxiv.org/abs/1908.04284",
    },
    "pvad1_st_ce": {
        "repo_id": "wq2012/personal-vad-v1-st-ce",
        "title": "Personal VAD 1.0 — Score-Conditioned Training (ST, CE)",
        "arch": (
            "2-Layer LSTM (64 units) + 64-d FC conditioned on frame cosine "
            "similarity"
        ),
        "paper_url": "https://arxiv.org/abs/1908.04284",
    },
    "pvad1_et_ce": {
        "repo_id": "wq2012/personal-vad-v1-et-ce",
        "title": "Personal VAD 1.0 — Embedding-Conditioned Training (ET, CE)",
        "arch": (
            "2-Layer LSTM (64 units) + 64-d FC conditioned on 256-d speaker "
            "d-vector (Cross-Entropy)"
        ),
        "paper_url": "https://arxiv.org/abs/1908.04284",
    },
    "pvad1_set_ce": {
        "repo_id": "wq2012/personal-vad-v1-set-ce",
        "title": (
            "Personal VAD 1.0 — Score & Embedding Conditioned Training "
            "(SET, CE)"
        ),
        "arch": (
            "2-Layer LSTM (64 units) + 64-d FC conditioned on d-vector and "
            "frame cosine similarity"
        ),
        "paper_url": "https://arxiv.org/abs/1908.04284",
    },
    "pvad1_et_wpl": {
        "repo_id": "wq2012/personal-vad-v1-et-wpl",
        "title": (
            "Personal VAD 1.0 — Embedding-Conditioned Training with Weighted "
            "Pairwise Loss (ET, WPL)"
        ),
        "arch": (
            "2-Layer LSTM (64 units) + 64-d FC conditioned on 256-d speaker "
            "d-vector (Weighted Pairwise Loss, w_<ns,ntss>=0.1)"
        ),
        "paper_url": "https://arxiv.org/abs/1908.04284",
    },
    "pvad2_b1_lstm_std_vad": {
        "repo_id": "wq2012/personal-vad-v2-lstm-std-vad",
        "title": "Personal VAD 2.0 — B1 3-Layer LayerNorm LSTM Standard VAD",
        "arch": (
            "256-d FC + 3-Layer LayerNorm LSTM (256 units) 2-class "
            "Standard VAD"
        ),
        "paper_url": "https://arxiv.org/abs/2204.03793",
    },
    "pvad2_b2_conformer_std_vad": {
        "repo_id": "wq2012/personal-vad-v2-conformer-std-vad",
        "title": "Personal VAD 2.0 — B2 4-Layer Causal Conformer Standard VAD",
        "arch": (
            "4-Layer Causal Streaming Conformer (64-d, 8 heads, 31 "
            "left-context) 2-class Standard VAD"
        ),
        "paper_url": "https://arxiv.org/abs/2204.03793",
    },
    "pvad2_b0_lstm_concat": {
        "repo_id": "wq2012/personal-vad-v2-lstm-concat",
        "title": (
            "Personal VAD 2.0 — B0 3-Layer LayerNorm LSTM Concat Personal VAD"
        ),
        "arch": (
            "256-d FC + 3-Layer LayerNorm LSTM (256 units) with d-vector "
            "input concatenation"
        ),
        "paper_url": "https://arxiv.org/abs/2204.03793",
    },
    "pvad2_e0_conformer_concat": {
        "repo_id": "wq2012/personal-vad-v2-conformer-concat",
        "title": "Personal VAD 2.0 — E0 Causal Conformer Concat Personal VAD",
        "arch": (
            "4-Layer Causal Streaming Conformer (64-d) with d-vector input "
            "concatenation"
        ),
        "paper_url": "https://arxiv.org/abs/2204.03793",
    },
    "pvad2_e1_conformer_film_dvector": {
        "repo_id": "wq2012/personal-vad-v2-conformer-film-dvector",
        "title": "Personal VAD 2.0 — E1 Causal Conformer + FiLM (d-vector)",
        "arch": (
            "4-Layer Causal Streaming Conformer (64-d) + Feature-wise Linear "
            "Modulation (FiLM) on target d-vector"
        ),
        "paper_url": "https://arxiv.org/abs/2204.03793",
    },
    "pvad2_e2_conformer_film_cos": {
        "repo_id": "wq2012/personal-vad-v2-conformer-film-cos",
        "title": (
            "Personal VAD 2.0 — E2 Causal Conformer + Speaker Pre-Net "
            "(FiLM Cosine)"
        ),
        "arch": (
            "4-Layer Causal Streaming Conformer (64-d) + 2-Layer Conformer "
            "Speaker Pre-Net + FiLM on cosine similarity"
        ),
        "paper_url": "https://arxiv.org/abs/2204.03793",
    },
    "pvad2_e3_conformer_film_dvector_cos": {
        "repo_id": "wq2012/personal-vad-v2-conformer-film-dvector-cos",
        "title": (
            "Personal VAD 2.0 — E3 Causal Conformer + Speaker Pre-Net + FiLM "
            "(d-vector & Cosine)"
        ),
        "arch": (
            "4-Layer Causal Streaming Conformer (64-d) + 2-Layer Speaker "
            "Pre-Net + FiLM on [d-vector, cosine]"
        ),
        "paper_url": "https://arxiv.org/abs/2204.03793",
    },
    "pvad2_e5_conformer_unified": {
        "repo_id": "wq2012/personal-vad-v2-conformer-unified",
        "title": (
            "Personal VAD 2.0 — E5 Unified Enrolled & Enrollment-Less Causal "
            "Conformer"
        ),
        "arch": (
            "4-Layer Causal Streaming Conformer (64-d) + Speaker Pre-Net + "
            "FiLM trained with Algorithm 1 (p_0 = 0.20)"
        ),
        "paper_url": "https://arxiv.org/abs/2204.03793",
    },
}


def build_model_card(
    model_id: str,
    meta: dict[str, str],
    model_summary: dict[str, Any],
    all_summary: dict[str, Any],
) -> str:
  """Generates a Hugging Face Model Card with zero LaTeX math."""
  art = model_summary["artifacts"]
  evals = model_summary["evaluations"]

  v1_rows = []
  for mid in [
      "pvad1_sc_baseline",
      "pvad1_st_ce",
      "pvad1_et_ce",
      "pvad1_set_ce",
      "pvad1_et_wpl",
  ]:
    if mid not in all_summary["models"]:
      continue
    m = all_summary["models"][mid]
    ev = m["evaluations"]["personal_vad_concat"]
    ev_q = m["evaluations"]["personal_vad_concat_tflite_quant"]
    ap = ev["average_precisions"]
    ci = ev.get("bootstrap_ci", {}).get("mAP_micro_95_ci", [0.0, 0.0])
    mark = "**" if mid == model_id else ""
    v1_rows.append(
        f"| {mark}`{mid}`{mark} | `{m['conditioning_mode'].upper()}` | "
        f"`{m['loss_type']}` | {ap['Speech (tss)']:.4f} | "
        f"{ap['Silence (ns)']:.4f} | {ap['Non-Target Speech (ntss)']:.4f} | "
        f"**{ev['mean_average_precision_micro']:.4f}** "
        f"([{ci[0]:.4f}, {ci[1]:.4f}]) | "
        f"{ev_q['mean_average_precision_micro']:.4f} | "
        f"{m['artifacts']['quant_size_kb']:.1f} KB |"
    )

  v2_rows = []
  for mid in [
      "pvad2_b0_lstm_concat",
      "pvad2_e0_conformer_concat",
      "pvad2_e1_conformer_film_dvector",
      "pvad2_e2_conformer_film_cos",
      "pvad2_e3_conformer_film_dvector_cos",
      "pvad2_e5_conformer_unified",
  ]:
    if mid not in all_summary["models"]:
      continue
    m = all_summary["models"][mid]
    ev_nc = m["evaluations"]["personal_vad_non_concat"]
    ev_c = m["evaluations"]["personal_vad_concat"]
    ev_cq = m["evaluations"]["personal_vad_concat_tflite_quant"]
    ap_c = ev_c["average_precisions"]
    mark = "**" if mid == model_id else ""
    v2_rows.append(
        f"| {mark}`{mid}`{mark} | `{m['backbone']}` | "
        f"`{m['conditioning_mode']}` | "
        f"{ev_nc['mean_average_precision_micro']:.4f} | "
        f"{ap_c['Speech (tss)']:.4f} | {ap_c['Silence (ns)']:.4f} | "
        f"{ap_c['Non-Target Speech (ntss)']:.4f} | "
        f"**{ev_c['mean_average_precision_micro']:.4f}** | "
        f"{ev_cq['mean_average_precision_micro']:.4f} | "
        f"{m['artifacts']['quant_size_kb']:.1f} KB |"
    )

  primary_eval_key = (
      "personal_vad_concat"
      if "personal_vad_concat" in evals
      else "std_vad_concat"
  )
  primary_eval = evals[primary_eval_key]

  v1_table = "\n".join(v1_rows)
  v2_table = "\n".join(v2_rows)
  feat_dim = 40 if model_summary["backbone"] == "lstm_v1" else 512

  lines = [
      "---",
      "license: apache-2.0",
      "library_name: personal-vad",
      "tags:",
      "  - audio",
      "  - voice-activity-detection",
      "  - personal-vad",
      "  - speaker-conditioning",
      "  - conformer",
      "  - lstm",
      "  - tensorflow",
      "  - tflite",
      "  - safetensors",
      "datasets:",
      "  - facebook/multilingual_librispeech",
      "metrics:",
      "  - average_precision",
      "  - roc_auc",
      "---",
      "",
      f"# {meta['title']} (`{meta['repo_id']}`)",
      "",
      "**This is not an officially supported Google product.**",
      "",
      (
          "> **Open-Source Reproduction Notice:** This model is part of an "
          "independent open-source reproduction of the **Personal VAD 1.0** "
          "([Ding et al., Odyssey 2020](https://arxiv.org/abs/1908.04284)) and "
          "**Personal VAD 2.0** ([Ding et al., INTERSPEECH 2022]"
          "(https://arxiv.org/abs/2204.03793)) papers. Because the original "
          "papers were developed using internal datasets and infrastructure, "
          "this model was trained from scratch using the open-source "
          "[`personal-vad`](https://github.com/wq2012/personal_vad) library "
          "on the public 8-language **Multilingual LibriSpeech (MLS)** corpus "
          "(`de`, `en`, `es`, `fr`, `it`, `nl`, `pl`, `pt`)."
      ),
      "",
      "## Model Summary",
      "",
      f"- **Model ID**: `{model_id}` (`{meta['repo_id']}`)",
      f"- **Paper**: [{model_summary['paper']}]({meta['paper_url']})",
      f"- **Architecture**: {meta['arch']}",
      f"- **Backbone**: `{model_summary['backbone']}`",
      f"- **Conditioning Mode**: `{model_summary['conditioning_mode']}`",
      f"- **Loss Function**: `{model_summary['loss_type']}`",
      f"- **Trainable Parameters**: `{art['num_parameters']:,}`",
      (
          "- **Safetensors Checkpoint (`model.safetensors`)**: "
          f"`{art['safetensors_size_kb']:.2f} KB`"
      ),
      (
          "- **FP32 TFLite Flatbuffer (`model_fp32.tflite`)**: "
          f"`{art['fp32_size_kb']:.2f} KB`"
      ),
      (
          "- **8-Bit Quantized TFLite (`model_quantized.tflite`)**: "
          f"`{art['quant_size_kb']:.2f} KB`"
      ),
      (
          "- **Primary Concatenated Test Set micro-mAP**: "
          f"**`{primary_eval['mean_average_precision_micro']:.4f}`** "
          f"(Overall frame accuracy: `{primary_eval['overall_accuracy']:.4f}`)"
      ),
      "",
      "## Included Files",
      "",
      "- `model.safetensors`: Safetensors weights for secure, fast loading.",
      "- `model.weights.h5`: Keras HDF5 checkpoint weights.",
      "- `model_config.json`: Serialized `personal_vad.ModelConfig` spec.",
      "- `model_fp32.tflite`: Unquantized 32-bit float TFLite flatbuffer.",
      (
          "- `model_quantized.tflite`: 8-bit dynamic-range quantized TFLite "
          "flatbuffer for on-device deployment."
      ),
      (
          "- `speaker_subspace.npz`: Open-set regularized LDA+PCA speaker "
          "subspace projection matrix (`256`-d L2-normalized d-vectors, "
          "trained from scratch on the 98 MLS training speakers)."
      ),
      (
          "- `evaluation_metrics.json`: Full test-set evaluation metrics on "
          "35 unseen multilingual speakers (with 95% bootstrap CIs)."
      ),
      "",
      "## Speaker Embedding (`d-vector`) Extractor Design & Transparency",
      "",
      (
          "In the original Personal VAD 1.0 and 2.0 papers, speaker "
          "embeddings (`e_target`) and frame-level cosine scores (`s_t`) were "
          "extracted using Google's internal 3-layer LSTM speaker verification "
          "network (`4.88M` parameters) trained with Generalized End-to-End "
          "(`GE2E`) loss on proprietary vendor-collected corpora."
      ),
      "",
      (
          "To keep this open-source reproduction completely self-contained in "
          "pure TensorFlow/NumPy with zero external PyTorch/SpeechBrain/JAX "
          "dependencies, **we trained our own lightweight, non-neural open-set "
          "256-D speaker subspace extractor (`OpenSetSpeakerSubspace`, "
          "serialized in `speaker_subspace.npz`) from scratch on the 98 "
          "training speakers (`922` utterances) of the 8-language Multilingual "
          "LibriSpeech (MLS) dataset**:"
      ),
      (
          "1. **360-D Multi-Resolution Acoustic Summary**: Over voiced speech "
          "frames (or a 31-frame / ~310 ms causal sliding window for "
          "frame-level scores `s_t`), it extracts 40-D log-Mel `[mean, std, "
          "p10, p90, delta_std]` (200-D) concatenated with 80-D log-Mel "
          "`[mean, std]` (160-D)."
      ),
      (
          "2. **256-D Regularized LDA + PCA Projection**: Applies Z-score "
          "standardization, projects onto a 256-D discriminative subspace "
          "fitted via regularized Linear Discriminant Analysis (LDA) and "
          "orthogonal PCA completion on the 98 training speakers, and "
          "L2-normalizes the resulting vector (`||e||_2 = 1`)."
      ),
      (
          "3. **Unseen-Speaker Verification Accuracy**: On the **35 held-out "
          "unseen test speakers** (`315` utterances across 8 languages), "
          "`speaker_subspace.npz` achieves an open-set speaker verification "
          "**ROC-AUC of `0.9521`** (mean positive cosine similarity `0.7130` "
          "vs. negative `-0.0082`). Users may also pass any external 256-D "
          "L2-normalized neural d-vector directly to `PersonalVadModel` or "
          "`PersonalVadInferenceEngine`."
      ),
      "",
      "## Quickstart Usage",
      "",
      "### 1. High-Level Audio Inference (`personal_vad.load_pretrained`)",
      "",
      "```python",
      "import personal_vad",
      "",
      f'engine = personal_vad.load_pretrained("{meta["repo_id"]}")',
      "result = engine.predict_audio(",
      '    audio="conversation_16k.wav",',
      '    enrollment_audio_or_embedding="target_speaker_enroll_16k.wav",',
      "    use_tflite=False,",
      "    threshold=0.5,",
      ")",
      'print("Summary:", result["summary"])',
      'print("Posteriors shape:", result["posteriors"].shape)',
      "```",
      "",
      "### 2. On-Device 8-Bit Quantized `.tflite` Execution",
      "",
      "```python",
      "from huggingface_hub import hf_hub_download",
      "import numpy as np",
      "import personal_vad",
      "",
      "tflite_path = hf_hub_download(",
      f'    repo_id="{meta["repo_id"]}", filename="model_quantized.tflite"',
      ")",
      "runner = personal_vad.TFLitePersonalVadRunner(tflite_path)",
      f"features = np.zeros((32, {feat_dim}), dtype=np.float32)",
      "speaker_embedding = np.ones((256,), dtype=np.float32) / np.sqrt(256.0)",
      "cosine_score = np.full((32,), 0.8, dtype=np.float32)",
      "probs = runner.predict(",
      "    features=features,",
      "    speaker_embedding=speaker_embedding,",
      "    cosine_score=cosine_score,",
      ")",
      'print("TFLite output shape:", probs.shape)',
      "```",
      "",
      "## Open-Source Multilingual LibriSpeech Benchmark Results",
      "",
      (
          "All models were trained on 98 training speakers (`922` utterances) "
          "across 8 languages (`de`, `en`, `es`, `fr`, `it`, `nl`, `pl`, `pt`) "
          "and evaluated on **35 held-out unseen test speakers** (`315` "
          "utterances, zero speaker overlap)."
      ),
      "",
      "### Personal VAD 1.0 Models (Concatenated Multi-Speaker Test Set)",
      "",
      (
          "| Model ID | Mode | Loss | Target Speech AP (`tss`) | "
          "Silence AP (`ns`) | Non-Target Speech AP (`ntss`) | "
          "Micro mAP (95% Bootstrap CI) | INT8 TFLite mAP | Quantized Size |"
      ),
      "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
      v1_table,
      "",
      "### Personal VAD 2.0 Models (Non-Concat & Concat Test Sets)",
      "",
      (
          "| Model ID | Backbone | Conditioning | Non-Concat mAP | "
          "Concat `tss` AP | Concat `ns` AP | Concat `ntss` AP | "
          "Concat Micro mAP | INT8 TFLite mAP | Quantized Size |"
      ),
      (
          "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | "
          ":---: | :---: |"
      ),
      v2_table,
      "",
      "## Original Paper Reference Benchmarks",
      "",
      "### Personal VAD 1.0 (arXiv:1908.04284, Table 1)",
      "",
      (
          "| Method | Loss | Without MTR `tss` | Without MTR `ns` | "
          "Without MTR `ntss` | Without MTR `mean` | With MTR `tss` | "
          "With MTR `ns` | With MTR `ntss` | With MTR `mean` | Params |"
      ),
      (
          "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | "
          ":---: | :---: | :---: |"
      ),
      (
          "| SC (baseline) | N/A | 0.886 | 0.970 | 0.872 | 0.900 | "
          "0.777 | 0.908 | 0.768 | 0.801 | 4.88M (SV) + 0.06M (VAD) |"
      ),
      (
          "| ST | CE | 0.956 | 0.968 | 0.956 | 0.957 | 0.905 | 0.885 | "
          "0.905 | 0.901 | 4.88M (SV) + 0.06M (PVAD) |"
      ),
      (
          "| ET | CE | 0.932 | 0.962 | 0.946 | 0.946 | 0.878 | 0.873 | "
          "0.890 | 0.883 | **0.13M (PVAD)** |"
      ),
      (
          "| SET | CE | **0.970** | 0.969 | 0.972 | 0.969 | **0.938** | "
          "0.888 | 0.938 | 0.928 | 4.88M (SV) + 0.13M (PVAD) |"
      ),
      (
          "| ET | WPL | 0.955 | 0.965 | 0.961 | 0.959 | 0.916 | 0.883 | "
          "0.920 | 0.912 | **0.13M (PVAD)** |"
      ),
      "",
      (
          "### Personal VAD 2.0 (arXiv:2204.03793, Table 1 & Table 2 "
          "Downstream ASR WER %)"
      ),
      "",
      (
          "| Exp | Model | Enrolled Non-Concat WER (%) | "
          "Enrolled Concat WER (%) | Enrollment-Less VS WER (%) | "
          "Enrollment-Less Non-Concat WER (%) | Size (MB) | FLOPs (M) |"
      ),
      "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
      "| B1 | LSTM Standard VAD | N/A | N/A | 7.2 | 11.5 | 1.5 | 3.41 |",
      "| B2 | Conformer Standard VAD | N/A | N/A | 6.9 | 10.1 | 0.7 | 8.77 |",
      (
          "| B0 | Personal VAD (LSTM Concat) | 17.9 | 41.0 | >=100.0 | "
          ">=100.0 | 5.8 | 3.54 |"
      ),
      "| E0 | + Conformer Concat | 15.3 | 31.5 | — | — | 2.8 | 9.51 |",
      "| E1 | + FiLM d-vector | 11.3 | 29.5 | — | — | 2.8 | 9.58 |",
      "| E2 | + Speaker PreNet (cos) | 11.7 | 27.5 | — | — | 4.0 | 9.51 |",
      "| E3 | + FiLM d-vector & cos | 11.5 | 27.6 | — | — | 4.0 | 9.58 |",
      (
          "| E4 | + 8-bit quantization | 12.2 | 27.2 | >=100.0 | "
          ">=100.0 | 1.0 | 9.58 |"
      ),
      (
          "| E5 | Personal VAD 2.0 (Unified) | 12.4 | 32.7 | 7.0 | "
          "10.1 | 1.0 | 9.58 |"
      ),
      "",
      "## Citation",
      "",
      "```bibtex",
      "@inproceedings{ding2020personal,",
      "  title={Personal VAD: Speaker-Conditioned Voice Activity Detection},",
      (
          "  author={Ding, Shaojin and Wang, Quan and Chang, Shuo-yiin and "
          "Wan, Li and Moreno, Ignacio Lopez},"
      ),
      (
          "  booktitle={Proc. Odyssey The Speaker and Language Recognition "
          "Workshop},"
      ),
      "  pages={433--439},",
      "  year={2020}",
      "}",
      "",
      "@inproceedings{ding2022personal,",
      (
          "  title={Personal VAD 2.0: Optimizing Personal Voice Activity "
          "Detection for On-Device Speech Recognition},"
      ),
      (
          "  author={Ding, Shaojin and Rikhye, Rajeev and Liang, Qiao and "
          "He, Yanzhang and Wang, Quan and Narayanan, Arun and O'Malley, Tom "
          "and McGraw, Ian},"
      ),
      "  booktitle={Proc. INTERSPEECH},",
      "  pages={3744--3748},",
      "  year={2022}",
      "}",
      "```",
      "",
  ]
  return "\n".join(lines)


def generate_hf_space_examples(
    mls_dir: str,
    pretrained_root: str,
    space_examples_dir: str,
) -> None:
  """Generates 3 multi-speaker conversation examples + JSON for hf_space."""
  os.makedirs(space_examples_dir, exist_ok=True)
  test_manifest = os.path.join(mls_dir, "test_manifest.json")
  with open(test_manifest, "r", encoding="utf-8") as f:
    records = json.load(f)

  by_spk: dict[str, dict[str, list[dict[str, Any]]]] = {}
  for r in records:
    key = f"{r['language']}_{r['speaker_id']}"
    by_spk.setdefault(key, {"enroll": [], "eval": []})
    bucket = "enroll" if r["role"] == "enroll" else "eval"
    by_spk[key][bucket].append(r)

  valid_spks = [
      k for k, v in by_spk.items() if v["enroll"] and len(v["eval"]) >= 2
  ]
  std_eng = personal_vad.load_pretrained(
      os.path.join(pretrained_root, "pvad2_b2_conformer_std_vad")
  )
  v1_eng = personal_vad.load_pretrained(
      os.path.join(pretrained_root, "pvad1_et_wpl")
  )
  v2_e3_eng = personal_vad.load_pretrained(
      os.path.join(pretrained_root, "pvad2_e3_conformer_film_dvector_cos")
  )
  v2_e5_eng = personal_vad.load_pretrained(
      os.path.join(pretrained_root, "pvad2_e5_conformer_unified")
  )

  pairings = [
      (
          0,
          3,
          6,
          (
              "Example 1: Multilingual 2-Speaker Turn (Target Speaker A -> "
              "Non-Target Speaker B)"
          ),
      ),
      (
          1,
          5,
          9,
          (
              "Example 2: 3-Speaker Conversation (Non-Target B -> Target "
              "Speaker A -> Non-Target C)"
          ),
      ),
      (
          2,
          7,
          11,
          "Example 3: Target Speaker A Alternating with Competing Speaker B",
      ),
  ]

  examples_payload = []
  rng = np.random.default_rng(2026)

  for idx, (s_a_idx, s_b_idx, s_c_idx, title) in enumerate(pairings, start=1):
    spk_a = valid_spks[s_a_idx % len(valid_spks)]
    spk_b = valid_spks[s_b_idx % len(valid_spks)]
    spk_c = valid_spks[s_c_idx % len(valid_spks)]

    enroll_path_src = by_spk[spk_a]["enroll"][0]["audio_path"]
    enroll_wav, sr = sf.read(enroll_path_src, dtype="float32")
    if enroll_wav.ndim > 1:
      enroll_wav = np.mean(enroll_wav, axis=-1)
    enroll_wav = enroll_wav[: int(2.2 * sr)]

    def _clip(spk_key: str, slot: int) -> np.ndarray:
      p = by_spk[spk_key]["eval"][slot % len(by_spk[spk_key]["eval"])][
          "audio_path"
      ]
      w, _ = sf.read(p, dtype="float32")
      if w.ndim > 1:
        w = np.mean(w, axis=-1)
      w = w[: int(1.8 * sr)]
      pk = float(np.max(np.abs(w)))
      if pk > 1e-4:
        w = w / pk * 0.85
      sil = rng.normal(0.0, 1e-4, size=(int(0.20 * sr),)).astype(np.float32)
      return np.concatenate([sil, w, sil], axis=0)

    if idx == 1:
      conv_wav = np.concatenate([_clip(spk_a, 0), _clip(spk_b, 0)], axis=0)
    elif idx == 2:
      conv_wav = np.concatenate(
          [_clip(spk_b, 0), _clip(spk_a, 0), _clip(spk_c, 0)], axis=0
      )
    else:
      conv_wav = np.concatenate(
          [_clip(spk_a, 0), _clip(spk_b, 0), _clip(spk_a, 1)], axis=0
      )

    enr_out = os.path.join(space_examples_dir, f"enroll_{idx}.wav")
    conv_out = os.path.join(space_examples_dir, f"conversation_{idx}.wav")
    filt_out = os.path.join(space_examples_dir, f"filtered_{idx}.wav")
    sf.write(enr_out, enroll_wav, sr)
    sf.write(conv_out, conv_wav, sr)

    std_res = std_eng.predict_audio(
        conv_out, enrollment_audio_or_embedding=None
    )
    v1_res = v1_eng.predict_audio(
        conv_out, enrollment_audio_or_embedding=enr_out
    )
    v2_res = v2_e3_eng.predict_audio(
        conv_out, enrollment_audio_or_embedding=enr_out
    )
    v2_no_enr = v2_e5_eng.predict_audio(
        conv_out, enrollment_audio_or_embedding=None
    )

    sf.write(filt_out, v2_res["filtered_audio"], sr)

    num_frames = v2_res["posteriors"].shape[0]
    hop = max(1, len(conv_wav) // num_frames)
    env = [
        round(
            float(
                np.max(
                    np.abs(
                        conv_wav[i * hop:min((i + 1) * hop, len(conv_wav))]
                    )
                )
            ),
            4,
        )
        for i in range(num_frames)
    ]

    examples_payload.append({
        "id": idx,
        "title": title,
        "target_speaker": spk_a,
        "enroll_wav": f"examples/enroll_{idx}.wav",
        "conversation_wav": f"examples/conversation_{idx}.wav",
        "filtered_wav": f"examples/filtered_{idx}.wav",
        "waveform_envelope": env,
        "std_posteriors": [
            [round(float(x), 4) for x in row] for row in std_res["posteriors"]
        ],
        "pvad1_posteriors": [
            [round(float(x), 4) for x in row] for row in v1_res["posteriors"]
        ],
        "pvad2_posteriors": [
            [round(float(x), 4) for x in row] for row in v2_res["posteriors"]
        ],
        "pvad2_no_enroll_posteriors": [
            [round(float(x), 4) for x in row]
            for row in v2_no_enr["posteriors"]
        ],
    })

  with open(
      os.path.join(space_examples_dir, "demo_examples.json"),
      "w",
      encoding="utf-8",
  ) as f:
    json.dump({"examples": examples_payload}, f, indent=2)


def package_all_models(
    source_models_dir: str,
    target_pretrained_dir: str,
) -> dict[str, Any]:
  """Packages all 13 models into `pretrained_models/<model_id>/`."""
  summary_path = os.path.join(source_models_dir, "benchmark_summary.json")
  with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

  subspace_src = os.path.join(source_models_dir, "speaker_subspace.npz")
  os.makedirs(target_pretrained_dir, exist_ok=True)
  shutil.copyfile(
      summary_path,
      os.path.join(target_pretrained_dir, "benchmark_summary.json"),
  )

  for model_id, meta in HF_MODEL_MAP.items():
    if model_id not in summary["models"]:
      continue
    src_dir = os.path.join(source_models_dir, model_id)
    dst_dir = os.path.join(target_pretrained_dir, model_id)
    os.makedirs(dst_dir, exist_ok=True)

    for fname in [
        "model_config.json",
        "model.weights.h5",
        "model.safetensors",
        "model_fp32.tflite",
        "model_quantized.tflite",
    ]:
      s_file = os.path.join(src_dir, fname)
      if os.path.exists(s_file):
        shutil.copyfile(s_file, os.path.join(dst_dir, fname))

    if os.path.exists(subspace_src):
      shutil.copyfile(
          subspace_src, os.path.join(dst_dir, "speaker_subspace.npz")
      )

    model_summary = summary["models"][model_id]
    with open(
        os.path.join(dst_dir, "evaluation_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
      json.dump(model_summary, f, indent=2)

    card_md = build_model_card(model_id, meta, model_summary, summary)
    with open(os.path.join(dst_dir, "README.md"), "w", encoding="utf-8") as f:
      f.write(card_md)

  return summary


def upload_to_huggingface(
    target_pretrained_dir: str,
    hf_space_dir: str,
    space_repo_id: str = "wq2012/personal_vad",
) -> None:
  """Uploads all 13 model repositories and the Space repository as PRIVATE."""
  from huggingface_hub import HfApi  # noqa: E402

  api = HfApi()
  for model_id, meta in HF_MODEL_MAP.items():
    folder = os.path.join(target_pretrained_dir, model_id)
    if not os.path.isdir(folder):
      continue
    repo_id = meta["repo_id"]
    print(f"Uploading private model repo: {repo_id} ...")
    api.create_repo(
        repo_id=repo_id, repo_type="model", private=True, exist_ok=True
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder,
        commit_message=(
            f"Upload {meta['title']} checkpoints, TFLite models, and metrics"
        ),
    )

  print(f"Uploading private Hugging Face Space: {space_repo_id} ...")
  try:
    api.create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=True,
        exist_ok=True,
    )
  except Exception:  # pylint: disable=broad-except
    api.create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="static",
        private=True,
        exist_ok=True,
    )
  api.upload_folder(
      repo_id=space_repo_id,
      repo_type="space",
      folder_path=hf_space_dir,
      commit_message="Upload Personal VAD 1.0 & 2.0 interactive demo Space",
  )
  print(
      "All Hugging Face models and Space uploaded successfully (private=True)."
  )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Package and upload Personal VAD models and Space to Hugging Face."
      )
  )
  parser.add_argument(
      "--source_models_dir",
      type=str,
      default="/usr/local/google/home/quanw/Data/pvad_MLS_Watch_models",
  )
  parser.add_argument(
      "--mls_dir",
      type=str,
      default="/usr/local/google/home/quanw/Data/multilingual_librispeech",
  )
  parser.add_argument(
      "--pretrained_dir",
      type=str,
      default=os.path.join(REPO_ROOT, "pretrained_models"),
  )
  parser.add_argument(
      "--hf_space_dir",
      type=str,
      default=os.path.join(REPO_ROOT, "hf_space"),
  )
  parser.add_argument(
      "--upload",
      action="store_true",
      help=(
          "Upload packaged models and Space to Hugging Face Hub (private=True)."
      ),
  )
  return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
  args = parse_args(argv)
  package_all_models(args.source_models_dir, args.pretrained_dir)
  generate_hf_space_examples(
      mls_dir=args.mls_dir,
      pretrained_root=args.pretrained_dir,
      space_examples_dir=os.path.join(args.hf_space_dir, "examples"),
  )
  if args.upload:
    upload_to_huggingface(args.pretrained_dir, args.hf_space_dir)


if __name__ == "__main__":
  main()
