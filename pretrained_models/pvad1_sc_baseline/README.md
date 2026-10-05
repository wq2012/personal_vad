---
license: apache-2.0
library_name: personal-vad
tags:
  - audio
  - voice-activity-detection
  - personal-vad
  - speaker-conditioning
  - conformer
  - lstm
  - tensorflow
  - tflite
  - safetensors
datasets:
  - facebook/multilingual_librispeech
metrics:
  - average_precision
  - roc_auc
---

# Personal VAD 1.0 — Score Combination (SC Baseline) (`wq2012/personal-vad-v1-sc`)

**This is not an officially supported Google product.**

> **Open-Source Reproduction Notice:** This model is part of an independent open-source reproduction of the **Personal VAD 1.0** ([Ding et al., Odyssey 2020](https://arxiv.org/abs/1908.04284)) and **Personal VAD 2.0** ([Ding et al., INTERSPEECH 2022](https://arxiv.org/abs/2204.03793)) papers. Because the original papers were developed using internal datasets and infrastructure, this model was trained from scratch using the open-source [`personal-vad`](https://github.com/wq2012/personal_vad) library on the public 8-language **Multilingual LibriSpeech (MLS)** corpus (`de`, `en`, `es`, `fr`, `it`, `nl`, `pl`, `pt`).

## Model Summary

- **Model ID**: `pvad1_sc_baseline` (`wq2012/personal-vad-v1-sc`)
- **Paper**: [Personal VAD 1.0 (arXiv:1908.04284)](https://arxiv.org/abs/1908.04284)
- **Architecture**: 2-Layer LSTM (64 units) Standard VAD + Cosine Score Combination
- **Backbone**: `lstm_v1`
- **Conditioning Mode**: `sc`
- **Loss Function**: `cross_entropy`
- **Trainable Parameters**: `64,194`
- **Safetensors Checkpoint (`model.safetensors`)**: `251.50 KB`
- **FP32 TFLite Flatbuffer (`model_fp32.tflite`)**: `369.30 KB`
- **8-Bit Quantized TFLite (`model_quantized.tflite`)**: `196.43 KB`
- **Primary Concatenated Test Set micro-mAP**: **`0.9161`** (Overall frame accuracy: `0.8584`)

## Included Files

- `model.safetensors`: Safetensors weights for secure, fast loading.
- `model.weights.h5`: Keras HDF5 checkpoint weights.
- `model_config.json`: Serialized `personal_vad.ModelConfig` spec.
- `model_fp32.tflite`: Unquantized 32-bit float TFLite flatbuffer.
- `model_quantized.tflite`: 8-bit dynamic-range quantized TFLite flatbuffer for on-device deployment.
- `speaker_subspace.npz`: Open-set regularized LDA+PCA speaker subspace projection matrix (`256`-d L2-normalized d-vectors, trained from scratch on the 98 MLS training speakers).
- `evaluation_metrics.json`: Full test-set evaluation metrics on 35 unseen multilingual speakers (with 95% bootstrap CIs).

## Speaker Embedding (`d-vector`) Extractor Design & Transparency

In the original Personal VAD 1.0 and 2.0 papers, speaker embeddings (`e_target`) and frame-level cosine scores (`s_t`) were extracted using Google's internal 3-layer LSTM speaker verification network (`4.88M` parameters) trained with Generalized End-to-End (`GE2E`) loss on proprietary vendor-collected corpora.

To keep this open-source reproduction completely self-contained in pure TensorFlow/NumPy with zero external PyTorch/SpeechBrain/JAX dependencies, **we trained our own lightweight, non-neural open-set 256-D speaker subspace extractor (`OpenSetSpeakerSubspace`, serialized in `speaker_subspace.npz`) from scratch on the 98 training speakers (`922` utterances) of the 8-language Multilingual LibriSpeech (MLS) dataset**:
1. **360-D Multi-Resolution Acoustic Summary**: Over voiced speech frames (or a 31-frame / ~310 ms causal sliding window for frame-level scores `s_t`), it extracts 40-D log-Mel `[mean, std, p10, p90, delta_std]` (200-D) concatenated with 80-D log-Mel `[mean, std]` (160-D).
2. **256-D Regularized LDA + PCA Projection**: Applies Z-score standardization, projects onto a 256-D discriminative subspace fitted via regularized Linear Discriminant Analysis (LDA) and orthogonal PCA completion on the 98 training speakers, and L2-normalizes the resulting vector (`||e||_2 = 1`).
3. **Unseen-Speaker Verification Accuracy**: On the **35 held-out unseen test speakers** (`315` utterances across 8 languages), `speaker_subspace.npz` achieves an open-set speaker verification **ROC-AUC of `0.9521`** (mean positive cosine similarity `0.7130` vs. negative `-0.0082`). Users may also pass any external 256-D L2-normalized neural d-vector directly to `PersonalVadModel` or `PersonalVadInferenceEngine`.

## Quickstart Usage

### 1. High-Level Audio Inference (`personal_vad.load_pretrained`)

```python
import personal_vad

engine = personal_vad.load_pretrained("wq2012/personal-vad-v1-sc")
result = engine.predict_audio(
    audio="conversation_16k.wav",
    enrollment_audio_or_embedding="target_speaker_enroll_16k.wav",
    use_tflite=False,
    threshold=0.5,
)
print("Summary:", result["summary"])
print("Posteriors shape:", result["posteriors"].shape)
```

### 2. On-Device 8-Bit Quantized `.tflite` Execution

```python
from huggingface_hub import hf_hub_download
import numpy as np
import personal_vad

tflite_path = hf_hub_download(
    repo_id="wq2012/personal-vad-v1-sc", filename="model_quantized.tflite"
)
runner = personal_vad.TFLitePersonalVadRunner(tflite_path)
features = np.zeros((32, 40), dtype=np.float32)
speaker_embedding = np.ones((256,), dtype=np.float32) / np.sqrt(256.0)
cosine_score = np.full((32,), 0.8, dtype=np.float32)
probs = runner.predict(
    features=features,
    speaker_embedding=speaker_embedding,
    cosine_score=cosine_score,
)
print("TFLite output shape:", probs.shape)
```

## Open-Source Multilingual LibriSpeech Benchmark Results

All models were trained on 98 training speakers (`922` utterances) across 8 languages (`de`, `en`, `es`, `fr`, `it`, `nl`, `pl`, `pt`) and evaluated on **35 held-out unseen test speakers** (`315` utterances, zero speaker overlap).

### Personal VAD 1.0 Models (Concatenated Multi-Speaker Test Set)

| Model ID | Mode | Loss | Target Speech AP (`tss`) | Silence AP (`ns`) | Non-Target Speech AP (`ntss`) | Micro mAP (95% Bootstrap CI) | INT8 TFLite mAP | Quantized Size |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pvad1_sc_baseline`** | `SC` | `cross_entropy` | 0.9435 | 0.9604 | 0.7848 | **0.9161** ([0.9056, 0.9277]) | 0.9090 | 196.4 KB |
| `pvad1_st_ce` | `ST` | `cross_entropy` | 0.9394 | 0.9336 | 0.7550 | **0.8834** ([0.8668, 0.9014]) | 0.8804 | 197.4 KB |
| `pvad1_et_ce` | `ET` | `cross_entropy` | 0.9362 | 0.9536 | 0.7817 | **0.8665** ([0.8460, 0.8897]) | 0.8609 | 262.4 KB |
| `pvad1_set_ce` | `SET` | `cross_entropy` | 0.9385 | 0.9510 | 0.7725 | **0.8469** ([0.8237, 0.8712]) | 0.8521 | 262.7 KB |
| `pvad1_et_wpl` | `ET` | `weighted_pairwise` | 0.9432 | 0.9433 | 0.7695 | **0.8652** ([0.8467, 0.8853]) | 0.8691 | 262.4 KB |

### Personal VAD 2.0 Models (Non-Concat & Concat Test Sets)

| Model ID | Backbone | Conditioning | Non-Concat mAP | Concat `tss` AP | Concat `ns` AP | Concat `ntss` AP | Concat Micro mAP | INT8 TFLite mAP | Quantized Size |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pvad2_b0_lstm_concat` | `lstm_v2` | `concat` | 0.7006 | 0.7464 | 0.9378 | 0.3172 | **0.8088** | 0.8092 | 2007.9 KB |
| `pvad2_e0_conformer_concat` | `conformer` | `concat` | 0.7408 | 0.8336 | 0.9678 | 0.5071 | **0.8502** | 0.8329 | 898.3 KB |
| `pvad2_e1_conformer_film_dvector` | `conformer` | `film_dvector` | 0.9025 | 0.9370 | 0.9710 | 0.7709 | **0.9060** | 0.9009 | 917.0 KB |
| `pvad2_e2_conformer_film_cos` | `conformer` | `film_cos` | 0.9207 | 0.9492 | 0.9739 | 0.8172 | **0.9153** | 0.9086 | 1364.0 KB |
| `pvad2_e3_conformer_film_dvector_cos` | `conformer` | `film_dvector_cos` | 0.9158 | 0.9476 | 0.9776 | 0.8169 | **0.9222** | 0.9127 | 1397.8 KB |
| `pvad2_e5_conformer_unified` | `conformer` | `film_dvector_cos` | 0.8574 | 0.9250 | 0.9644 | 0.7558 | **0.9092** | 0.8939 | 1397.8 KB |

## Original Paper Reference Benchmarks

### Personal VAD 1.0 (arXiv:1908.04284, Table 1)

| Method | Loss | Without MTR `tss` | Without MTR `ns` | Without MTR `ntss` | Without MTR `mean` | With MTR `tss` | With MTR `ns` | With MTR `ntss` | With MTR `mean` | Params |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SC (baseline) | N/A | 0.886 | 0.970 | 0.872 | 0.900 | 0.777 | 0.908 | 0.768 | 0.801 | 4.88M (SV) + 0.06M (VAD) |
| ST | CE | 0.956 | 0.968 | 0.956 | 0.957 | 0.905 | 0.885 | 0.905 | 0.901 | 4.88M (SV) + 0.06M (PVAD) |
| ET | CE | 0.932 | 0.962 | 0.946 | 0.946 | 0.878 | 0.873 | 0.890 | 0.883 | **0.13M (PVAD)** |
| SET | CE | **0.970** | 0.969 | 0.972 | 0.969 | **0.938** | 0.888 | 0.938 | 0.928 | 4.88M (SV) + 0.13M (PVAD) |
| ET | WPL | 0.955 | 0.965 | 0.961 | 0.959 | 0.916 | 0.883 | 0.920 | 0.912 | **0.13M (PVAD)** |

### Personal VAD 2.0 (arXiv:2204.03793, Table 1 & Table 2 Downstream ASR WER %)

| Exp | Model | Enrolled Non-Concat WER (%) | Enrolled Concat WER (%) | Enrollment-Less VS WER (%) | Enrollment-Less Non-Concat WER (%) | Size (MB) | FLOPs (M) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| B1 | LSTM Standard VAD | N/A | N/A | 7.2 | 11.5 | 1.5 | 3.41 |
| B2 | Conformer Standard VAD | N/A | N/A | 6.9 | 10.1 | 0.7 | 8.77 |
| B0 | Personal VAD (LSTM Concat) | 17.9 | 41.0 | >=100.0 | >=100.0 | 5.8 | 3.54 |
| E0 | + Conformer Concat | 15.3 | 31.5 | — | — | 2.8 | 9.51 |
| E1 | + FiLM d-vector | 11.3 | 29.5 | — | — | 2.8 | 9.58 |
| E2 | + Speaker PreNet (cos) | 11.7 | 27.5 | — | — | 4.0 | 9.51 |
| E3 | + FiLM d-vector & cos | 11.5 | 27.6 | — | — | 4.0 | 9.58 |
| E4 | + 8-bit quantization | 12.2 | 27.2 | >=100.0 | >=100.0 | 1.0 | 9.58 |
| E5 | Personal VAD 2.0 (Unified) | 12.4 | 32.7 | 7.0 | 10.1 | 1.0 | 9.58 |

## Citation

```bibtex
@inproceedings{ding2020personal,
  title={Personal VAD: Speaker-Conditioned Voice Activity Detection},
  author={Ding, Shaojin and Wang, Quan and Chang, Shuo-yiin and Wan, Li and Moreno, Ignacio Lopez},
  booktitle={Proc. Odyssey The Speaker and Language Recognition Workshop},
  pages={433--439},
  year={2020}
}

@inproceedings{ding2022personal,
  title={Personal VAD 2.0: Optimizing Personal Voice Activity Detection for On-Device Speech Recognition},
  author={Ding, Shaojin and Rikhye, Rajeev and Liang, Qiao and He, Yanzhang and Wang, Quan and Narayanan, Arun and O'Malley, Tom and McGraw, Ian},
  booktitle={Proc. INTERSPEECH},
  pages={3744--3748},
  year={2022}
}
```
