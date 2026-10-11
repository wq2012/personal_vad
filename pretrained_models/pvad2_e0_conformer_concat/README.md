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

# Personal VAD 2.0 — E0 Causal Conformer Concat Personal VAD (`wq2012/personal-vad-v2-conformer-concat`)

**This is not an officially supported Google product.**

> **Open-Source Reproduction Notice:** This model is part of an independent open-source reproduction of the **Personal VAD 1.0** ([Ding et al., Odyssey 2020](https://arxiv.org/abs/1908.04284)) and **Personal VAD 2.0** ([Ding et al., INTERSPEECH 2022](https://arxiv.org/abs/2204.03793)) papers. Because the original papers were developed using internal datasets and infrastructure, this model was trained from scratch using the open-source [`personal-vad`](https://github.com/wq2012/personal_vad) library on our 8-language **Multilingual LibriSpeech (MLS)** benchmark (`de`, `en`, `es`, `fr`, `it`, `nl`, `pl`, `pt`, combining the 7 non-English `facebook/multilingual_librispeech` languages with OpenSLR 12 English LibriSpeech).

## Model Summary

- **Model ID**: `pvad2_e0_conformer_concat` (`wq2012/personal-vad-v2-conformer-concat`)
- **Paper**: [Personal VAD 2.0 (arXiv:2204.03793)](https://arxiv.org/abs/2204.03793)
- **Architecture**: 4-Layer Causal Streaming Conformer (64-d) with d-vector input concatenation
- **Backbone**: `conformer`
- **Conditioning Mode**: `concat`
- **Loss Function**: `cross_entropy`
- **Trainable Parameters**: `699,907`
- **Safetensors Checkpoint (`model.safetensors`)**: `2744.36 KB`
- **FP32 TFLite Flatbuffer (`model_fp32.tflite`)**: `2820.20 KB`
- **8-Bit Quantized TFLite (`model_quantized.tflite`)**: `880.96 KB`
- **Primary Concatenated Test Set micro-mAP**: **`0.7865`** (Overall frame accuracy: `0.6796`)

## Included Files

- `model.safetensors`: Safetensors weights for secure, fast loading.
- `model.weights.h5`: Keras HDF5 checkpoint weights.
- `model_config.json`: Serialized `personal_vad.ModelConfig` spec.
- `model_fp32.tflite`: Unquantized 32-bit float stateful streaming TFLite flatbuffer (`state_in` -> `(probs, state_out)`).
- `model_quantized.tflite`: 8-bit dynamic-range quantized stateful streaming TFLite flatbuffer for on-device deployment.
- `speaker_subspace.npz`: Open-set regularized LDA+WCCN+PCA speaker subspace projection matrix (`256`-d L2-normalized d-vectors, full rank `256`, trained from scratch on the 98 MLS training speakers).
- `evaluation_metrics.json`: Full test-set evaluation metrics on 40 unseen multilingual speakers (with utterance-cluster 95% bootstrap CIs).

## Speaker Embedding (`d-vector`) Extractor Design & Transparency

In the original Personal VAD 1.0 and 2.0 papers, speaker embeddings (`e_target`) and frame-level cosine scores (`s_t`) were extracted using Google's internal 3-layer LSTM speaker verification network (`4.88M` parameters) trained with Generalized End-to-End (`GE2E`) loss on proprietary vendor-collected corpora.

To keep this open-source reproduction completely self-contained in pure TensorFlow/NumPy with zero external PyTorch/SpeechBrain/JAX dependencies, **we trained our own lightweight, non-neural open-set 256-D speaker subspace extractor (`OpenSetSpeakerSubspace`, serialized in `speaker_subspace.npz`, `472.8 KB`) from scratch on the 98 training speakers (`1,159` utterances) of the 8-language Multilingual LibriSpeech (MLS) benchmark** (`16` unseen development speakers / `192` utterances; `40` unseen test speakers / `480` utterances):
1. **360-D Causal Multi-Resolution Acoustic Summary**: Over voiced speech frames (or a strictly causal 14-frame / 140 ms sliding window for frame-level scores `s_t`), it extracts 40-D log-Mel `[mean, std, pos_dev, neg_dev, delta_rms]` (200-D) concatenated with 80-D log-Mel `[mean, std]` (160-D), while the Personal VAD network's acoustic input `x_t` remains 100% pure log-Mel filterbank energies.
2. **256-D Regularized LDA + WCCN + PCA Projection**: Applies Z-score standardization, projects onto a full-rank 256-D discriminative subspace fitted via regularized Linear Discriminant Analysis (LDA), Within-Class Covariance Normalization (WCCN), and orthogonal PCA completion on the 98 training speakers, adds a fixed positive-cone offset on coordinate 255, and L2-normalizes the resulting vector (`||e||_2 = 1`).
3. **Unseen-Speaker Verification Accuracy**: On the **16 unseen development speakers** (`192` utterances), `speaker_subspace.npz` achieves an open-set speaker verification **ROC-AUC of `0.9579`**, and on the **40 held-out unseen test speakers** (`480` utterances across 8 languages), it achieves a **ROC-AUC of `0.9449`** (mean positive cosine similarity `0.5424` vs. negative `-0.0166`). Users may also pass any external 256-D L2-normalized neural d-vector directly to `PersonalVadModel` or `PersonalVadInferenceEngine`.

## Quickstart Usage

### 1. High-Level Audio Inference (`personal_vad.load_pretrained`)

```python
import personal_vad

engine = personal_vad.load_pretrained("wq2012/personal-vad-v2-conformer-concat")
result = engine.infer_from_files(
    audio_path="conversation_16k.wav",
    enroll_path="target_speaker_enroll_16k.wav",
    use_tflite=False,
    threshold=0.5,
)
print("Summary:", result["summary"])
print("Probabilities shape:", result["probabilities"].shape)
```

### 2. On-Device Stateful 8-Bit Quantized `.tflite` Execution

```python
from huggingface_hub import hf_hub_download
import numpy as np
import personal_vad

tflite_path = hf_hub_download(
    repo_id="wq2012/personal-vad-v2-conformer-concat", filename="model_quantized.tflite"
)
runner = personal_vad.TFLitePersonalVadRunner(tflite_path)
features = np.zeros((32, 512), dtype=np.float32)
speaker_embedding = np.ones((256,), dtype=np.float32) / np.sqrt(256.0)
cosine_score = np.full((32,), 0.8, dtype=np.float32)
probs = runner.predict(
    features=features,
    speaker_embedding=speaker_embedding,
    cosine_score=cosine_score,
    chunk_size=16,
)
print("TFLite output shape:", probs.shape)
```

## Open-Source Multilingual LibriSpeech Benchmark Results

All models were trained on 98 training speakers (`1,159` utterances) across 8 languages (`de`, `en`, `es`, `fr`, `it`, `nl`, `pl`, `pt`, combining the 7 non-English `facebook/multilingual_librispeech` languages with OpenSLR 12 English LibriSpeech), validated on 16 unseen dev speakers (`192` utterances), and evaluated on **40 held-out unseen test speakers** (`480` utterances: `120` enrollment + `360` evaluation utterances, zero speaker overlap).

### Personal VAD 1.0 Models (Concatenated Multi-Speaker Test Set)

| Model ID | Mode | Loss | Target Speech AP (`tss`) | Silence AP (`ns`) | Non-Target Speech AP (`ntss`) | Clean Micro mAP (95% CI) | MTR Micro mAP | INT8 TFLite mAP | Quantized Size |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pvad1_sc_baseline` | `SC` | `cross_entropy` | 0.7974 | 0.9072 | 0.8303 | **0.8422** ([0.8279, 0.8559]) | 0.6303 | 0.8422 | 199.4 KB |
| `pvad1_st_ce` | `ST` | `cross_entropy` | 0.8549 | 0.9144 | 0.8804 | **0.8826** ([0.8695, 0.8954]) | 0.6883 | 0.8827 | 200.4 KB |
| `pvad1_et_ce` | `ET` | `cross_entropy` | 0.8253 | 0.8996 | 0.8322 | **0.8452** ([0.8307, 0.8592]) | 0.6475 | 0.8418 | 265.4 KB |
| `pvad1_set_ce` | `SET` | `cross_entropy` | 0.8708 | 0.9098 | 0.8814 | **0.8858** ([0.8712, 0.9008]) | 0.7072 | 0.8844 | 265.7 KB |
| `pvad1_et_wpl` | `ET` | `weighted_pairwise` | 0.8215 | 0.8947 | 0.8358 | **0.8463** ([0.8321, 0.8604]) | 0.6438 | 0.8464 | 265.4 KB |

### Personal VAD 2.0 Models (Non-Concat & Concat Test Sets)

| Model ID | Backbone | Conditioning | Non-Concat mAP | Concat `tss` AP | Concat `ns` AP | Concat `ntss` AP | Clean Concat mAP (95% CI) | MTR Concat mAP | INT8 TFLite mAP | Quantized Size |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pvad2_b0_lstm_concat` | `lstm_v2` | `concat` | 0.6911 | 0.4976 | 0.9227 | 0.5322 | **0.7037** ([0.6904, 0.7161]) | 0.5420 | 0.7036 | 2012.2 KB |
| **`pvad2_e0_conformer_concat`** | `conformer` | `concat` | 0.7640 | 0.5960 | 0.9593 | 0.6876 | **0.7865** ([0.7706, 0.8025]) | 0.6866 | 0.7857 | 881.0 KB |
| `pvad2_e1_conformer_film_dvector` | `conformer` | `film_dvector` | 0.8248 | 0.7816 | 0.9359 | 0.8206 | **0.8436** ([0.8267, 0.8590]) | 0.7200 | 0.8434 | 899.4 KB |
| `pvad2_e2_conformer_film_cos` | `conformer` | `film_cos` | 0.8742 | 0.8423 | 0.9430 | 0.8695 | **0.8788** ([0.8669, 0.8912]) | 0.7067 | 0.8784 | 1333.4 KB |
| `pvad2_e3_conformer_film_dvector_cos` | `conformer` | `film_dvector_cos` | 0.8673 | 0.8360 | 0.9454 | 0.8688 | **0.8830** ([0.8706, 0.8946]) | 0.7222 | 0.8831 | 1367.1 KB |
| `pvad2_e4_conformer_int8` | `conformer` | `film_dvector_cos` | 0.8674 | 0.8363 | 0.9455 | 0.8689 | **0.8831** ([0.8708, 0.8947]) | 0.7221 | 0.8831 | 1367.1 KB |
| `pvad2_e5_conformer_unified` | `conformer` | `film_dvector_cos` | 0.7919 | 0.6773 | 0.9265 | 0.7327 | **0.7850** ([0.7639, 0.8038]) | 0.6203 | 0.7849 | 1367.1 KB |

### Per-Language Breakdown for `pvad2_e0_conformer_concat` (8 Languages)

| Language | Unseen Test Speakers | Eval Mixtures | Frames | Target Speech AP (`tss`) | Silence AP (`ns`) | Non-Target Speech AP (`ntss`) | Micro mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `de` | 5 | 42 | 5,040 | 0.6153 | 0.8965 | 0.6608 | **0.7421** |
| `en` | 6 | 54 | 6,048 | 0.6656 | 0.9648 | 0.7038 | **0.8009** |
| `es` | 5 | 42 | 5,040 | 0.5665 | 0.9677 | 0.6495 | **0.7728** |
| `fr` | 5 | 42 | 5,040 | 0.5511 | 0.9620 | 0.6358 | **0.7800** |
| `it` | 5 | 42 | 5,040 | 0.7173 | 0.9804 | 0.8241 | **0.8678** |
| `nl` | 5 | 42 | 5,040 | 0.5946 | 0.9698 | 0.6443 | **0.7976** |
| `pl` | 4 | 37 | 4,032 | 0.6346 | 0.9516 | 0.6632 | **0.7726** |
| `pt` | 5 | 42 | 5,040 | 0.5600 | 0.9733 | 0.5806 | **0.7367** |

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
