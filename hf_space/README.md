---
title: Personal VAD (1.0 & 2.0) Interactive Demo
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Speaker-conditioned voice activity detection (Personal VAD)
---

# Personal VAD (1.0 & 2.0): Speaker-Conditioned Voice Activity Detection

**This is not an officially supported Google product.**

> **Open-Source Reproduction Notice:** This interactive Space and the accompanying [`personal-vad` GitHub repository](https://github.com/wq2012/personal_vad) provide an independent open-source reproduction of the **Personal VAD 1.0** ([Ding et al., Odyssey 2020](https://arxiv.org/abs/1908.04284)) and **Personal VAD 2.0** ([Ding et al., INTERSPEECH 2022](https://arxiv.org/abs/2204.03793)) papers. Because the original studies were conducted using proprietary internal training corpora and infrastructure, this project re-implements the complete pipeline from scratch using standard open-source Python/TensorFlow components and trains all models on the public 8-language **Multilingual LibriSpeech (MLS)** corpus.

## Interactive Features

1. **Dual Input Modes**:
   - **Predefined Multilingual Multi-Speaker Examples**: Select from curated 2–3 speaker conversational audio clips across English, German, French, and Spanish with enrolled target speaker reference audio.
   - **Custom Audio Upload & Enrollment-Less Mode**: Upload your own target speaker enrollment clip and multi-speaker test recording, or toggle **Enrollment-Less Mode** (`e_target = 0`) to see how **Personal VAD 2.0 Unified (`E5`)** seamlessly falls back to standard voice activity detection.
2. **Side-by-Side Architecture Comparison**:
   - **Standard VAD (`B2` Conformer)**: Detects all speech (`speech` vs. `silence`) regardless of who is speaking.
   - **Personal VAD 1.0 (`ET + WPL`)**: 2-layer LSTM (`64` units, `130K` parameters) conditioned on the enrolled speaker d-vector and trained with Weighted Pairwise Loss (`w_<ns,ntss> = 0.1`).
   - **Personal VAD 2.0 (`E3` / `E5` Conformer + FiLM + Speaker Pre-Net)**: 4-layer causal streaming Conformer (`64`-d, `8` heads, `31` left context) with Speaker Pre-Net cosine similarity and Feature-wise Linear Modulation (FiLM).
3. **Target-Speaker-Filtered Audio Playback**:
   - Listen to the original multi-speaker recording alongside the gated output audio where non-target speakers (`ntss`) and background silence (`ns`) are suppressed so only the target speaker (`tss`) passes through.

## Running the Full Gradio App Locally

This repository includes both a browser-native interactive application (`index.html`) and a full Python Gradio application (`app.py`):

```bash
pip install -r requirements.txt
python app.py
```

## Links & Resources

- **GitHub Repository**: [https://github.com/wq2012/personal_vad](https://github.com/wq2012/personal_vad)
- **Personal VAD 1.0 Paper**: [arXiv:1908.04284](https://arxiv.org/abs/1908.04284)
- **Personal VAD 2.0 Paper**: [arXiv:2204.03793](https://arxiv.org/abs/2204.03793)
- **Pretrained Models on Hugging Face**: [https://huggingface.co/wq2012](https://huggingface.co/wq2012)
