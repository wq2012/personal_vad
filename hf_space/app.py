"""Interactive Gradio application for Personal VAD 1.0 and 2.0.

Provides side-by-side comparison of Standard VAD (`B2` Conformer), Personal VAD
1.0 (`ET + WPL` LSTM), and Personal VAD 2.0 (`E3` / `E5` Conformer + FiLM +
Speaker Pre-Net) on predefined multilingual multi-speaker conversations and
custom uploaded audio, including playable target-speaker-filtered audio and
enrollment-less fallback.
"""

import json
import os
import sys
import tempfile
from typing import Any, Optional
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import personal_vad  # noqa: E402

MODEL_REPOS = {
    "std_vad": (
        os.path.join(
            REPO_ROOT, "pretrained_models", "pvad2_b2_conformer_std_vad"
        ),
        "wq2012/personal-vad-v2-conformer-std-vad",
    ),
    "pvad1_et_wpl": (
        os.path.join(REPO_ROOT, "pretrained_models", "pvad1_et_wpl"),
        "wq2012/personal-vad-v1-et-wpl",
    ),
    "pvad2_e3": (
        os.path.join(
            REPO_ROOT,
            "pretrained_models",
            "pvad2_e3_conformer_film_dvector_cos",
        ),
        "wq2012/personal-vad-v2-conformer-film-dvector-cos",
    ),
    "pvad2_e5": (
        os.path.join(
            REPO_ROOT, "pretrained_models", "pvad2_e5_conformer_unified"
        ),
        "wq2012/personal-vad-v2-conformer-unified",
    ),
}

_ENGINES: dict[str, personal_vad.PersonalVadInferenceEngine] = {}


def _get_engine(key: str) -> personal_vad.PersonalVadInferenceEngine:
  """Lazily loads a pretrained inference engine from local disk or HF Hub."""
  if key not in _ENGINES:
    local_dir, hf_id = MODEL_REPOS[key]
    source = local_dir if os.path.isdir(local_dir) else hf_id
    _ENGINES[key] = personal_vad.load_pretrained(source)
  return _ENGINES[key]


def _render_comparison_figure(
    wav: np.ndarray,
    sr: int,
    std_res: dict[str, Any],
    v1_res: dict[str, Any],
    v2_res: dict[str, Any],
    threshold: float,
    enrollment_less: bool,
) -> str:
  """Renders a 4-panel waveform + posterior comparison plot."""
  fig, axes = plt.subplots(4, 1, figsize=(11, 8.2), sharex=True)
  t_wav = np.arange(len(wav), dtype=np.float32) / float(sr)

  axes[0].plot(t_wav, wav, color="#334155", linewidth=0.7, alpha=0.85)
  axes[0].set_ylabel("Amplitude")
  axes[0].set_title(
      "1. Input Multi-Speaker Waveform (16 kHz) & Target Speaker Gating Mask",
      fontsize=10,
      fontweight="bold",
  )
  t2 = v2_res["frame_times_sec"]
  p_tss_v2 = v2_res["posteriors"][:, 0]
  axes[0].fill_between(
      t2,
      -1.0,
      1.0,
      where=p_tss_v2 >= threshold,
      color="#22c55e",
      alpha=0.18,
      label=f"Target Speech Active (p >= {threshold:.2f})",
  )
  axes[0].set_ylim(-1.05, 1.05)
  axes[0].legend(loc="upper right", fontsize=8)
  axes[0].grid(True, alpha=0.25)

  t_std = std_res["frame_times_sec"]
  p_std = std_res["posteriors"]
  axes[1].plot(
      t_std, p_std[:, 0], label="Speech (all speakers)", color="#2563eb", lw=1.8
  )
  axes[1].plot(
      t_std, p_std[:, 1], label="Silence (ns)", color="#94a3b8", lw=1.3, ls="--"
  )
  axes[1].set_ylabel("Posterior")
  axes[1].set_ylim(-0.05, 1.05)
  axes[1].set_title(
      "2. Standard VAD Baseline (B2 Conformer) — Triggers on All Speakers",
      fontsize=10,
      fontweight="bold",
  )
  axes[1].legend(loc="upper right", fontsize=8)
  axes[1].grid(True, alpha=0.25)

  t1 = v1_res["frame_times_sec"]
  p1 = v1_res["posteriors"]
  axes[2].plot(
      t1, p1[:, 0], label="Target Speech (tss)", color="#16a34a", lw=1.9
  )
  axes[2].plot(
      t1, p1[:, 1], label="Silence (ns)", color="#94a3b8", lw=1.3, ls="--"
  )
  if p1.shape[1] > 2:
    axes[2].plot(
        t1,
        p1[:, 2],
        label="Non-Target Speech (ntss)",
        color="#dc2626",
        lw=1.6,
    )
  axes[2].axhline(threshold, color="#15803d", ls=":", lw=1.0)
  axes[2].set_ylabel("Posterior")
  axes[2].set_ylim(-0.05, 1.05)
  axes[2].set_title(
      "3. Personal VAD 1.0 (ET + Weighted Pairwise Loss, 2-Layer LSTM, 130K)",
      fontsize=10,
      fontweight="bold",
  )
  axes[2].legend(loc="upper right", fontsize=8)
  axes[2].grid(True, alpha=0.25)

  p2 = v2_res["posteriors"]
  axes[3].plot(
      t2, p2[:, 0], label="Target Speech (tss)", color="#16a34a", lw=2.0
  )
  axes[3].plot(
      t2, p2[:, 1], label="Silence (ns)", color="#94a3b8", lw=1.3, ls="--"
  )
  if p2.shape[1] > 2:
    axes[3].plot(
        t2,
        p2[:, 2],
        label="Non-Target Speech (ntss)",
        color="#dc2626",
        lw=1.6,
    )
  axes[3].axhline(threshold, color="#15803d", ls=":", lw=1.0)
  axes[3].set_ylabel("Posterior")
  axes[3].set_xlabel("Time (seconds)")
  axes[3].set_ylim(-0.05, 1.05)
  mode_tag = (
      "Enrollment-Less Fallback (e_target = 0)"
      if enrollment_less
      else "Enrolled Target Speaker Conditioning"
  )
  axes[3].set_title(
      f"4. Personal VAD 2.0 (Conformer + FiLM + Speaker Pre-Net) — {mode_tag}",
      fontsize=10,
      fontweight="bold",
  )
  axes[3].legend(loc="upper right", fontsize=8)
  axes[3].grid(True, alpha=0.25)

  fig.tight_layout()
  tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
  fig.savefig(tmp_png.name, dpi=150)
  plt.close(fig)
  return tmp_png.name


def run_demo_inference(
    conversation_audio_path: Optional[str],
    enrollment_audio_path: Optional[str],
    enrollment_less: bool = False,
    use_tflite: bool = False,
    threshold: float = 0.5,
) -> tuple[Optional[str], Optional[str], str]:
  """Runs Standard VAD, PVAD 1.0, and PVAD 2.0 on the input audio."""
  if not conversation_audio_path or not os.path.exists(conversation_audio_path):
    return (
        None,
        None,
        json.dumps({"error": "Please select or upload an audio clip."}),
    )

  wav, sr = sf.read(conversation_audio_path, dtype="float32")
  if wav.ndim > 1:
    wav = np.mean(wav, axis=-1)

  enroll_arg = (
      None
      if (enrollment_less or not enrollment_audio_path)
      else enrollment_audio_path
  )

  std_engine = _get_engine("std_vad")
  v1_engine = _get_engine("pvad1_et_wpl")
  v2_engine = _get_engine("pvad2_e5" if enrollment_less else "pvad2_e3")

  std_res = std_engine.predict_audio(
      conversation_audio_path,
      enrollment_audio_or_embedding=None,
      use_tflite=use_tflite,
      threshold=threshold,
  )
  v1_res = v1_engine.predict_audio(
      conversation_audio_path,
      enrollment_audio_or_embedding=enroll_arg,
      use_tflite=use_tflite,
      threshold=threshold,
  )
  v2_res = v2_engine.predict_audio(
      conversation_audio_path,
      enrollment_audio_or_embedding=enroll_arg,
      use_tflite=use_tflite,
      threshold=threshold,
  )

  plot_path = _render_comparison_figure(
      wav=wav,
      sr=sr,
      std_res=std_res,
      v1_res=v1_res,
      v2_res=v2_res,
      threshold=threshold,
      enrollment_less=enrollment_less or (enroll_arg is None),
  )

  out_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
  sf.write(out_wav_file.name, v2_res["filtered_audio"], sr)

  report = {
      "mode": (
          "enrollment_less"
          if (enrollment_less or enroll_arg is None)
          else "enrolled_target_speaker"
      ),
      "runtime": "TFLite 8-bit Quantized" if use_tflite else "TensorFlow FP32",
      "threshold": threshold,
      "standard_vad_b2": std_res["summary"],
      "personal_vad_v1_et_wpl": v1_res["summary"],
      "personal_vad_v2_conformer": v2_res["summary"],
  }
  return plot_path, out_wav_file.name, json.dumps(report, indent=2)


def build_gradio_interface() -> gr.Blocks:
  """Constructs the Gradio Blocks UI."""
  examples_dir = os.path.join(os.path.dirname(__file__), "examples")
  preset_rows = []
  for idx in range(1, 4):
    conv_p = os.path.join(examples_dir, f"conversation_{idx}.wav")
    enr_p = os.path.join(examples_dir, f"enroll_{idx}.wav")
    if os.path.exists(conv_p) and os.path.exists(enr_p):
      preset_rows.append([conv_p, enr_p, False, False, 0.5])

  with gr.Blocks(
      title="Personal VAD (1.0 & 2.0) Interactive Demo"
  ) as demo:
    gr.Markdown(
        "# Personal VAD (1.0 & 2.0): Speaker-Conditioned VAD\n"
        "**This is not an officially supported Google product.**\n\n"
        "> **Open-Source Reproduction Notice:** This interactive demo "
        "showcases pretrained models from our open-source reproduction of "
        "**Personal VAD 1.0** ([arXiv:1908.04284]"
        "(https://arxiv.org/abs/1908.04284)) and **Personal VAD 2.0** "
        "([arXiv:2204.03793](https://arxiv.org/abs/2204.03793)) trained on "
        "the 8-language Multilingual LibriSpeech (MLS) benchmark. Instead of "
        "the original papers' proprietary 4.88M-parameter 3-layer GE2E LSTM "
        "speaker encoder, enrollment 256-D d-vectors are extracted using a "
        "self-contained open-set regularized LDA+PCA speaker subspace "
        "extractor (`speaker_subspace.npz`) trained from scratch on 98 MLS "
        "training speakers (`0.9521` ROC-AUC on 35 unseen test speakers)."
    )
    with gr.Row():
      with gr.Column(scale=1):
        conv_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Multi-Speaker Conversation Audio (16 kHz)",
        )
        enroll_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Target Speaker Enrollment Audio (16 kHz)",
        )
        enroll_less_chk = gr.Checkbox(
            value=False,
            label=(
                "Enrollment-Less Mode (set e_target = 0 for PVAD 2.0 "
                "Unified E5)"
            ),
        )
        tflite_chk = gr.Checkbox(
            value=False,
            label="Use 8-bit Quantized TFLite Flatbuffer (.tflite)",
        )
        thresh_slider = gr.Slider(
            minimum=0.1,
            maximum=0.9,
            value=0.5,
            step=0.05,
            label="Target Speaker Gate Threshold",
        )
        run_btn = gr.Button("Run Personal VAD Comparison", variant="primary")
      with gr.Column(scale=2):
        plot_output = gr.Image(
            type="filepath",
            label=(
                "Frame-Level Posteriors: Standard VAD vs. PVAD 1.0 vs. "
                "PVAD 2.0"
            ),
        )
        filtered_audio_output = gr.Audio(
            type="filepath",
            label=(
                "Target-Speaker-Filtered Output Audio (Non-Target Speech "
                "Suppressed)"
            ),
        )
        json_output = gr.Code(
            language="json",
            label="Frame Posterior Summary Metrics (JSON)",
        )

    run_btn.click(
        fn=run_demo_inference,
        inputs=[
            conv_input,
            enroll_input,
            enroll_less_chk,
            tflite_chk,
            thresh_slider,
        ],
        outputs=[plot_output, filtered_audio_output, json_output],
    )

    if preset_rows:
      gr.Examples(
          examples=preset_rows,
          inputs=[
              conv_input,
              enroll_input,
              enroll_less_chk,
              tflite_chk,
              thresh_slider,
          ],
          outputs=[plot_output, filtered_audio_output, json_output],
          fn=run_demo_inference,
          label="Predefined Multilingual Multi-Speaker Conversation Examples",
      )
  return demo


if __name__ == "__main__":
  app = build_gradio_interface()
  app.launch()
