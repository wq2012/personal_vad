"""CLI script for end-to-end Personal VAD inference on audio files.

Loads a local checkpoint directory or a Hugging Face Hub model repository ID,
runs frame-level Personal VAD inference on a 16 kHz audio file (with optional
enrollment audio for the target speaker), and writes the frame posteriors JSON
and optional target-speaker-filtered `.wav` file.
"""

import argparse
from collections.abc import Sequence
import json
import os
import sys
from typing import Any, Optional
import soundfile as sf

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import inference  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Run Personal VAD inference on a 16 kHz audio file."
  )
  parser.add_argument(
      "--model",
      type=str,
      required=True,
      help=(
          "Local checkpoint directory or Hugging Face repo ID "
          "(e.g., 'wq2012/personal-vad-v2-conformer-unified')."
      ),
  )
  parser.add_argument(
      "--audio_path",
      type=str,
      required=True,
      help="Path to input 16 kHz audio file (.wav or .flac).",
  )
  parser.add_argument(
      "--enrollment_path",
      type=str,
      default=None,
      help="Optional path to target speaker enrollment audio (.wav or .flac).",
  )
  parser.add_argument(
      "--use_tflite",
      action="store_true",
      help="Run inference using the 8-bit quantized .tflite flatbuffer.",
  )
  parser.add_argument(
      "--threshold",
      type=float,
      default=0.5,
      help="Posterior threshold for target-speaker speech gating.",
  )
  parser.add_argument(
      "--output_wav",
      type=str,
      default=None,
      help="Optional path to save target-speaker-filtered audio (.wav).",
  )
  parser.add_argument(
      "--output_json",
      type=str,
      default=None,
      help="Optional path to save frame posteriors and summary JSON.",
  )
  return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
  args = parse_args(argv)
  engine = inference.load_pretrained(args.model)
  result = engine.predict_audio(
      audio=args.audio_path,
      enrollment_audio_or_embedding=args.enrollment_path,
      use_tflite=args.use_tflite,
      threshold=args.threshold,
  )

  if args.output_wav is not None:
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_wav)), exist_ok=True
    )
    sf.write(args.output_wav, result["filtered_audio"], 16000)

  payload = {
      "model": args.model,
      "audio_path": args.audio_path,
      "enrollment_path": args.enrollment_path,
      "use_tflite": args.use_tflite,
      "threshold": args.threshold,
      "summary": result["summary"],
      "frame_times_sec": [
          round(float(t), 4) for t in result["frame_times_sec"]
      ],
      "posteriors": [
          [round(float(p), 4) for p in row] for row in result["posteriors"]
      ],
  }
  if args.output_json is not None:
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
    )
    with open(args.output_json, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2)

  print(json.dumps(payload["summary"], indent=2))
  return result


if __name__ == "__main__":
  main()
