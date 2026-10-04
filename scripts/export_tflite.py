# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLI script to export Personal VAD models to TensorFlow Lite (.tflite).

Supports 8-bit dynamic-range post-training quantization.
"""

import argparse
from collections.abc import Sequence
import os
import sys
from typing import Optional

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import model as model_lib  # noqa: E402
from personal_vad import tflite_export  # noqa: E402
from scripts import train  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Export a Personal VAD model to TFLite format."
  )
  parser.add_argument(
      "--checkpoint_dir",
      type=str,
      default=None,
      help="Optional path to a trained checkpoint directory.",
  )
  parser.add_argument(
      "--backbone",
      type=str,
      choices=["lstm_v1", "lstm_v2", "conformer"],
      default="conformer",
      help="Backbone architecture if --checkpoint_dir is not provided.",
  )
  parser.add_argument(
      "--conditioning_mode",
      type=str,
      default="film_dvector",
      help="Conditioning mode if --checkpoint_dir is not provided.",
  )
  parser.add_argument(
      "--output_tflite",
      type=str,
      required=True,
      help="Output path for the exported .tflite file.",
  )
  parser.add_argument(
      "--sequence_length",
      type=int,
      default=32,
      help="Number of frames per inference chunk in the TFLite graph.",
  )
  parser.add_argument(
      "--no_quantize",
      action="store_true",
      help="Disable 8-bit dynamic range post-training quantization.",
  )
  return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> bytes:
  args = parse_args(argv)
  if args.checkpoint_dir is not None:
    pvad_model = train.load_checkpoint(args.checkpoint_dir)
  else:
    cfg = train.build_model_config(
        backbone=args.backbone,
        conditioning_mode=args.conditioning_mode,
    )
    pvad_model = model_lib.PersonalVadModel(config=cfg)

  return tflite_export.export_to_tflite(
      pvad_model=pvad_model,
      output_path=args.output_tflite,
      quantize=not args.no_quantize,
      sequence_length=args.sequence_length,
  )


if __name__ == "__main__":
  main()
