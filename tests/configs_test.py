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
"""Tests for personal_vad.configs."""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from personal_vad import configs  # noqa: E402


class ConfigsTest(unittest.TestCase):

  def test_frontend_config_pvad_v1(self):
    cfg = configs.FrontendConfig.pvad_v1()
    self.assertEqual(cfg.sample_rate, 16000)
    self.assertEqual(cfg.num_mel_bins, 40)
    self.assertEqual(cfg.output_dim, 40)
    self.assertAlmostEqual(cfg.output_frame_step_ms, 10.0)

  def test_frontend_config_pvad_v2(self):
    cfg = configs.FrontendConfig.pvad_v2()
    self.assertEqual(cfg.sample_rate, 16000)
    self.assertEqual(cfg.num_mel_bins, 128)
    self.assertEqual(cfg.stack_left_context, 3)
    self.assertEqual(cfg.frame_stride, 3)
    self.assertEqual(cfg.output_dim, 512)
    self.assertAlmostEqual(cfg.output_frame_step_ms, 30.0)

  def test_model_config_factories(self):
    v1_et = configs.ModelConfig.pvad_v1(configs.ConditioningMode.ET)
    self.assertEqual(v1_et.backbone, configs.BackboneType.LSTM_V1)
    self.assertEqual(v1_et.num_classes, 3)
    self.assertEqual(v1_et.feature_dim, 40)

    v1_sc = configs.ModelConfig.standard_vad_v1()
    self.assertEqual(v1_sc.conditioning_mode, configs.ConditioningMode.SC)
    self.assertEqual(v1_sc.num_classes, 2)

    v2_conf = configs.ModelConfig.pvad_v2_conformer(
        configs.ConditioningMode.FILM_DVECTOR_COS
    )
    self.assertEqual(v2_conf.backbone, configs.BackboneType.CONFORMER)
    self.assertEqual(v2_conf.conformer_num_layers, 4)
    self.assertEqual(v2_conf.conformer_dim, 64)

    v2_lstm = configs.ModelConfig.pvad_v2_lstm()
    self.assertEqual(v2_lstm.backbone, configs.BackboneType.LSTM_V2)
    self.assertEqual(v2_lstm.lstm_num_layers, 3)
    self.assertEqual(v2_lstm.lstm_units, 256)
    self.assertTrue(v2_lstm.use_layer_norm)


if __name__ == "__main__":
  unittest.main()
