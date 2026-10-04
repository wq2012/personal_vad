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
"""Personal VAD: Target-speaker voice activity detection library."""

from . import configs
from . import dataset
from . import eval_lib
from . import frontend
from . import layers
from . import loss
from . import model
from . import online_percentile
from . import tflite_export

__version__ = "0.1.0"

FrameLabel = configs.FrameLabel
ConditioningMode = configs.ConditioningMode
BackboneType = configs.BackboneType
LossType = configs.LossType
FrontendConfig = configs.FrontendConfig
LossConfig = configs.LossConfig
ModelConfig = configs.ModelConfig
DatasetConfig = configs.DatasetConfig

UtteranceData = dataset.UtteranceData
concat_utterance_sequence = dataset.concat_utterance_sequence
concat_utterance_group = dataset.concat_utterance_group
convert_alignment_for_vad_baseline = dataset.convert_alignment_for_vad_baseline
apply_enrollment_less_conditioning = dataset.apply_enrollment_less_conditioning
use_true_speaker_embedding_for_utterance = (
    dataset.use_true_speaker_embedding_for_utterance
)
create_tf_dataset = dataset.create_tf_dataset

EvaluationMetrics = eval_lib.EvaluationMetrics
PersonalVadEvaluator = eval_lib.PersonalVadEvaluator
convert_scores_for_sc_baseline = eval_lib.convert_scores_for_sc_baseline

LogMelFrontend = frontend.LogMelFrontend
concat_meanstd = frontend.concat_meanstd
stack_and_subsample_frames = frontend.stack_and_subsample_frames

FeatureWiseModulationLayer = layers.FeatureWiseModulationLayer
ConformerBlock = layers.ConformerBlock
SpeakerPreNet = layers.SpeakerPreNet

WeightedPairwiseLoss = loss.WeightedPairwiseLoss
CrossEntropyLoss = loss.CrossEntropyLoss
normalize_weights = loss.normalize_weights

PersonalVadModel = model.PersonalVadModel

OnlinePercentileValue = online_percentile.OnlinePercentileValue
rescale_cosine_scores = online_percentile.rescale_cosine_scores

export_to_tflite = tflite_export.export_to_tflite
TFLitePersonalVadRunner = tflite_export.TFLitePersonalVadRunner
