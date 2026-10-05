"""Personal VAD: Target-speaker voice activity detection library."""

from . import configs
from . import dataset
from . import eval_lib
from . import online_percentile

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

OnlinePercentileValue = online_percentile.OnlinePercentileValue
rescale_cosine_scores = online_percentile.rescale_cosine_scores

try:
  from . import frontend
  from . import inference
  from . import layers
  from . import loss
  from . import model
  from . import tflite_export

  LogMelFrontend = frontend.LogMelFrontend
  concat_meanstd = frontend.concat_meanstd
  stack_and_subsample_frames = frontend.stack_and_subsample_frames

  PersonalVadInferenceEngine = inference.PersonalVadInferenceEngine
  load_pretrained = inference.load_pretrained

  FeatureWiseModulationLayer = layers.FeatureWiseModulationLayer
  ConformerBlock = layers.ConformerBlock
  SpeakerPreNet = layers.SpeakerPreNet

  WeightedPairwiseLoss = loss.WeightedPairwiseLoss
  CrossEntropyLoss = loss.CrossEntropyLoss
  normalize_weights = loss.normalize_weights

  PersonalVadModel = model.PersonalVadModel

  export_to_tflite = tflite_export.export_to_tflite
  TFLitePersonalVadRunner = tflite_export.TFLitePersonalVadRunner
except ImportError:  # pragma: no cover
  pass
