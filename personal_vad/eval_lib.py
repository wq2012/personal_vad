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
"""Evaluation library for Personal VAD and Standard VAD models.

Computes per-class Accuracy, overall Accuracy, per-class Average Precision (AP),
micro-averaged mean Average Precision (mAP), Precision-Recall curves, ROC
curves, AUC, Score Combination (`SC`) baseline conversion, and HTML/JSON report
generation.
"""

import dataclasses
import os
from typing import Any, Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import metrics  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from . import online_percentile  # noqa: E402


PERSONAL_VAD_CLASS_NAMES = (
    "Speech (tss)",
    "Silence (ns)",
    "Non-Target Speech (ntss)",
)
STANDARD_VAD_CLASS_NAMES = (
    "Speech",
    "Silence",
)


@dataclasses.dataclass
class EvaluationMetrics:
  """Container for computed evaluation metrics.

  All metric fields default to `None` until computed from actual predictions.
  """

  eval_mode: Optional[str] = None
  num_frames: Optional[int] = None
  overall_accuracy: Optional[float] = None
  class_accuracies: Optional[dict[str, float]] = None
  average_precisions: Optional[dict[str, float]] = None
  mean_average_precision_micro: Optional[float] = None
  roc_aucs: Optional[dict[str, float]] = None

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary of computed metrics."""
    return dataclasses.asdict(self)


def convert_scores_for_sc_baseline(
    scores: np.ndarray,
    bins: int = 200,
    hist_range: tuple[float, float] = (-1.0, 1.0),
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> np.ndarray:
  """Converts 2-class VAD scores + cosine similarity into 3-class PVAD scores.

  Implements Equations (1)-(3) in Personal VAD 1.0 (arXiv:1908.04284):
    - `z(tss) = s_t * z(speech)`
    - `z(ns) = z(silence)`
    - `z(ntss) = (1 - s_t) * z(speech)`

  Args:
    scores: Float NumPy array of shape `[N, 3]` where:
      - `scores[:, 0]` is the standard VAD speech score `z(speech)`.
      - `scores[:, 1]` is the standard VAD silence score `z(silence)`.
      - `scores[:, 2]` is the raw frame-level cosine similarity in `[-1, 1]`.
    bins: Number of histogram bins for online percentile estimation (200).
    hist_range: Range of cosine similarity histogram `(-1.0, 1.0)`.
    low_percentile: Lower percentile mapped to 0.0 (5.0).
    high_percentile: Upper percentile mapped to 1.0 (95.0).

  Returns:
    Float NumPy array of shape `[N, 3]` containing `[z(tss), z(ns), z(ntss)]`.
  """
  scores = np.asarray(scores, dtype=np.float32)
  if scores.ndim != 2 or scores.shape[1] != 3:
    raise ValueError(
        f"Expected scores of shape [N, 3] for SC baseline, got {scores.shape}"
    )
  rescaled_sim = online_percentile.rescale_cosine_scores(
      scores[:, 2],
      bins=bins,
      hist_range=hist_range,
      low_percentile=low_percentile,
      high_percentile=high_percentile,
  )
  converted = np.zeros_like(scores, dtype=np.float32)
  converted[:, 0] = scores[:, 0] * rescaled_sim
  converted[:, 1] = scores[:, 1]
  converted[:, 2] = scores[:, 0] * (1.0 - rescaled_sim)
  return converted


class PersonalVadEvaluator:
  """Evaluates Personal VAD and Standard VAD frame-level predictions.

  Supports three evaluation modes:
    - `"personal_vad"`: 3-class Personal VAD (`tss=0`, `ns=1`, `ntss=2`).
    - `"personal_vad_std_eval"`: Evaluates a 3-class Personal VAD model on
      standard 2-class VAD (`tss=0` vs `ns=1`).
    - `"std_vad_std_eval"`: Evaluates a 2-class Standard VAD model (`speech=0`
      vs `silence=1`).
  """

  def __init__(self, eval_mode: str = "personal_vad"):
    if eval_mode not in (
        "personal_vad",
        "personal_vad_std_eval",
        "std_vad_std_eval",
    ):
      raise ValueError(f"Unsupported eval_mode: {eval_mode}")
    self.eval_mode = eval_mode
    if eval_mode == "personal_vad":
      self.classes = [0, 1, 2]
      self.class_names = list(PERSONAL_VAD_CLASS_NAMES)
    else:
      self.classes = [0, 1]
      self.class_names = list(STANDARD_VAD_CLASS_NAMES)

  def _prepare_scores_and_labels(
      self,
      scores: np.ndarray,
      labels: np.ndarray,
      sc_baseline: bool = False,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Formats scores, integer labels, and one-hot binarized labels."""
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if scores.ndim > 2:
      scores = scores.reshape(-1, scores.shape[-1])

    if sc_baseline:
      scores = convert_scores_for_sc_baseline(scores)

    if self.eval_mode == "personal_vad_std_eval":
      # When evaluating a 3-class Personal VAD model on standard 2-class VAD,
      # map non-target speech (2) to speech (0) in labels and use the first two
      # columns (or combine speech columns if appropriate). Following
      # sklearn_eval_lib.py, we evaluate classes [0, 1] using scores[:, :2] and
      # map label 2 -> 0.
      labels = np.where(labels == 2, 0, labels)
      scores = scores[:, :2]
    elif self.eval_mode == "std_vad_std_eval":
      labels = np.where(labels == 2, 0, labels)
      scores = scores[:, :2]

    if scores.shape[0] != labels.shape[0]:
      raise ValueError(
          f"Number of score frames ({scores.shape[0]}) does not match "
          f"number of label frames ({labels.shape[0]})."
      )
    if scores.shape[0] == 0:
      raise ValueError("Cannot evaluate on empty predictions.")

    binarized_labels = preprocessing.label_binarize(
        labels, classes=[0, 1, 2]
    )
    if self.eval_mode in ("personal_vad_std_eval", "std_vad_std_eval"):
      binarized_labels = binarized_labels[:, :2]

    return scores, labels, binarized_labels

  def compute_accuracies(
      self, scores: np.ndarray, labels: np.ndarray
  ) -> tuple[dict[str, float], float]:
    """Computes per-class accuracy and overall accuracy.

    Args:
      scores: Array of shape `[N, num_classes]`.
      labels: Integer array of shape `[N]`.

    Returns:
      Tuple of `(per_class_accuracy_dict, overall_accuracy)`.
    """
    predicted_labels = np.argmax(scores, axis=1)
    overall_acc = float(metrics.accuracy_score(labels, predicted_labels))
    class_accs: dict[str, float] = {}
    for cls_idx, cls_name in zip(self.classes, self.class_names):
      mask = labels == cls_idx
      if np.any(mask):
        cls_acc = float(
            metrics.accuracy_score(labels[mask], predicted_labels[mask])
        )
      else:
        cls_acc = 0.0
      class_accs[cls_name] = cls_acc
    return class_accs, overall_acc

  def compute_precision_recall(
      self, scores: np.ndarray, binarized_labels: np.ndarray
  ) -> tuple[
      dict[int, np.ndarray],
      dict[int, np.ndarray],
      dict[str, float],
      float,
  ]:
    """Computes Precision-Recall curves, per-class AP, and micro-mAP."""
    precisions: dict[int, np.ndarray] = {}
    recalls: dict[int, np.ndarray] = {}
    average_precisions: dict[str, float] = {}
    for i, cls_name in zip(self.classes, self.class_names):
      precisions[i], recalls[i], _ = metrics.precision_recall_curve(
          binarized_labels[:, i], scores[:, i]
      )
      average_precisions[cls_name] = float(
          metrics.average_precision_score(binarized_labels[:, i], scores[:, i])
      )
    precisions[-1], recalls[-1], _ = metrics.precision_recall_curve(
        binarized_labels.ravel(), scores.ravel()
    )
    mean_ap_micro = float(
        metrics.average_precision_score(
            binarized_labels, scores, average="micro"
        )
    )
    return precisions, recalls, average_precisions, mean_ap_micro

  def compute_roc(
      self, scores: np.ndarray, binarized_labels: np.ndarray
  ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, float]]:
    """Computes ROC curves (FPR, TPR) and per-class ROC AUC."""
    fprs: dict[int, np.ndarray] = {}
    tprs: dict[int, np.ndarray] = {}
    roc_aucs: dict[str, float] = {}
    for i, cls_name in zip(self.classes, self.class_names):
      fprs[i], tprs[i], _ = metrics.roc_curve(
          binarized_labels[:, i], scores[:, i]
      )
      roc_aucs[cls_name] = float(metrics.auc(fprs[i], tprs[i]))
    return fprs, tprs, roc_aucs

  def evaluate(
      self,
      scores: np.ndarray,
      labels: np.ndarray,
      sc_baseline: bool = False,
      output_dir: Optional[str] = None,
  ) -> EvaluationMetrics:
    """Runs full evaluation and optionally saves plots and an HTML report.

    Args:
      scores: Frame-level prediction scores of shape `[N, C]`.
      labels: Frame-level integer ground truth labels of shape `[N]`.
      sc_baseline: Whether to apply SC baseline score combination first.
      output_dir: Optional directory path to save `roc_curves.png`,
        `pr_curves.png`, and `report.html`.

    Returns:
      An `EvaluationMetrics` instance populated with measured values.
    """
    scores_prep, labels_prep, binarized_labels = (
        self._prepare_scores_and_labels(
            scores=scores, labels=labels, sc_baseline=sc_baseline
        )
    )
    class_accs, overall_acc = self.compute_accuracies(scores_prep, labels_prep)
    precisions, recalls, avg_precisions, mean_ap_micro = (
        self.compute_precision_recall(scores_prep, binarized_labels)
    )
    fprs, tprs, roc_aucs = self.compute_roc(scores_prep, binarized_labels)

    result = EvaluationMetrics(
        eval_mode=self.eval_mode,
        num_frames=int(labels_prep.shape[0]),
        overall_accuracy=overall_acc,
        class_accuracies=class_accs,
        average_precisions=avg_precisions,
        mean_average_precision_micro=mean_ap_micro,
        roc_aucs=roc_aucs,
    )

    if output_dir is not None:
      os.makedirs(output_dir, exist_ok=True)
      roc_path = os.path.join(output_dir, "roc_curves.png")
      pr_path = os.path.join(output_dir, "pr_curves.png")
      html_path = os.path.join(output_dir, "report.html")
      self.plot_roc_curves(fprs, tprs, roc_aucs, roc_path)
      self.plot_pr_curves(
          recalls, precisions, avg_precisions, mean_ap_micro, pr_path
      )
      self.generate_html_report(result, html_path)

    return result

  def plot_roc_curves(
      self,
      fprs: dict[int, np.ndarray],
      tprs: dict[int, np.ndarray],
      roc_aucs: dict[str, float],
      output_path: str,
  ) -> None:
    """Plots per-class ROC curves and saves to `output_path`."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["navy", "turquoise", "darkorange"]
    for i, cls_name, color in zip(self.classes, self.class_names, colors):
      ax.plot(
          fprs[i],
          tprs[i],
          color=color,
          lw=2,
          label=f"{cls_name} (AUC = {roc_aucs[cls_name]:.3f})",
      )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

  def plot_pr_curves(
      self,
      recalls: dict[int, np.ndarray],
      precisions: dict[int, np.ndarray],
      average_precisions: dict[str, float],
      mean_ap_micro: float,
      output_path: str,
  ) -> None:
    """Plots per-class and micro-averaged Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["navy", "turquoise", "darkorange"]

    # Iso-F1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
      x = np.linspace(0.01, 1.0, 100)
      with np.errstate(divide="ignore", invalid="ignore"):
        y = f_score * x / (2.0 * x - f_score)
      valid = (y >= 0) & np.isfinite(y)
      ax.plot(x[valid], y[valid], color="gray", alpha=0.2)

    ax.plot(
        recalls[-1],
        precisions[-1],
        color="gold",
        lw=2,
        label=f"micro-average (mAP = {mean_ap_micro:.3f})",
    )
    for i, cls_name, color in zip(self.classes, self.class_names, colors):
      ax.plot(
          recalls[i],
          precisions[i],
          color=color,
          lw=2,
          label=f"{cls_name} (AP = {average_precisions[cls_name]:.3f})",
      )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

  def generate_html_report(
      self, result: EvaluationMetrics, output_path: str
  ) -> None:
    """Writes an HTML evaluation report."""
    lines = [
        "<html><head><title>Personal VAD Evaluation Report</title>"
        "</head><body>",
        f"<h1>Evaluation Mode: {result.eval_mode}</h1>",
        f"<p>Total Evaluated Frames: {result.num_frames}</p>",
        "<h2>Accuracy</h2>",
        "<ul>",
    ]
    if result.class_accuracies:
      for cls_name, acc in result.class_accuracies.items():
        lines.append(f"  <li>{cls_name}: {acc:.4f}</li>")
    if result.overall_accuracy is not None:
      lines.append(
          f"  <li><b>Overall Accuracy</b>: {result.overall_accuracy:.4f}</li>"
      )
    lines.append("</ul>")

    lines.append("<h2>Average Precision (AP)</h2>")
    lines.append("<ul>")
    if result.average_precisions:
      for cls_name, ap in result.average_precisions.items():
        lines.append(f"  <li>{cls_name}: {ap:.4f}</li>")
    if result.mean_average_precision_micro is not None:
      lines.append(
          f"  <li><b>Mean Average Precision (micro)</b>: "
          f"{result.mean_average_precision_micro:.4f}</li>"
      )
    lines.append("</ul>")

    lines.append("<h2>ROC and Precision-Recall Curves</h2>")
    lines.append('<img src="roc_curves.png" alt="ROC Curves" />')
    lines.append('<img src="pr_curves.png" alt="PR Curves" />')
    lines.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")
