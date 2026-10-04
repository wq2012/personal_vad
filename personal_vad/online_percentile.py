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
"""Online histogram-based percentile estimator.

Used for rescaling frame-level speaker verification cosine similarity scores in
the Personal VAD 1.0 Score Combination (`SC`) baseline (arXiv:1908.04284,
Section 2.1).
"""

from typing import Union
import numpy as np


class OnlinePercentileValue:
  """Computes online approximate percentiles using a fixed-bin histogram."""

  def __init__(
      self,
      bins: int = 10,
      hist_range: tuple[float, float] = (0.0, 1.0),
  ):
    """Initializes OnlinePercentileValue.

    Args:
      bins: Number of equal-width bins in the histogram (default 10; 200 is used
        in the Personal VAD SC baseline).
      hist_range: The `(lower, upper)` range of the histogram bins.

    Raises:
      ValueError: If `bins <= 0` or `hist_range[0] >= hist_range[1]`.
    """
    if bins <= 0:
      raise ValueError(f"bins must be positive, got {bins}")
    if hist_range[0] >= hist_range[1]:
      raise ValueError(
          f"hist_range lower bound must be < upper bound, got {hist_range}"
      )
    self.bins = bins
    self.hist_range = hist_range
    self.counts = np.zeros((bins,), dtype=np.int64)
    _, self.bin_edges = np.histogram([], bins=bins, range=hist_range)

  @property
  def histogram(self) -> np.ndarray:
    """Returns normalized histogram counts of shape `[bins]`."""
    total = int(np.sum(self.counts))
    if total == 0:
      return np.zeros((self.bins,), dtype=np.float64)
    return self.counts.astype(np.float64) / float(total)

  def observe(self, x: Union[float, np.ndarray, list[float]]) -> None:
    """Observes a scalar or array of values and updates histogram counts.

    Args:
      x: A scalar float, list of floats, or NumPy array of observations.
    """
    hist, _ = np.histogram(x, bins=self.bins, range=self.hist_range)
    self.counts += hist

  def Update(self, x: Union[float, np.ndarray, list[float]]) -> None:
    """Alias for `observe`."""
    self.observe(x)

  def get_percentile_value(self, percentile: float) -> float:
    """Returns the estimated value at the requested percentile.

    Args:
      percentile: Target percentile in `[0.0, 100.0]`.

    Returns:
      The lower edge of the first histogram bin where the cumulative proportion
      of observations is at least `percentile / 100.0`.

    Raises:
      ValueError: If `percentile` is outside `[0.0, 100.0]` or no observations
        have been recorded within `hist_range`.
    """
    if not 0.0 <= percentile <= 100.0:
      raise ValueError(
          f"percentile must be in [0.0, 100.0], got {percentile}"
      )
    total = int(np.sum(self.counts))
    if total == 0:
      raise ValueError("Cannot compute percentile before observing any values.")
    index = (percentile / 100.0) * float(total)
    count = 0
    for i in range(self.bins):
      if count >= index:
        return float(self.bin_edges[i])
      count += int(self.counts[i])
    return float(self.hist_range[1])

  def GetPercentile(self, percentile: float) -> float:
    """Alias for `get_percentile_value`."""
    return self.get_percentile_value(percentile)


def rescale_cosine_scores(
    cosine_scores: np.ndarray,
    bins: int = 200,
    hist_range: tuple[float, float] = (-1.0, 1.0),
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> np.ndarray:
  """Rescales cosine similarity scores to `[0, 1]` using online percentiles.

  Implements the score rescaling in Section 2.1 of Personal VAD
  (arXiv:1908.04284).

  Args:
    cosine_scores: NumPy array of cosine similarity scores in `[-1, 1]`.
    bins: Number of histogram bins (default 200).
    hist_range: Histogram range (default `(-1.0, 1.0)`).
    low_percentile: Lower percentile mapped to 0.0 (default 5.0).
    high_percentile: Upper percentile mapped to 1.0 (default 95.0).

  Returns:
    Rescaled scores clipped to `[0.0, 1.0]` with the same shape as
    `cosine_scores`.
  """
  scores = np.asarray(cosine_scores, dtype=np.float32)
  opv = OnlinePercentileValue(bins=bins, hist_range=hist_range)
  opv.observe(scores)
  p_low = opv.get_percentile_value(low_percentile)
  p_high = opv.get_percentile_value(high_percentile)
  denom = max(p_high - p_low, 1e-6)
  rescaled = (scores - p_low) / denom
  return np.clip(rescaled, 0.0, 1.0)
