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
    self.bin_edges = np.linspace(
        hist_range[0], hist_range[1], num=bins + 1, dtype=np.float64
    )

  @property
  def histogram(self) -> np.ndarray:
    """Returns normalized histogram probabilities of shape `[bins]`."""
    total = int(np.sum(self.counts))
    if total == 0:
      return np.zeros((self.bins,), dtype=np.float64)
    return self.counts.astype(np.float64) / float(total)

  def reset(self) -> None:
    """Resets all accumulated histogram counts to zero."""
    self.counts.fill(0)

  def observe(self, x: Union[float, np.ndarray, list[float]]) -> None:
    """Observes a scalar or array of values and updates histogram counts.

    Args:
      x: A scalar float, list of floats, or NumPy array of observations.
    """
    hist, _ = np.histogram(x, bins=self.bins, range=self.hist_range)
    self.counts += hist

  def get_percentile_value(self, percentile: float) -> float:
    """Returns the estimated value at the requested percentile.

    Args:
      percentile: Target percentile in `[0.0, 100.0]`.

    Returns:
      The lower edge of the first histogram bin where the cumulative proportion
      of observations preceding the bin is at least `percentile / 100.0`.

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
    threshold = (percentile / 100.0) * float(total)
    cum_before = np.concatenate(([0], np.cumsum(self.counts[:-1])))
    bin_idx = int(np.searchsorted(cum_before, threshold, side="left"))
    return float(self.bin_edges[bin_idx])


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
