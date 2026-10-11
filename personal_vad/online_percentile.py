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
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    lo, hi = self.hist_range
    valid = arr[(arr >= lo) & (arr <= hi)]
    if valid.size == 0:
      return
    idx = np.floor((valid - lo) / (hi - lo) * self.bins).astype(np.int64)
    idx = np.clip(idx, 0, self.bins - 1)
    np.add.at(self.counts, idx, 1)

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
    causal: bool = True,
) -> np.ndarray:
  """Rescales cosine similarity scores to `[0, 1]` using online percentiles.

  Implements the online score rescaling in Section 2.1 of Personal VAD
  (arXiv:1908.04284). When `causal=True` (default), the histogram is updated
  frame-by-frame so each frame `t` only depends on observations up to frame `t`
  (seeded with a two-point prior at `-0.35` and `+0.55` so the initial frames
  have a non-degenerate dynamic range).

  Args:
    cosine_scores: NumPy array of cosine similarity scores in `[-1, 1]`.
    bins: Number of histogram bins (default 200).
    hist_range: Histogram range (default `(-1.0, 1.0)`).
    low_percentile: Lower percentile mapped to 0.0 (default 5.0).
    high_percentile: Upper percentile mapped to 1.0 (default 95.0).
    causal: If True, updates percentiles causally frame-by-frame.

  Returns:
    Rescaled scores clipped to `[0.0, 1.0]` with the same shape as
    `cosine_scores`.
  """
  scores = np.asarray(cosine_scores, dtype=np.float32)
  orig_shape = scores.shape
  flat = scores.reshape(-1)
  if flat.size == 0:
    return scores.copy()

  opv = OnlinePercentileValue(bins=bins, hist_range=hist_range)
  if not causal:
    opv.observe(flat)
    p_low = opv.get_percentile_value(low_percentile)
    p_high = opv.get_percentile_value(high_percentile)
    denom = max(p_high - p_low, 1e-6)
    return np.clip((scores - p_low) / denom, 0.0, 1.0).astype(np.float32)

  # Seed with two anchor points so t=0 has a well-calibrated initial range
  opv.observe([-0.35, 0.55])
  out = np.zeros_like(flat, dtype=np.float32)
  for t in range(flat.size):
    val = float(flat[t])
    opv.observe(val)
    p_low = opv.get_percentile_value(low_percentile)
    p_high = opv.get_percentile_value(high_percentile)
    denom = max(p_high - p_low, 1e-2)
    out[t] = np.clip((val - p_low) / denom, 0.0, 1.0)
  return out.reshape(orig_shape)
