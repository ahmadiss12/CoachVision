"""
OneEuroFilter implementation for real-time landmark smoothing.

Based on: "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input"
by Géry Casiez, Nicolas Roussel, and Daniel Vogel (CHI 2012)

Paper: https://cristal.univ-lille.fr/~casiez/1euro/
"""
import math
from typing import Optional


class OneEuroFilter:
    """
    Adaptive low-pass filter that balances smoothness and responsiveness.

    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing.
                    Default 1.0 works well for pose landmarks.
        beta: Speed coefficient. Higher = less smoothing during fast movement.
              Default 0.0 means constant smoothing. Try 0.007 for more responsiveness.
        d_cutoff: Cutoff frequency for derivative computation. Usually leave at 1.0.

    Usage:
        filter = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        for timestamp, raw_value in data_stream:
            smoothed = filter.filter(raw_value, timestamp)
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # Internal state
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Calculate the smoothing factor alpha."""
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def _exponential_smoothing(self, alpha: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing."""
        return alpha * x + (1.0 - alpha) * x_prev

    def filter(self, x: float, t: float) -> float:
        """
        Filter a single value.

        Args:
            x: Raw input value (e.g., landmark coordinate)
            t: Timestamp in seconds (must be monotonically increasing)

        Returns:
            Smoothed value
        """
        # First sample: initialize and return as-is
        if self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        # Time delta
        t_e = t - self._t_prev

        # Guard against zero or negative time delta
        if t_e <= 0:
            return self._x_prev

        # Estimate derivative (speed)
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self._dx_prev)

        # Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter the value
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self._x_prev)

        # Update state
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    def reset(self) -> None:
        """Reset filter state. Call when tracking is lost."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class LandmarkFilter:
    """
    Convenience wrapper to filter all coordinates of a landmark.

    Maintains separate OneEuroFilter instances for x and y coordinates.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0):
        self.filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def filter(self, x: float, y: float, t: float) -> tuple[float, float]:
        """
        Filter a 2D landmark coordinate.

        Args:
            x: Raw x coordinate
            y: Raw y coordinate
            t: Timestamp in seconds

        Returns:
            Tuple of (smoothed_x, smoothed_y)
        """
        return (
            self.filter_x.filter(x, t),
            self.filter_y.filter(y, t)
        )

    def reset(self) -> None:
        """Reset both coordinate filters."""
        self.filter_x.reset()
        self.filter_y.reset()
