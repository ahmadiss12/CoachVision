"""
Geometry utility functions for pose estimation and exercise analysis.
Includes angle calculations, landmark filtering, and coordinate transformations.
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

# Re-export OneEuroFilter for convenience
from .one_euro import OneEuroFilter, LandmarkFilter


def calculate_angle(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float]
) -> float:
    """
    Calculate the angle between three points (in degrees).
    
    Args:
        a: First point (x, y) - e.g., hip
        b: Second point (x, y) - e.g., knee (vertex of angle)
        c: Third point (x, y) - e.g., ankle
        
    Returns:
        Angle in degrees (0-180)
        
    Example:
        angle = calculate_angle(hip, knee, ankle)  # Knee angle
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vectors from the vertex
    ba = a - b
    bc = c - b
    
    # Calculate cosine of angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Clip to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


def calculate_angle_confidence(
    a_conf: float,
    b_conf: float,
    c_conf: float
) -> float:
    """
    Calculate confidence for an angle based on landmark confidences.
    
    Args:
        a_conf: Confidence of first point
        b_conf: Confidence of second point (vertex)
        c_conf: Confidence of third point
        
    Returns:
        Combined confidence (minimum of the three)
    """
    return min(a_conf, b_conf, c_conf)


def calculate_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Distance in pixels/normalized units
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def calculate_midpoint(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calculate midpoint between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Midpoint coordinates (x, y)
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def calculate_angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the smallest difference between two angles (in degrees).
    
    Args:
        angle1: First angle (0-360)
        angle2: Second angle (0-360)
        
    Returns:
        Smallest angle difference (0-180)
    """
    diff = abs(angle1 - angle2) % 360
    return min(diff, 360 - diff)


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [0, 360) degrees.
    
    Args:
        angle: Angle in degrees
        
    Returns:
        Normalized angle
    """
    return angle % 360


def is_angle_between(angle: float, low: float, high: float) -> bool:
    """
    Check if an angle falls within a range (considering circular nature).
    
    Args:
        angle: Angle to check
        low: Lower bound
        high: Upper bound
        
    Returns:
        True if angle is between low and high (inclusive)
    """
    angle = normalize_angle(angle)
    low = normalize_angle(low)
    high = normalize_angle(high)
    
    if low <= high:
        return low <= angle <= high
    else:  # Range wraps around 360
        return angle >= low or angle <= high


@dataclass
class Line:
    """Represents a line in 2D space."""
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    
    @property
    def length(self) -> float:
        """Length of the line segment."""
        return calculate_distance(self.p1, self.p2)
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        """Midpoint of the line segment."""
        return calculate_midpoint(self.p1, self.p2)
    
    @property
    def slope(self) -> Optional[float]:
        """Slope of the line (None for vertical lines)."""
        dx = self.p2[0] - self.p1[0]
        if dx == 0:
            return None
        return (self.p2[1] - self.p1[1]) / dx
    
    @property
    def angle_with_horizontal(self) -> float:
        """Angle of the line with horizontal (in degrees)."""
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        return math.degrees(math.atan2(dy, dx))


def calculate_body_segment_ratio(
    segment1_length: float,
    segment2_length: float,
    reference_ratio: float = 1.0
) -> float:
    """
    Calculate ratio between two body segments and compare to reference.
    
    Useful for detecting stance width, arm length proportions, etc.
    
    Args:
        segment1_length: Length of first segment
        segment2_length: Length of second segment
        reference_ratio: Expected ratio (e.g., 1.0 for symmetry)
        
    Returns:
        Ratio value (positive = longer than reference, negative = shorter)
    """
    if segment2_length == 0:
        return 0.0
    actual_ratio = segment1_length / segment2_length
    return actual_ratio - reference_ratio


def smooth_angle_sequence(
    angles: List[float],
    window_size: int = 3
) -> List[float]:
    """
    Apply moving average smoothing to a sequence of angles.
    Handles angle wrapping correctly.
    
    Args:
        angles: List of angles in degrees
        window_size: Size of moving average window
        
    Returns:
        Smoothed angle sequence
    """
    if len(angles) < window_size:
        return angles.copy()
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(angles)):
        start = max(0, i - half_window)
        end = min(len(angles), i + half_window + 1)
        window = angles[start:end]
        
        # Convert to vectors to handle angle wrapping
        vectors = [(math.cos(math.radians(a)), math.sin(math.radians(a))) 
                  for a in window]
        mean_vector = np.mean(vectors, axis=0)
        mean_angle = math.degrees(math.atan2(mean_vector[1], mean_vector[0]))
        smoothed.append(mean_angle)
    
    return smoothed


def is_pose_symmetric(
    left_points: List[Tuple[float, float]],
    right_points: List[Tuple[float, float]],
    threshold: float = 0.05
) -> bool:
    """
    Check if pose is symmetric by comparing left and right landmarks.
    
    Args:
        left_points: List of left-side landmark coordinates
        right_points: List of right-side landmark coordinates (same order)
        threshold: Maximum allowed difference for symmetry
        
    Returns:
        True if pose is symmetric within threshold
    """
    if len(left_points) != len(right_points):
        return False
    
    for left, right in zip(left_points, right_points):
        # For symmetry, x-coordinates should be mirrored
        if abs(left[0] - (1.0 - right[0])) > threshold:
            return False
        # y-coordinates should be similar
        if abs(left[1] - right[1]) > threshold:
            return False
    
    return True


def convert_to_pixel_coordinates(
    landmark: Tuple[float, float],
    frame_width: int,
    frame_height: int
) -> Tuple[int, int]:
    """
    Convert normalized landmark coordinates to pixel coordinates.
    
    Args:
        landmark: Normalized coordinates (x, y) in [0, 1]
        frame_width: Width of frame in pixels
        frame_height: Height of frame in pixels
        
    Returns:
        Pixel coordinates (x, y) as integers
    """
    return (
        int(landmark[0] * frame_width),
        int(landmark[1] * frame_height)
    )


def convert_to_normalized_coordinates(
    pixel: Tuple[int, int],
    frame_width: int,
    frame_height: int
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to normalized coordinates.
    
    Args:
        pixel: Pixel coordinates (x, y)
        frame_width: Width of frame in pixels
        frame_height: Height of frame in pixels
        
    Returns:
        Normalized coordinates (x, y) in [0, 1]
    """
    return (
        pixel[0] / frame_width,
        pixel[1] / frame_height
    )


class AngleBuffer:
    """
    Circular buffer for smoothing angles with proper wrapping.
    """
    
    def __init__(self, size: int = 5):
        """
        Initialize angle buffer.
        
        Args:
            size: Maximum number of angles to store
        """
        self.size = size
        self.buffer = []
    
    def add(self, angle: float) -> None:
        """Add an angle to the buffer."""
        self.buffer.append(angle)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
    
    def get_smoothed(self) -> Optional[float]:
        """
        Get smoothed angle from buffer.
        
        Returns:
            Smoothed angle or None if buffer empty
        """
        if not self.buffer:
            return None
        
        # Convert to vectors for proper averaging
        vectors = [(math.cos(math.radians(a)), math.sin(math.radians(a))) 
                  for a in self.buffer]
        mean_vector = np.mean(vectors, axis=0)
        return math.degrees(math.atan2(mean_vector[1], mean_vector[0]))
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = []
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) == self.size
    
    @property
    def count(self) -> int:
        """Number of angles in buffer."""
        return len(self.buffer)