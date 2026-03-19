"""
Shoulder press counter using a finite state machine with hysteresis.
Tracks elbow angle and arm position to count repetitions.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle
from .interface import ExerciseCounter, RepCounterConfig


class ShoulderPressState(Enum):
    """Shoulder press FSM states."""
    IDLE = "IDLE"           # No valid press pose detected
    DOWN = "DOWN"           # Arms down (elbows bent, weights at shoulder level)
    ASCENDING = "ASCENDING" # In the ascent phase (pressing up)
    UP = "UP"               # Arms extended overhead
    DESCENDING = "DESCENDING" # In the descent phase (lowering down)


@dataclass
class ShoulderPressConfig(RepCounterConfig):
    """
    Configuration for shoulder press counter.
    """
    # Angle thresholds (in degrees)
    extension_threshold: float = 170.0  # Angle for arms extended overhead
    flexion_threshold: float = 90.0     # Angle for starting position (elbows at 90°)
    
    # Hysteresis buffer
    buffer: float = 15.0
    
    # Smoothing settings
    angle_buffer_size: int = 3
    
    # Form thresholds
    symmetry_threshold: float = 15.0    # Max difference between left and right angles
    lean_threshold: float = 0.05        # Max body lean (normalized)
    
    # Which arm to track (left, right, or average)
    track_both_arms: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.extension_threshold <= self.flexion_threshold:
            raise ValueError("extension_threshold must be greater than flexion_threshold")

class ShoulderPressCounter(ExerciseCounter):
    """
    Shoulder press counter using finite state machine.
    
    Tracks elbow angle through a complete press cycle:
    DOWN -> ASCENDING -> UP -> DESCENDING -> DOWN (count increments)
    """
    
    def __init__(self, config: Optional[ShoulderPressConfig] = None):
        self.config = config or ShoulderPressConfig()
        
        self.state = ShoulderPressState.IDLE
        self.count = 0
        
        # Tracking
        self._last_angle = 0.0
        self._last_hip_pos: Optional[Tuple[float, float]] = None
        self._initial_hip_pos: Optional[Tuple[float, float]] = None
        
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time: Optional[float] = None
        self._rep_end_time: Optional[float] = None
        self._rep_metrics: List[Dict[str, Any]] = []
        
        # Smoothing
        self._angle_buffer = []
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
        
        # Symmetry tracking
        self._max_asymmetry = 0.0
        self._max_lean = 0.0
    
    @property
    def name(self) -> str:
        return "shoulder_press"
    
    @property
    def required_landmarks(self) -> list:
        return [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip'
        ]
    
    def _calculate_press_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> Tuple[float, float, float]:
        """
        Calculate elbow angles for both arms.
        
        Returns:
            Tuple of (left_angle, right_angle, avg_angle)
        """
        left_angle = calculate_angle(
            landmarks['left_shoulder'],
            landmarks['left_elbow'],
            landmarks['left_wrist']
        )
        
        right_angle = calculate_angle(
            landmarks['right_shoulder'],
            landmarks['right_elbow'],
            landmarks['right_wrist']
        )
        
        avg_angle = (left_angle + right_angle) / 2.0
        
        return left_angle, right_angle, avg_angle
    
    def _check_symmetry(self, left_angle: float, right_angle: float) -> bool:
        """Check if arms are moving symmetrically."""
        asymmetry = abs(left_angle - right_angle)
        self._max_asymmetry = max(self._max_asymmetry, asymmetry)
        
        if asymmetry > self.config.symmetry_threshold:
            if asymmetry > self.config.symmetry_threshold * 2:
                self._form_warnings.append("Arms very uneven - press equally")
            else:
                self._form_warnings.append("Keep arms symmetrical")
            return False
        return True
    
    def _check_body_lean(self, hip_pos: Tuple[float, float]) -> bool:
        """Check if body is leaning too much."""
        if self._last_hip_pos is None:
            self._last_hip_pos = hip_pos
            self._initial_hip_pos = hip_pos
            return True
        
        # Calculate horizontal movement
        lean = abs(hip_pos[0] - self._initial_hip_pos[0])
        self._max_lean = max(self._max_lean, lean)
        
        if lean > self.config.lean_threshold:
            self._form_warnings.append("Don't lean back - keep core tight")
            return False
        
        self._last_hip_pos = hip_pos
        return True
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        self._total_frames += 1
        self._form_warnings = []
        
        if confidence < self.config.min_confidence:
            if self.state != ShoulderPressState.IDLE:
                self.state = ShoulderPressState.IDLE
            return self.count, self.state, self._last_angle
        
        try:
            left_angle, right_angle, avg_angle = self._calculate_press_angle(landmarks)
            
            # Average hip position for lean check
            hip_pos = (
                (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2,
                (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Smooth angle
        self._angle_buffer.append(avg_angle)
        if len(self._angle_buffer) > self.config.angle_buffer_size:
            self._angle_buffer.pop(0)
        
        smoothed_angle = np.mean(self._angle_buffer)
        self._last_angle = smoothed_angle
        
        # Check form
        self._check_symmetry(left_angle, right_angle)
        self._check_body_lean(hip_pos)
        
        # Update FSM
        self._update_state(smoothed_angle)
        
        return self.count, self.state, smoothed_angle
    
    def _update_state(self, angle: float) -> None:
        cfg = self.config
        
        if self.state == ShoulderPressState.IDLE:
            if angle < cfg.flexion_threshold + 20:  # Arms at starting position
                self.state = ShoulderPressState.DOWN
                self._total_valid_frames += 1
        
        elif self.state == ShoulderPressState.DOWN:
            if angle > cfg.flexion_threshold + cfg.buffer:
                self.state = ShoulderPressState.ASCENDING
                self._rep_start_time = time.time()
                self._min_angle_in_rep = angle
                self._max_angle_in_rep = angle
        
        elif self.state == ShoulderPressState.ASCENDING:
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            if angle > cfg.extension_threshold:
                self.state = ShoulderPressState.UP
            elif angle < cfg.flexion_threshold:
                self.state = ShoulderPressState.DOWN
                self._reset_rep_tracking()
        
        elif self.state == ShoulderPressState.UP:
            if angle < cfg.extension_threshold - cfg.buffer:
                self.state = ShoulderPressState.DESCENDING
        
        elif self.state == ShoulderPressState.DESCENDING:
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            
            if angle < cfg.flexion_threshold:
                self.count += 1
                self.state = ShoulderPressState.DOWN
                self._rep_end_time = time.time()
                self._store_rep_metrics()
                self._reset_rep_tracking()
            elif angle > cfg.extension_threshold:
                self.state = ShoulderPressState.UP
        
        self._total_valid_frames += 1
    
    def _store_rep_metrics(self) -> None:
        if self._rep_start_time and self._rep_end_time:
            duration = self._rep_end_time - self._rep_start_time
        else:
            duration = 0.0
        
        self._rep_metrics.append({
            'rep_number': len(self._rep_metrics) + 1,
            'min_angle': self._min_angle_in_rep,
            'max_angle': self._max_angle_in_rep,
            'range_of_motion': self._max_angle_in_rep - self._min_angle_in_rep,
            'duration': duration,
            'max_asymmetry': self._max_asymmetry,
            'max_lean': self._max_lean,
            'form_issues': self._form_warnings.copy()
        })
    
    def _reset_rep_tracking(self) -> None:
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time = None
        self._rep_end_time = None
        self._last_hip_pos = None
        self._initial_hip_pos = None
        self._max_asymmetry = 0.0
        self._max_lean = 0.0
    
    def reset(self) -> None:
        self.state = ShoulderPressState.IDLE
        self.count = 0
        self._last_angle = 0.0
        self._angle_buffer.clear()
        self._reset_rep_tracking()
        self._rep_metrics = []
        self._form_warnings = []
        self._total_valid_frames = 0
        self._total_frames = 0
    
    def get_progress(self) -> float:
        cfg = self.config
        angle = self._last_angle
        
        if self.state in [ShoulderPressState.IDLE, ShoulderPressState.DOWN]:
            return 0.0
        elif self.state == ShoulderPressState.ASCENDING:
            span = cfg.extension_threshold - cfg.flexion_threshold
            prog = (angle - cfg.flexion_threshold) / span
            return min(0.5, prog * 0.5)
        elif self.state == ShoulderPressState.UP:
            return 0.5
        elif self.state == ShoulderPressState.DESCENDING:
            span = cfg.extension_threshold - cfg.flexion_threshold
            prog = (cfg.extension_threshold - angle) / span
            return min(1.0, 0.5 + prog * 0.5)
        return 0.0
    
    def get_feedback(self) -> Optional[str]:
        if self._form_warnings:
            return self._form_warnings[0]
        
        angle = self._last_angle
        cfg = self.config
        
        if self.state == ShoulderPressState.IDLE:
            return "Bring weights to shoulder height"
        elif self.state == ShoulderPressState.DOWN:
            return "Press upward"
        elif self.state == ShoulderPressState.ASCENDING:
            if angle < cfg.extension_threshold - 30:
                return "Press all the way up"
            return "Keep going"
        elif self.state == ShoulderPressState.UP:
            if angle < cfg.extension_threshold - 10:
                return "Fully extend arms at the top"
            return "Lower with control"
        elif self.state == ShoulderPressState.DESCENDING:
            if angle > cfg.flexion_threshold + 20:
                return "Lower to shoulder level"
            return "Control the descent"
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        if not self._rep_metrics:
            return {
                'exercise': 'shoulder_press',
                'total_reps': 0,
                'avg_range_of_motion': 0,
                'avg_duration': 0,
                'avg_asymmetry': 0,
                'avg_lean': 0,
                'reps_with_form_issues': 0,
                'detection_quality': 0
            }
        
        avg_rom = np.mean([r['range_of_motion'] for r in self._rep_metrics])
        avg_duration = np.mean([r['duration'] for r in self._rep_metrics])
        avg_asymmetry = np.mean([r['max_asymmetry'] for r in self._rep_metrics])
        avg_lean = np.mean([r['max_lean'] for r in self._rep_metrics])
        reps_with_issues = sum(1 for r in self._rep_metrics if r['form_issues'])
        
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'shoulder_press',
            'total_reps': len(self._rep_metrics),
            'avg_range_of_motion': round(avg_rom, 1),
            'avg_duration': round(avg_duration, 2),
            'avg_asymmetry': round(avg_asymmetry, 1),
            'avg_lean': round(avg_lean, 3),
            'reps_with_form_issues': reps_with_issues,
            'detection_quality': round(detection_quality, 1)
        }