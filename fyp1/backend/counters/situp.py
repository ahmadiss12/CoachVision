"""
Sit-up counter using a finite state machine with hysteresis.
Tracks torso angle relative to ground to count repetitions.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle
from .interface import ExerciseCounter, RepCounterConfig


class SitUpState(Enum):
    """Sit-up FSM states."""
    IDLE = "IDLE"           # No valid sit-up pose detected
    DOWN = "DOWN"           # Lying on back (start position)
    ASCENDING = "ASCENDING" # In the ascent phase (sitting up)
    UP = "UP"               # Sitting up position
    DESCENDING = "DESCENDING" # In the descent phase (lowering down)


@dataclass
class SitUpConfig(RepCounterConfig):
    """
    Configuration for sit-up counter.
    """
    # Angle thresholds (in degrees)
    # Torso angle relative to ground: 0° = lying flat, 90° = sitting up
    extension_threshold: float = 80.0   # Angle for "up" position (nearly vertical)
    flexion_threshold: float = 20.0     # Angle for "down" position (nearly flat)
    
    # Hysteresis buffer
    buffer: float = 10.0
    
    # Smoothing settings
    angle_buffer_size: int = 3
    
    # Form thresholds
    knee_angle_target: float = 90.0     # Ideal knee angle (feet anchored)
    knee_angle_tolerance: float = 20.0  # Acceptable knee bend
    hand_position_threshold: float = 0.1  # Hands should stay near head
    
    def __post_init__(self):
        """Validate configuration."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.extension_threshold <= self.flexion_threshold:
            raise ValueError("extension_threshold must be greater than flexion_threshold")

class SitUpCounter(ExerciseCounter):
    """
    Sit-up counter using finite state machine.
    
    Tracks torso angle (shoulder-hip line relative to horizontal):
    DOWN -> ASCENDING -> UP -> DESCENDING -> DOWN (count increments)
    """
    
    def __init__(self, config: Optional[SitUpConfig] = None):
        self.config = config or SitUpConfig()
        
        self.state = SitUpState.IDLE
        self.count = 0
        
        # Tracking
        self._last_torso_angle = 0.0
        self._last_knee_angle = 0.0
        self._last_hand_pos: Optional[Tuple[float, float]] = None
        
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time: Optional[float] = None
        self._rep_end_time: Optional[float] = None
        self._rep_metrics: List[Dict[str, Any]] = []
        
        # Smoothing
        self._torso_buffer = []
        self._knee_buffer = []
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
    
    @property
    def name(self) -> str:
        return "situp"
    
    @property
    def required_landmarks(self) -> list:
        return [
            'left_shoulder', 'right_shoulder',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_wrist', 'right_wrist'
        ]
    
    def _calculate_torso_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate torso angle relative to horizontal.
        Uses shoulder and hip positions.
        """
        # Average shoulder and hip positions
        shoulder = (
            (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2,
            (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
        )
        hip = (
            (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2,
            (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
        )
        
        # Create a horizontal reference point
        horizontal_ref = (shoulder[0] + 1.0, shoulder[1])  # Point to the right
        
        # Angle between shoulder-hip line and horizontal
        angle = calculate_angle(shoulder, hip, horizontal_ref)
        
        # Convert to torso angle relative to ground
        # 0° = lying flat, 90° = sitting up
        return 90.0 - angle
    
    def _calculate_knee_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """Calculate knee angle to ensure feet are anchored properly."""
        left_angle = calculate_angle(
            landmarks['left_hip'],
            landmarks['left_knee'],
            landmarks['left_ankle']
        )
        right_angle = calculate_angle(
            landmarks['right_hip'],
            landmarks['right_knee'],
            landmarks['right_ankle']
        )
        return (left_angle + right_angle) / 2.0
    
    def _check_knee_position(self, knee_angle: float) -> bool:
        """Check if knees are bent correctly (feet anchored)."""
        deviation = abs(knee_angle - self.config.knee_angle_target)
        
        if deviation > self.config.knee_angle_tolerance:
            if knee_angle > self.config.knee_angle_target:
                self._form_warnings.append("Knees too straight - bend them more")
            else:
                self._form_warnings.append("Knees too bent - adjust foot position")
            return False
        return True
    
    def _check_hand_position(self, wrist_pos: Tuple[float, float], shoulder_pos: Tuple[float, float]) -> bool:
        """Check if hands are staying near head (not pulling neck)."""
        if self._last_hand_pos is None:
            self._last_hand_pos = wrist_pos
            return True
        
        # Hands should stay near head/shoulders
        distance_from_shoulder = calculate_distance(wrist_pos, shoulder_pos)
        
        if distance_from_shoulder > self.config.hand_position_threshold * 2:
            self._form_warnings.append("Hands behind head - don't pull on neck")
            return False
        
        self._last_hand_pos = wrist_pos
        return True
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        self._total_frames += 1
        self._form_warnings = []
        
        if confidence < self.config.min_confidence:
            if self.state != SitUpState.IDLE:
                self.state = SitUpState.IDLE
            return self.count, self.state, self._last_torso_angle
        
        try:
            torso_angle = self._calculate_torso_angle(landmarks)
            knee_angle = self._calculate_knee_angle(landmarks)
            
            # Average wrist position for hand check
            wrist_pos = (
                (landmarks['left_wrist'][0] + landmarks['right_wrist'][0]) / 2,
                (landmarks['left_wrist'][1] + landmarks['right_wrist'][1]) / 2
            )
            shoulder_pos = (
                (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2,
                (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Smooth angles
        self._torso_buffer.append(torso_angle)
        self._knee_buffer.append(knee_angle)
        
        for buffer in [self._torso_buffer, self._knee_buffer]:
            if len(buffer) > self.config.angle_buffer_size:
                buffer.pop(0)
        
        smoothed_torso = np.mean(self._torso_buffer)
        smoothed_knee = np.mean(self._knee_buffer)
        
        self._last_torso_angle = smoothed_torso
        self._last_knee_angle = smoothed_knee
        
        # Check form
        self._check_knee_position(smoothed_knee)
        self._check_hand_position(wrist_pos, shoulder_pos)
        
        # Update FSM
        self._update_state(smoothed_torso)
        
        return self.count, self.state, smoothed_torso
    
    def _update_state(self, angle: float) -> None:
        cfg = self.config
        
        if self.state == SitUpState.IDLE:
            if angle < cfg.flexion_threshold + 10:  # Lying down
                self.state = SitUpState.DOWN
                self._total_valid_frames += 1
        
        elif self.state == SitUpState.DOWN:
            if angle > cfg.flexion_threshold + cfg.buffer:
                self.state = SitUpState.ASCENDING
                self._rep_start_time = time.time()
                self._min_angle_in_rep = angle
                self._max_angle_in_rep = angle
        
        elif self.state == SitUpState.ASCENDING:
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            if angle > cfg.extension_threshold:
                self.state = SitUpState.UP
            elif angle < cfg.flexion_threshold:
                self.state = SitUpState.DOWN
                self._reset_rep_tracking()
        
        elif self.state == SitUpState.UP:
            if angle < cfg.extension_threshold - cfg.buffer:
                self.state = SitUpState.DESCENDING
        
        elif self.state == SitUpState.DESCENDING:
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            
            if angle < cfg.flexion_threshold:
                self.count += 1
                self.state = SitUpState.DOWN
                self._rep_end_time = time.time()
                self._store_rep_metrics()
                self._reset_rep_tracking()
            elif angle > cfg.extension_threshold:
                self.state = SitUpState.UP
        
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
            'avg_knee_angle': self._last_knee_angle,
            'form_issues': self._form_warnings.copy()
        })
    
    def _reset_rep_tracking(self) -> None:
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time = None
        self._rep_end_time = None
        self._last_hand_pos = None
    
    def reset(self) -> None:
        self.state = SitUpState.IDLE
        self.count = 0
        self._last_torso_angle = 0.0
        self._last_knee_angle = 0.0
        self._torso_buffer.clear()
        self._knee_buffer.clear()
        self._reset_rep_tracking()
        self._rep_metrics = []
        self._form_warnings = []
        self._total_valid_frames = 0
        self._total_frames = 0
    
    def get_progress(self) -> float:
        cfg = self.config
        angle = self._last_torso_angle
        
        if self.state in [SitUpState.IDLE, SitUpState.DOWN]:
            return 0.0
        elif self.state == SitUpState.ASCENDING:
            span = cfg.extension_threshold - cfg.flexion_threshold
            prog = (angle - cfg.flexion_threshold) / span
            return min(0.5, prog * 0.5)
        elif self.state == SitUpState.UP:
            return 0.5
        elif self.state == SitUpState.DESCENDING:
            span = cfg.extension_threshold - cfg.flexion_threshold
            prog = (cfg.extension_threshold - angle) / span
            return min(1.0, 0.5 + prog * 0.5)
        return 0.0
    
    def get_feedback(self) -> Optional[str]:
        if self._form_warnings:
            return self._form_warnings[0]
        
        angle = self._last_torso_angle
        cfg = self.config
        
        if self.state == SitUpState.IDLE:
            return "Lie on your back with knees bent"
        elif self.state == SitUpState.DOWN:
            return "Sit up - curl your torso"
        elif self.state == SitUpState.ASCENDING:
            if angle < cfg.extension_threshold - 20:
                return "Come all the way up"
            return "Keep going"
        elif self.state == SitUpState.UP:
            if angle < cfg.extension_threshold - 10:
                return "Sit up taller"
            return "Lower with control"
        elif self.state == SitUpState.DESCENDING:
            if angle > cfg.flexion_threshold + 15:
                return "Lower all the way down"
            return "Control the descent"
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        if not self._rep_metrics:
            return {
                'exercise': 'situp',
                'total_reps': 0,
                'avg_range_of_motion': 0,
                'avg_duration': 0,
                'avg_knee_angle': 0,
                'reps_with_form_issues': 0,
                'detection_quality': 0
            }
        
        avg_rom = np.mean([r['range_of_motion'] for r in self._rep_metrics])
        avg_duration = np.mean([r['duration'] for r in self._rep_metrics])
        avg_knee = np.mean([r['avg_knee_angle'] for r in self._rep_metrics])
        reps_with_issues = sum(1 for r in self._rep_metrics if r['form_issues'])
        
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'situp',
            'total_reps': len(self._rep_metrics),
            'avg_range_of_motion': round(avg_rom, 1),
            'avg_duration': round(avg_duration, 2),
            'avg_knee_angle': round(avg_knee, 1),
            'reps_with_form_issues': reps_with_issues,
            'detection_quality': round(detection_quality, 1)
        }