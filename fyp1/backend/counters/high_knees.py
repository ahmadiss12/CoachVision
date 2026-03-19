"""
High knees counter using a finite state machine with hysteresis.
Tracks knee lift height and alternation between legs.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle
from .interface import ExerciseCounter, RepCounterConfig


class HighKneesState(Enum):
    """High knees FSM states."""
    IDLE = "IDLE"               # No valid pose detected
    STANDING = "STANDING"       # Standing ready
    LEFT_UP = "LEFT_UP"         # Left knee raised
    LEFT_DOWN = "LEFT_DOWN"     # Left knee lowering
    RIGHT_UP = "RIGHT_UP"       # Right knee raised
    RIGHT_DOWN = "RIGHT_DOWN"   # Right knee lowering


@dataclass
class HighKneesConfig(RepCounterConfig):
    """
    Configuration for high knees counter.
    """
    # Knee height threshold (normalized y-coordinate difference from hip)
    knee_height_threshold: float = 0.15   # How high knee should rise
    
    # Hip angle thresholds (for detecting knee raise)
    hip_flexion_threshold: float = 90.0   # Angle when knee is high (hip flexed)
    hip_extension_threshold: float = 170.0 # Angle when standing
    
    # Arm drive threshold
    arm_drive_threshold: float = 0.1      # Opposite arm should drive
    
    # Hysteresis buffer
    buffer: float = 10.0
    
    # Smoothing settings
    angle_buffer_size: int = 3
    
    # Cadence expectations
    expected_cadence: float = 180.0       # Steps per minute (ideal)
    
    def __post_init__(self):
        """Validate configuration."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.hip_extension_threshold <= self.hip_flexion_threshold:
            raise ValueError("hip_extension_threshold must be greater than hip_flexion_threshold")

class HighKneesCounter(ExerciseCounter):
    """
    High knees counter using finite state machine.
    
    Tracks alternating knee raises:
    Alternates between left and right knee lifts.
    Each complete cycle (left+right) counts as 1 rep.
    """
    
    def __init__(self, config: Optional[HighKneesConfig] = None):
        self.config = config or HighKneesConfig()
        
        # FSM state
        self.state = HighKneesState.IDLE
        self.count = 0
        
        # Tracking
        self._last_left_hip_angle = 0.0
        self._last_right_hip_angle = 0.0
        self._last_left_knee_height = 0.0
        self._last_right_knee_height = 0.0
        self._last_arm_position: Optional[Tuple[float, float]] = None
        
        self._current_leg = None
        self._last_leg_time = 0.0
        self._step_times: List[float] = []
        
        self._rep_start_time: Optional[float] = None
        self._rep_end_time: Optional[float] = None
        self._rep_metrics: List[Dict[str, Any]] = []
        
        # Smoothing buffers
        self._left_hip_buffer = []
        self._right_hip_buffer = []
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
        
        # Max heights
        self._max_left_height = 0.0
        self._max_right_height = 0.0
    
    @property
    def name(self) -> str:
        return "high_knees"
    
    @property
    def required_landmarks(self) -> list:
        return [
            # Shoulders (for posture)
            'left_shoulder', 'right_shoulder',
            # Hips (primary tracking)
            'left_hip', 'right_hip',
            # Knees
            'left_knee', 'right_knee',
            # Ankles
            'left_ankle', 'right_ankle',
            # Arms (for arm drive)
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist'
        ]
    
    def _calculate_hip_angles(self, landmarks: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate hip flexion angles (shoulder-hip-knee).
        Smaller angle = more flexion = knee higher.
        """
        left_angle = calculate_angle(
            landmarks['left_shoulder'],
            landmarks['left_hip'],
            landmarks['left_knee']
        )
        
        right_angle = calculate_angle(
            landmarks['right_shoulder'],
            landmarks['right_hip'],
            landmarks['right_knee']
        )
        
        return left_angle, right_angle
    
    def _calculate_knee_height(self, knee_pos: Tuple[float, float], hip_pos: Tuple[float, float]) -> float:
        """
        Calculate how high knee is raised relative to hip.
        Negative value means knee above hip.
        """
        return hip_pos[1] - knee_pos[1]  # In image coords, smaller y = higher
    
    def _check_arm_drive(self, landmarks: Dict[str, Tuple[float, float]], raised_leg: str) -> bool:
        """
        Check if opposite arm is driving (moving forward).
        
        Args:
            landmarks: Current landmarks
            raised_leg: Which leg is raised ('left' or 'right')
        """
        if raised_leg == 'left':
            # Right arm should be forward
            driving_arm_pos = landmarks['right_wrist']
            shoulder_pos = landmarks['right_shoulder']
        else:
            # Left arm should be forward
            driving_arm_pos = landmarks['left_wrist']
            shoulder_pos = landmarks['left_shoulder']
        
        # Check if arm is forward of shoulder
        arm_forward = driving_arm_pos[0] > shoulder_pos[0]  # In image coords, right is positive
        
        if not arm_forward:
            self._form_warnings.append("Drive opposite arm for momentum")
            return False
        
        return True
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update counter state with new frame data.
        
        Returns:
            Tuple of (count, state, avg_hip_angle)
        """
        self._total_frames += 1
        self._form_warnings = []
        
        if confidence < self.config.min_confidence:
            if self.state != HighKneesState.IDLE:
                self.state = HighKneesState.IDLE
            return self.count, self.state, (self._last_left_hip_angle + self._last_right_hip_angle) / 2
        
        try:
            left_hip_angle, right_hip_angle = self._calculate_hip_angles(landmarks)
            
            # Calculate knee heights
            left_knee_height = self._calculate_knee_height(
                landmarks['left_knee'],
                landmarks['left_hip']
            )
            right_knee_height = self._calculate_knee_height(
                landmarks['right_knee'],
                landmarks['right_hip']
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Smooth angles
        self._left_hip_buffer.append(left_hip_angle)
        self._right_hip_buffer.append(right_hip_angle)
        
        for buffer in [self._left_hip_buffer, self._right_hip_buffer]:
            if len(buffer) > self.config.angle_buffer_size:
                buffer.pop(0)
        
        smoothed_left = np.mean(self._left_hip_buffer)
        smoothed_right = np.mean(self._right_hip_buffer)
        
        self._last_left_hip_angle = smoothed_left
        self._last_right_hip_angle = smoothed_right
        self._last_left_knee_height = left_knee_height
        self._last_right_knee_height = right_knee_height
        
        # Track max heights
        self._max_left_height = max(self._max_left_height, left_knee_height)
        self._max_right_height = max(self._max_right_height, right_knee_height)
        
        # Determine which knee is raised
        left_raised = left_knee_height > self.config.knee_height_threshold
        right_raised = right_knee_height > self.config.knee_height_threshold
        
        # Check form
        if left_raised:
            self._check_arm_drive(landmarks, 'left')
        elif right_raised:
            self._check_arm_drive(landmarks, 'right')
        
        # Check knee height
        target_height = self.config.knee_height_threshold
        if left_raised and left_knee_height < target_height * 0.8:
            self._form_warnings.append("Left knee higher - aim for waist height")
        if right_raised and right_knee_height < target_height * 0.8:
            self._form_warnings.append("Right knee higher - aim for waist height")
        
        # Update FSM
        self._update_state(
            left_raised=left_raised,
            right_raised=right_raised,
            left_angle=smoothed_left,
            right_angle=smoothed_right,
            timestamp=time.time()
        )
        
        # Return average hip angle as primary metric
        avg_angle = (smoothed_left + smoothed_right) / 2
        return self.count, self.state, avg_angle
    
    def _update_state(self, left_raised: bool, right_raised: bool, 
                     left_angle: float, right_angle: float, timestamp: float) -> None:
        """
        Update FSM based on which knee is raised.
        """
        cfg = self.config
        
        if self.state == HighKneesState.IDLE:
            # IDLE -> STANDING: Person standing
            if not left_raised and not right_raised:
                self.state = HighKneesState.STANDING
                self._total_valid_frames += 1
        
        elif self.state == HighKneesState.STANDING:
            # STANDING -> LEFT_UP: Left knee raises
            if left_raised and not right_raised:
                self.state = HighKneesState.LEFT_UP
                self._rep_start_time = timestamp
                self._current_leg = 'left'
                self._max_left_height = self._last_left_knee_height
                self._step_times.append(timestamp)
            
            # STANDING -> RIGHT_UP: Right knee raises
            elif right_raised and not left_raised:
                self.state = HighKneesState.RIGHT_UP
                self._rep_start_time = timestamp
                self._current_leg = 'right'
                self._max_right_height = self._last_right_knee_height
                self._step_times.append(timestamp)
        
        elif self.state == HighKneesState.LEFT_UP:
            # LEFT_UP -> LEFT_DOWN: Left knee lowering
            if not left_raised and self._current_leg == 'left':
                self.state = HighKneesState.LEFT_DOWN
            
            # LEFT_UP -> STANDING: Aborted (both feet down)
            elif not left_raised and not right_raised:
                self.state = HighKneesState.STANDING
                self._reset_rep_tracking()
        
        elif self.state == HighKneesState.LEFT_DOWN:
            # LEFT_DOWN -> RIGHT_UP: Right knee raises (alternation)
            if right_raised and not left_raised:
                self.state = HighKneesState.RIGHT_UP
                self._current_leg = 'right'
                self._max_right_height = self._last_right_knee_height
                self._step_times.append(timestamp)
            
            # LEFT_DOWN -> STANDING: Both feet down
            elif not left_raised and not right_raised:
                self.state = HighKneesState.STANDING
                self._reset_rep_tracking()
        
        elif self.state == HighKneesState.RIGHT_UP:
            # RIGHT_UP -> RIGHT_DOWN: Right knee lowering
            if not right_raised and self._current_leg == 'right':
                self.state = HighKneesState.RIGHT_DOWN
            
            # RIGHT_UP -> STANDING: Aborted
            elif not left_raised and not right_raised:
                self.state = HighKneesState.STANDING
                self._reset_rep_tracking()
        
        elif self.state == HighKneesState.RIGHT_DOWN:
            # RIGHT_DOWN -> LEFT_UP: Left knee raises (alternation)
            if left_raised and not right_raised:
                self.state = HighKneesState.LEFT_UP
                self._current_leg = 'left'
                self._max_left_height = self._last_left_knee_height
                self._step_times.append(timestamp)
                
                # Each complete left-right cycle counts as 1 rep
                if len(self._step_times) >= 2:
                    self.count += 1
                    self._rep_end_time = timestamp
                    self._store_rep_metrics()
                    self._reset_rep_tracking()
                    self._step_times = [timestamp]  # Keep current step for next cycle
            
            # RIGHT_DOWN -> STANDING: Both feet down
            elif not left_raised and not right_raised:
                self.state = HighKneesState.STANDING
                self._reset_rep_tracking()
        
        self._total_valid_frames += 1
    
    def _store_rep_metrics(self) -> None:
        """Store metrics for the completed repetition (one left+right cycle)."""
        if self._rep_start_time and self._rep_end_time and len(self._step_times) >= 2:
            duration = self._rep_end_time - self._rep_start_time
            cadence = 60.0 / (duration / 2) if duration > 0 else 0  # Steps per minute
            
            self._rep_metrics.append({
                'rep_number': len(self._rep_metrics) + 1,
                'duration': duration,
                'cadence': cadence,
                'max_left_height': self._max_left_height,
                'max_right_height': self._max_right_height,
                'avg_height': (self._max_left_height + self._max_right_height) / 2,
                'form_issues': self._form_warnings.copy() if self._form_warnings else []
            })
    
    def _reset_rep_tracking(self) -> None:
        """Reset tracking variables for a new rep."""
        self._rep_start_time = None
        self._rep_end_time = None
        self._current_leg = None
        self._max_left_height = 0.0
        self._max_right_height = 0.0
    
    def reset(self) -> None:
        """Reset counter to initial state."""
        self.state = HighKneesState.IDLE
        self.count = 0
        self._last_left_hip_angle = 0.0
        self._last_right_hip_angle = 0.0
        self._last_left_knee_height = 0.0
        self._last_right_knee_height = 0.0
        self._left_hip_buffer.clear()
        self._right_hip_buffer.clear()
        self._step_times.clear()
        self._reset_rep_tracking()
        self._rep_metrics = []
        self._form_warnings = []
        self._total_valid_frames = 0
        self._total_frames = 0
    
    def get_progress(self) -> float:
        """
        Get normalized progress of current step cycle.
        """
        if self.state in [HighKneesState.IDLE, HighKneesState.STANDING]:
            return 0.0
        elif self.state in [HighKneesState.LEFT_UP, HighKneesState.RIGHT_UP]:
            return 0.25
        elif self.state in [HighKneesState.LEFT_DOWN, HighKneesState.RIGHT_DOWN]:
            return 0.5
        return 0.0
    
    def get_feedback(self) -> Optional[str]:
        """Get form feedback based on current state."""
        if self._form_warnings:
            return self._form_warnings[0]
        
        if self.state == HighKneesState.IDLE:
            return "Stand ready to begin"
        elif self.state == HighKneesState.STANDING:
            return "Start running in place - drive knees up"
        elif self.state in [HighKneesState.LEFT_UP, HighKneesState.RIGHT_UP]:
            return "Knees up! Aim for waist height"
        elif self.state in [HighKneesState.LEFT_DOWN, HighKneesState.RIGHT_DOWN]:
            return "Quick feet - drive opposite arm"
        
        return None
    
    def get_cadence(self) -> float:
        """Calculate current step cadence (steps per minute)."""
        if len(self._step_times) < 2:
            return 0.0
        
        # Calculate average time between steps
        intervals = []
        for i in range(1, len(self._step_times)):
            intervals.append(self._step_times[i] - self._step_times[i-1])
        
        avg_interval = np.mean(intervals)
        return 60.0 / avg_interval if avg_interval > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        if not self._rep_metrics:
            return {
                'exercise': 'high_knees',
                'total_reps': 0,
                'avg_duration': 0,
                'avg_cadence': 0,
                'avg_knee_height': 0,
                'reps_with_form_issues': 0,
                'detection_quality': 0
            }
        
        avg_duration = np.mean([r['duration'] for r in self._rep_metrics])
        avg_cadence = np.mean([r['cadence'] for r in self._rep_metrics])
        avg_height = np.mean([r['avg_height'] for r in self._rep_metrics])
        reps_with_issues = sum(1 for r in self._rep_metrics if r['form_issues'])
        
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'high_knees',
            'total_reps': len(self._rep_metrics),
            'avg_duration': round(avg_duration, 2),
            'avg_cadence': round(avg_cadence, 1),
            'avg_knee_height': round(avg_height, 3),
            'reps_with_form_issues': reps_with_issues,
            'detection_quality': round(detection_quality, 1)
        }