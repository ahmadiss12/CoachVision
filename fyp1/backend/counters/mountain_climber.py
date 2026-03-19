"""
Mountain climber counter using a finite state machine with hysteresis.
Tracks alternating knee drives while in plank position.
Combines aspects of push-up (plank position) and high knees (alternating legs).
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle, calculate_distance
from .interface import ExerciseCounter, RepCounterConfig


class MountainClimberState(Enum):
    """Mountain climber FSM states."""
    IDLE = "IDLE"               # No valid pose detected
    PLANK = "PLANK"             # In plank position, both feet back
    LEFT_UP = "LEFT_UP"         # Left knee driving toward chest
    LEFT_DOWN = "LEFT_DOWN"     # Left knee returning
    RIGHT_UP = "RIGHT_UP"       # Right knee driving toward chest
    RIGHT_DOWN = "RIGHT_DOWN"   # Right knee returning


@dataclass
class MountainClimberConfig(RepCounterConfig):
    """
    Configuration for mountain climber counter.
    """
    # Plank position thresholds (shoulder-hip-ankle angle)
    plank_angle_threshold: float = 160.0  # Minimum angle for straight back
    
    # Knee drive thresholds (hip flexion angle)
    knee_drive_threshold: float = 90.0    # Angle when knee is fully driven (smaller = closer to chest)
    knee_start_threshold: float = 160.0   # Angle when leg is extended back
    
    # Hysteresis buffer
    buffer: float = 15.0
    
    # Smoothing settings
    angle_buffer_size: int = 3
    
    # Form thresholds
    hip_stability_threshold: float = 0.05  # Max hip movement (normalized)
    arm_stability_threshold: float = 0.03  # Max arm movement
    
    # Cadence expectations
    expected_cadence: float = 120.0        # Steps per minute (ideal)
    
    def __post_init__(self):
        """Validate configuration."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.knee_start_threshold <= self.knee_drive_threshold:
            raise ValueError("knee_start_threshold must be greater than knee_drive_threshold")


class MountainClimberCounter(ExerciseCounter):
    """
    Mountain climber counter using finite state machine.
    
    Tracks alternating knee drives while maintaining plank position:
    PLANK -> LEFT_UP -> LEFT_DOWN -> PLANK (or RIGHT_UP) -> etc.
    Each complete left+right cycle counts as 1 rep.
    """
    
    def __init__(self, config: Optional[MountainClimberConfig] = None):
        self.config = config or MountainClimberConfig()
        
        # FSM state
        self.state = MountainClimberState.IDLE
        self.count = 0
        
        # Tracking
        self._last_hip_angle = 0.0  # For plank position
        self._last_left_hip_flexion = 0.0
        self._last_right_hip_flexion = 0.0
        self._last_hip_pos: Optional[Tuple[float, float]] = None
        self._last_shoulder_pos: Optional[Tuple[float, float]] = None
        
        self._current_leg = None
        self._step_times: List[float] = []
        
        self._rep_start_time: Optional[float] = None
        self._rep_end_time: Optional[float] = None
        self._rep_metrics: List[Dict[str, Any]] = []
        
        # Smoothing buffers
        self._left_hip_buffer = []
        self._right_hip_buffer = []
        self._plank_angle_buffer = []
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
        
        # Max drive tracking
        self._max_left_drive = 0.0
        self._max_right_drive = 0.0
        self._max_hip_movement = 0.0
        self._max_arm_movement = 0.0
    
    @property
    def name(self) -> str:
        return "mountain_climber"
    
    @property
    def required_landmarks(self) -> list:
        return [
            # Shoulders (for plank position)
            'left_shoulder', 'right_shoulder',
            # Hips (for plank and leg drive)
            'left_hip', 'right_hip',
            # Knees
            'left_knee', 'right_knee',
            # Ankles
            'left_ankle', 'right_ankle',
            # Arms (for stability)
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist'
        ]
    
    def _calculate_plank_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """Calculate shoulder-hip-ankle angle for plank position."""
        # Average positions
        shoulder = (
            (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2,
            (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
        )
        hip = (
            (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2,
            (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
        )
        ankle = (
            (landmarks['left_ankle'][0] + landmarks['right_ankle'][0]) / 2,
            (landmarks['left_ankle'][1] + landmarks['right_ankle'][1]) / 2
        )
        
        return calculate_angle(shoulder, hip, ankle)
    
    def _calculate_hip_flexion(self, landmarks: Dict[str, Tuple[float, float]], leg: str) -> float:
        """
        Calculate hip flexion angle (shoulder-hip-knee).
        Smaller angle = knee closer to chest.
        """
        if leg == 'left':
            return calculate_angle(
                landmarks['left_shoulder'],
                landmarks['left_hip'],
                landmarks['left_knee']
            )
        else:
            return calculate_angle(
                landmarks['right_shoulder'],
                landmarks['right_hip'],
                landmarks['right_knee']
            )
    
    def _check_plank_position(self, plank_angle: float) -> bool:
        """Check if plank position is maintained."""
        if plank_angle < self.config.plank_angle_threshold:
            if plank_angle < self.config.plank_angle_threshold - 20:
                self._form_warnings.append("Hips too high/low - maintain straight line")
            else:
                self._form_warnings.append("Keep back straight - engage core")
            return False
        return True
    
    def _check_stability(self, hip_pos: Tuple[float, float], shoulder_pos: Tuple[float, float]) -> bool:
        """Check if hips and shoulders are stable (not bouncing)."""
        if self._last_hip_pos is not None:
            hip_movement = calculate_distance(hip_pos, self._last_hip_pos)
            self._max_hip_movement = max(self._max_hip_movement, hip_movement)
            
            if hip_movement > self.config.hip_stability_threshold:
                self._form_warnings.append("Keep hips stable - don't bounce")
        
        if self._last_shoulder_pos is not None:
            shoulder_movement = calculate_distance(shoulder_pos, self._last_shoulder_pos)
            self._max_arm_movement = max(self._max_arm_movement, shoulder_movement)
            
            if shoulder_movement > self.config.arm_stability_threshold:
                self._form_warnings.append("Keep upper body stable")
        
        self._last_hip_pos = hip_pos
        self._last_shoulder_pos = shoulder_pos
        return True
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update counter state with new frame data.
        
        Returns:
            Tuple of (count, state, plank_angle)
        """
        self._total_frames += 1
        self._form_warnings = []
        
        if confidence < self.config.min_confidence:
            if self.state != MountainClimberState.IDLE:
                self.state = MountainClimberState.IDLE
            return self.count, self.state, self._last_hip_angle
        
        try:
            # Calculate all metrics
            plank_angle = self._calculate_plank_angle(landmarks)
            left_flexion = self._calculate_hip_flexion(landmarks, 'left')
            right_flexion = self._calculate_hip_flexion(landmarks, 'right')
            
            # Average hip and shoulder positions for stability
            hip_pos = (
                (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2,
                (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
            )
            shoulder_pos = (
                (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2,
                (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Smooth angles
        for buffer, value in [
            (self._left_hip_buffer, left_flexion),
            (self._right_hip_buffer, right_flexion),
            (self._plank_angle_buffer, plank_angle)
        ]:
            buffer.append(value)
            if len(buffer) > self.config.angle_buffer_size:
                buffer.pop(0)
        
        smoothed_left = np.mean(self._left_hip_buffer)
        smoothed_right = np.mean(self._right_hip_buffer)
        smoothed_plank = np.mean(self._plank_angle_buffer)
        
        self._last_left_hip_flexion = smoothed_left
        self._last_right_hip_flexion = smoothed_right
        self._last_hip_angle = smoothed_plank
        
        # Check form
        self._check_plank_position(smoothed_plank)
        self._check_stability(hip_pos, shoulder_pos)
        
        # Determine knee drive status
        left_driven = smoothed_left < self.config.knee_drive_threshold
        right_driven = smoothed_right < self.config.knee_drive_threshold
        
        # Track max drive
        if left_driven:
            self._max_left_drive = max(self._max_left_drive, smoothed_left)
        if right_driven:
            self._max_right_drive = max(self._max_right_drive, smoothed_right)
        
        # Update FSM
        self._update_state(
            left_driven=left_driven,
            right_driven=right_driven,
            left_angle=smoothed_left,
            right_angle=smoothed_right,
            timestamp=time.time()
        )
        
        return self.count, self.state, smoothed_plank
    
    def _update_state(self, left_driven: bool, right_driven: bool,
                     left_angle: float, right_angle: float, timestamp: float) -> None:
        """
        Update FSM based on knee drive status.
        """
        cfg = self.config
        
        if self.state == MountainClimberState.IDLE:
            # IDLE -> PLANK: Person in good plank position
            if not left_driven and not right_driven:
                self.state = MountainClimberState.PLANK
                self._total_valid_frames += 1
        
        elif self.state == MountainClimberState.PLANK:
            # PLANK -> LEFT_UP: Left knee drives forward
            if left_driven and not right_driven:
                self.state = MountainClimberState.LEFT_UP
                self._rep_start_time = timestamp
                self._current_leg = 'left'
                self._step_times.append(timestamp)
                self._max_left_drive = left_angle
            
            # PLANK -> RIGHT_UP: Right knee drives forward
            elif right_driven and not left_driven:
                self.state = MountainClimberState.RIGHT_UP
                self._rep_start_time = timestamp
                self._current_leg = 'right'
                self._step_times.append(timestamp)
                self._max_right_drive = right_angle
        
        elif self.state == MountainClimberState.LEFT_UP:
            # LEFT_UP -> LEFT_DOWN: Left knee returning
            if not left_driven and self._current_leg == 'left':
                self.state = MountainClimberState.LEFT_DOWN
            
            # LEFT_UP -> PLANK: Aborted (both legs back)
            elif not left_driven and not right_driven:
                self.state = MountainClimberState.PLANK
                self._reset_rep_tracking()
        
        elif self.state == MountainClimberState.LEFT_DOWN:
            # LEFT_DOWN -> RIGHT_UP: Right knee drives (alternation)
            if right_driven and not left_driven:
                self.state = MountainClimberState.RIGHT_UP
                self._current_leg = 'right'
                self._step_times.append(timestamp)
                self._max_right_drive = right_angle
            
            # LEFT_DOWN -> PLANK: Both legs back
            elif not left_driven and not right_driven:
                self.state = MountainClimberState.PLANK
                self._reset_rep_tracking()
        
        elif self.state == MountainClimberState.RIGHT_UP:
            # RIGHT_UP -> RIGHT_DOWN: Right knee returning
            if not right_driven and self._current_leg == 'right':
                self.state = MountainClimberState.RIGHT_DOWN
            
            # RIGHT_UP -> PLANK: Aborted
            elif not left_driven and not right_driven:
                self.state = MountainClimberState.PLANK
                self._reset_rep_tracking()
        
        elif self.state == MountainClimberState.RIGHT_DOWN:
            # RIGHT_DOWN -> LEFT_UP: Left knee drives (alternation)
            if left_driven and not right_driven:
                self.state = MountainClimberState.LEFT_UP
                self._current_leg = 'left'
                self._step_times.append(timestamp)
                self._max_left_drive = left_angle
                
                # Each complete left+right cycle counts as 1 rep
                if len(self._step_times) >= 2:
                    self.count += 1
                    self._rep_end_time = timestamp
                    self._store_rep_metrics()
                    self._reset_rep_tracking()
                    self._step_times = [timestamp]  # Keep current step for next cycle
            
            # RIGHT_DOWN -> PLANK: Both legs back
            elif not left_driven and not right_driven:
                self.state = MountainClimberState.PLANK
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
                'max_left_drive_angle': self._max_left_drive,
                'max_right_drive_angle': self._max_right_drive,
                'avg_plank_angle': self._last_hip_angle,
                'max_hip_movement': self._max_hip_movement,
                'max_arm_movement': self._max_arm_movement,
                'form_issues': self._form_warnings.copy() if self._form_warnings else []
            })
    
    def _reset_rep_tracking(self) -> None:
        """Reset tracking variables for a new rep."""
        self._rep_start_time = None
        self._rep_end_time = None
        self._current_leg = None
        self._max_left_drive = 0.0
        self._max_right_drive = 0.0
        self._max_hip_movement = 0.0
        self._max_arm_movement = 0.0
        self._last_hip_pos = None
        self._last_shoulder_pos = None
    
    def reset(self) -> None:
        """Reset counter to initial state."""
        self.state = MountainClimberState.IDLE
        self.count = 0
        self._last_hip_angle = 0.0
        self._last_left_hip_flexion = 0.0
        self._last_right_hip_flexion = 0.0
        self._left_hip_buffer.clear()
        self._right_hip_buffer.clear()
        self._plank_angle_buffer.clear()
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
        if self.state in [MountainClimberState.IDLE, MountainClimberState.PLANK]:
            return 0.0
        elif self.state in [MountainClimberState.LEFT_UP, MountainClimberState.RIGHT_UP]:
            return 0.25
        elif self.state in [MountainClimberState.LEFT_DOWN, MountainClimberState.RIGHT_DOWN]:
            return 0.5
        return 0.0
    
    def get_feedback(self) -> Optional[str]:
        """Get form feedback based on current state."""
        if self._form_warnings:
            return self._form_warnings[0]
        
        if self.state == MountainClimberState.IDLE:
            return "Get into plank position"
        elif self.state == MountainClimberState.PLANK:
            return "Drive knees toward chest - quick!"
        elif self.state in [MountainClimberState.LEFT_UP, MountainClimberState.RIGHT_UP]:
            return "Knee to chest - drive!"
        elif self.state in [MountainClimberState.LEFT_DOWN, MountainClimberState.RIGHT_DOWN]:
            return "Switch legs - keep core tight"
        
        return None
    
    def get_cadence(self) -> float:
        """Calculate current step cadence (steps per minute)."""
        if len(self._step_times) < 2:
            return 0.0
        
        intervals = []
        for i in range(1, len(self._step_times)):
            intervals.append(self._step_times[i] - self._step_times[i-1])
        
        avg_interval = np.mean(intervals)
        return 60.0 / avg_interval if avg_interval > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        if not self._rep_metrics:
            return {
                'exercise': 'mountain_climber',
                'total_reps': 0,
                'avg_duration': 0,
                'avg_cadence': 0,
                'avg_drive_angle': 0,
                'avg_plank_angle': 0,
                'reps_with_form_issues': 0,
                'detection_quality': 0
            }
        
        avg_duration = np.mean([r['duration'] for r in self._rep_metrics])
        avg_cadence = np.mean([r['cadence'] for r in self._rep_metrics])
        avg_drive = np.mean([(r['max_left_drive_angle'] + r['max_right_drive_angle']) / 2 
                            for r in self._rep_metrics])
        avg_plank = np.mean([r['avg_plank_angle'] for r in self._rep_metrics])
        reps_with_issues = sum(1 for r in self._rep_metrics if r['form_issues'])
        
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'mountain_climber',
            'total_reps': len(self._rep_metrics),
            'avg_duration': round(avg_duration, 2),
            'avg_cadence': round(avg_cadence, 1),
            'avg_drive_angle': round(avg_drive, 1),
            'avg_plank_angle': round(avg_plank, 1),
            'reps_with_form_issues': reps_with_issues,
            'detection_quality': round(detection_quality, 1)
        }