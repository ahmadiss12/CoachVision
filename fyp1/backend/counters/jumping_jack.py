"""
Jumping jack counter using a finite state machine with hysteresis.
Tracks arm and leg angles simultaneously to detect the starfish position.
More complex due to coordinated movement of all four limbs.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle
from .interface import ExerciseCounter, RepCounterConfig


class JumpingJackState(Enum):
    """Jumping jack FSM states."""
    IDLE = "IDLE"               # No valid pose detected
    CLOSED = "CLOSED"           # Arms at sides, feet together
    OPENING = "OPENING"         # Moving to open position
    OPEN = "OPEN"               # Arms overhead, feet apart (starfish)
    CLOSING = "CLOSING"         # Moving back to closed position


@dataclass
class JumpingJackConfig(RepCounterConfig):
    """
    Configuration for jumping jack counter.
    """
    # Angle thresholds for arms (shoulder-elbow-wrist angle)
    arm_extension_threshold: float = 160.0   # Arms nearly straight
    arm_abduction_threshold: float = 45.0    # Angle between arm and torso for open position
    
    # Angle thresholds for legs (hip-knee-ankle angle for straightness)
    leg_straight_threshold: float = 170.0    # Legs straight
    
    # Position thresholds for legs apart/together
    feet_apart_threshold: float = 0.15       # Normalized distance between feet for open
    feet_together_threshold: float = 0.05    # Normalized distance for closed
    
    # Arm position relative to shoulder
    arm_overhead_threshold: float = 0.1      # Arm y-coordinate relative to shoulder
    
    # Hysteresis buffer
    buffer: float = 0.1                       # Normalized buffer for positions
    
    # Smoothing settings
    angle_buffer_size: int = 3
    
    # Coordination thresholds (arms and legs should move together)
    coordination_threshold: float = 0.2       # Max timing difference between arms and legs
    
    def __post_init__(self):
        """Validate configuration."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.feet_apart_threshold <= self.feet_together_threshold:
            raise ValueError("feet_apart_threshold must be greater than feet_together_threshold")


class JumpingJackCounter(ExerciseCounter):
    """
    Jumping jack counter using finite state machine.
    
    Tracks both arm and leg positions simultaneously:
    CLOSED -> OPENING -> OPEN -> CLOSING -> CLOSED (count increments)
    
    Requires coordinated movement of all four limbs.
    """
    
    def __init__(self, config: Optional[JumpingJackConfig] = None):
        self.config = config or JumpingJackConfig()
        
        # FSM state
        self.state = JumpingJackState.IDLE
        self.count = 0
        
        # Tracking
        self._last_arm_angle = 0.0
        self._last_leg_angle = 0.0
        self._last_feet_distance = 0.0
        self._last_arm_height = 0.0
        
        self._min_arm_angle_in_rep = float('inf')
        self._max_arm_angle_in_rep = 0.0
        self._max_feet_distance_in_rep = 0.0
        
        self._rep_start_time: Optional[float] = None
        self._rep_end_time: Optional[float] = None
        self._rep_metrics: List[Dict[str, Any]] = []
        
        # Smoothing buffers
        self._arm_angle_buffer = []
        self._leg_angle_buffer = []
        self._feet_distance_buffer = []
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
        
        # Coordination tracking
        self._arm_movement_start: Optional[float] = None
        self._leg_movement_start: Optional[float] = None
        self._max_coordination_delay = 0.0
    
    @property
    def name(self) -> str:
        return "jumping_jack"
    
    @property
    def required_landmarks(self) -> list:
        return [
            # Shoulders
            'left_shoulder', 'right_shoulder',
            # Arms
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            # Hips
            'left_hip', 'right_hip',
            # Legs
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            # Feet (for distance)
            'left_foot_index', 'right_foot_index'
        ]
    
    def _calculate_arm_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """Calculate average arm angle (shoulder-elbow-wrist)."""
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
        
        return (left_angle + right_angle) / 2.0
    
    def _calculate_leg_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """Calculate average leg straightness (hip-knee-ankle)."""
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
    
    def _calculate_feet_distance(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """Calculate normalized distance between feet."""
        left_foot = landmarks.get('left_foot_index', landmarks['left_ankle'])
        right_foot = landmarks.get('right_foot_index', landmarks['right_ankle'])
        
        return calculate_distance(left_foot, right_foot)
    
    def _calculate_arm_height(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """Calculate average arm height relative to shoulders."""
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        
        # Average wrist y-coordinate
        wrist_y = (left_wrist[1] + right_wrist[1]) / 2
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Return height difference (negative means arms above shoulders)
        return shoulder_y - wrist_y
    
    def _is_arms_open(self, arm_height: float, landmarks: Dict[str, Tuple[float, float]]) -> bool:
        """Check if arms are in open position (overhead)."""
        # Arms should be above shoulders
        if arm_height < -self.config.arm_overhead_threshold:
            return True
        
        # Also check if arms are abducted (away from body)
        left_abduction = abs(landmarks['left_wrist'][0] - landmarks['left_shoulder'][0])
        right_abduction = abs(landmarks['right_wrist'][0] - landmarks['right_shoulder'][0])
        avg_abduction = (left_abduction + right_abduction) / 2
        
        return avg_abduction > self.config.arm_abduction_threshold / 100  # Convert to normalized
    
    def _check_coordination(self, arm_moving: bool, leg_moving: bool, timestamp: float) -> bool:
        """
        Check if arms and legs are moving together.
        
        Returns:
            True if coordinated, False otherwise
        """
        if arm_moving and self._arm_movement_start is None:
            self._arm_movement_start = timestamp
        
        if leg_moving and self._leg_movement_start is None:
            self._leg_movement_start = timestamp
        
        # If both started moving, check delay
        if self._arm_movement_start is not None and self._leg_movement_start is not None:
            delay = abs(self._arm_movement_start - self._leg_movement_start)
            self._max_coordination_delay = max(self._max_coordination_delay, delay)
            
            if delay > self.config.coordination_threshold:
                self._form_warnings.append("Arms and legs should move together")
                return False
        
        return True
    
    def _reset_coordination(self):
        """Reset coordination tracking for next rep."""
        self._arm_movement_start = None
        self._leg_movement_start = None
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update counter state with new frame data.
        
        Returns:
            Tuple of (count, state, arm_angle) - arm_angle as primary metric
        """
        self._total_frames += 1
        self._form_warnings = []
        
        # Reset if confidence is too low
        if confidence < self.config.min_confidence:
            if self.state != JumpingJackState.IDLE:
                self.state = JumpingJackState.IDLE
            return self.count, self.state, self._last_arm_angle
        
        # Extract required landmarks
        try:
            # Calculate all metrics
            arm_angle = self._calculate_arm_angle(landmarks)
            leg_angle = self._calculate_leg_angle(landmarks)
            feet_distance = self._calculate_feet_distance(landmarks)
            arm_height = self._calculate_arm_height(landmarks)
            
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Smooth metrics
        for buffer, value in [
            (self._arm_angle_buffer, arm_angle),
            (self._leg_angle_buffer, leg_angle),
            (self._feet_distance_buffer, feet_distance)
        ]:
            buffer.append(value)
            if len(buffer) > self.config.angle_buffer_size:
                buffer.pop(0)
        
        smoothed_arm = np.mean(self._arm_angle_buffer)
        smoothed_leg = np.mean(self._leg_angle_buffer)
        smoothed_feet = np.mean(self._feet_distance_buffer)
        
        self._last_arm_angle = smoothed_arm
        self._last_leg_angle = smoothed_leg
        self._last_feet_distance = smoothed_feet
        self._last_arm_height = arm_height
        
        # Determine if limbs are moving
        arm_moving = len(self._arm_angle_buffer) > 1 and abs(self._arm_angle_buffer[-1] - self._arm_angle_buffer[-2]) > 5
        leg_moving = len(self._leg_angle_buffer) > 1 and abs(self._leg_angle_buffer[-1] - self._leg_angle_buffer[-2]) > 5
        
        # Check coordination
        self._check_coordination(arm_moving, leg_moving, time.time())
        
        # Check if arms are straight
        if smoothed_arm < self.config.arm_extension_threshold - 20:
            self._form_warnings.append("Keep arms straight")
        
        # Check if legs are straight
        if smoothed_leg < self.config.leg_straight_threshold - 20:
            self._form_warnings.append("Keep legs straight")
        
        # Update FSM
        self._update_state(
            arm_height=arm_height,
            feet_distance=smoothed_feet,
            timestamp=time.time()
        )
        
        return self.count, self.state, smoothed_arm
    
    def _update_state(self, arm_height: float, feet_distance: float, timestamp: float) -> None:
        """
        Update FSM based on arm height and feet distance.
        """
        cfg = self.config
        arms_open = arm_height < -cfg.arm_overhead_threshold
        legs_open = feet_distance > cfg.feet_apart_threshold
        legs_closed = feet_distance < cfg.feet_together_threshold
        
        if self.state == JumpingJackState.IDLE:
            # IDLE -> CLOSED: Person standing with feet together, arms down
            if legs_closed and not arms_open:
                self.state = JumpingJackState.CLOSED
                self._total_valid_frames += 1
        
        elif self.state == JumpingJackState.CLOSED:
            # CLOSED -> OPENING: Starting to open
            if arms_open or legs_open:
                self.state = JumpingJackState.OPENING
                self._rep_start_time = timestamp
                self._min_arm_angle_in_rep = self._last_arm_angle
                self._max_arm_angle_in_rep = self._last_arm_angle
                self._max_feet_distance_in_rep = feet_distance
                self._reset_coordination()
        
        elif self.state == JumpingJackState.OPENING:
            # Update max metrics
            self._max_arm_angle_in_rep = max(self._max_arm_angle_in_rep, self._last_arm_angle)
            self._max_feet_distance_in_rep = max(self._max_feet_distance_in_rep, feet_distance)
            
            # OPENING -> OPEN: Reached full open position
            if arms_open and legs_open:
                self.state = JumpingJackState.OPEN
            
            # OPENING -> CLOSED: Aborted
            elif legs_closed and not arms_open:
                self.state = JumpingJackState.CLOSED
                self._reset_rep_tracking()
        
        elif self.state == JumpingJackState.OPEN:
            # OPEN -> CLOSING: Starting to close
            if not arms_open or not legs_open:
                self.state = JumpingJackState.CLOSING
        
        elif self.state == JumpingJackState.CLOSING:
            # Update max metrics
            self._max_arm_angle_in_rep = max(self._max_arm_angle_in_rep, self._last_arm_angle)
            
            # CLOSING -> CLOSED: Completed rep
            if legs_closed and not arms_open:
                self.count += 1
                self.state = JumpingJackState.CLOSED
                self._rep_end_time = timestamp
                self._store_rep_metrics()
                self._reset_rep_tracking()
            
            # CLOSING -> OPEN: Aborted closing
            elif arms_open and legs_open:
                self.state = JumpingJackState.OPEN
        
        self._total_valid_frames += 1
    
    def _store_rep_metrics(self) -> None:
        """Store metrics for the completed repetition."""
        if self._rep_start_time and self._rep_end_time:
            duration = self._rep_end_time - self._rep_start_time
        else:
            duration = 0.0
        
        self._rep_metrics.append({
            'rep_number': len(self._rep_metrics) + 1,
            'max_arm_angle': self._max_arm_angle_in_rep,
            'max_feet_distance': self._max_feet_distance_in_rep,
            'duration': duration,
            'max_coordination_delay': self._max_coordination_delay,
            'form_issues': self._form_warnings.copy() if self._form_warnings else []
        })
    
    def _reset_rep_tracking(self) -> None:
        """Reset tracking variables for a new rep."""
        self._min_arm_angle_in_rep = float('inf')
        self._max_arm_angle_in_rep = 0.0
        self._max_feet_distance_in_rep = 0.0
        self._rep_start_time = None
        self._rep_end_time = None
        self._reset_coordination()
    
    def reset(self) -> None:
        """Reset counter to initial state."""
        self.state = JumpingJackState.IDLE
        self.count = 0
        self._last_arm_angle = 0.0
        self._last_leg_angle = 0.0
        self._last_feet_distance = 0.0
        self._last_arm_height = 0.0
        self._arm_angle_buffer.clear()
        self._leg_angle_buffer.clear()
        self._feet_distance_buffer.clear()
        self._reset_rep_tracking()
        self._rep_metrics = []
        self._form_warnings = []
        self._total_valid_frames = 0
        self._total_frames = 0
    
    def get_progress(self) -> float:
        """
        Get normalized progress of current repetition (0.0 to 1.0).
        
        Returns:
            0.0 = closed position
            0.25 = opening
            0.5 = open position (starfish)
            0.75 = closing
            1.0 = back to closed (completed)
        """
        if self.state == JumpingJackState.IDLE or self.state == JumpingJackState.CLOSED:
            return 0.0
        elif self.state == JumpingJackState.OPENING:
            return 0.25
        elif self.state == JumpingJackState.OPEN:
            return 0.5
        elif self.state == JumpingJackState.CLOSING:
            return 0.75
        return 0.0
    
    def get_feedback(self) -> Optional[str]:
        """Get form feedback based on current state."""
        # Return form warnings first
        if self._form_warnings:
            return self._form_warnings[0]
        
        if self.state == JumpingJackState.IDLE:
            return "Stand with feet together, arms at sides"
        elif self.state == JumpingJackState.CLOSED:
            return "Jump and spread arms/legs"
        elif self.state == JumpingJackState.OPENING:
            if self._last_arm_height > -self.config.arm_overhead_threshold:
                return "Arms all the way up"
            if self._last_feet_distance < self.config.feet_apart_threshold:
                return "Feet wider apart"
            return "Keep going"
        elif self.state == JumpingJackState.OPEN:
            return "Good - now jump back to start"
        elif self.state == JumpingJackState.CLOSING:
            if self._last_feet_distance > self.config.feet_together_threshold:
                return "Feet together"
            return "Arms down"
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        if not self._rep_metrics:
            return {
                'exercise': 'jumping_jack',
                'total_reps': 0,
                'avg_duration': 0,
                'avg_max_arm_angle': 0,
                'avg_max_feet_distance': 0,
                'avg_coordination_delay': 0,
                'reps_with_form_issues': 0,
                'detection_quality': 0
            }
        
        avg_duration = np.mean([r['duration'] for r in self._rep_metrics])
        avg_arm_angle = np.mean([r['max_arm_angle'] for r in self._rep_metrics])
        avg_feet_distance = np.mean([r['max_feet_distance'] for r in self._rep_metrics])
        avg_delay = np.mean([r['max_coordination_delay'] for r in self._rep_metrics])
        reps_with_issues = sum(1 for r in self._rep_metrics if r['form_issues'])
        
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'jumping_jack',
            'total_reps': len(self._rep_metrics),
            'avg_duration': round(avg_duration, 2),
            'avg_max_arm_angle': round(avg_arm_angle, 1),
            'avg_max_feet_distance': round(avg_feet_distance, 3),
            'avg_coordination_delay': round(avg_delay, 3),
            'reps_with_form_issues': reps_with_issues,
            'detection_quality': round(detection_quality, 1)
        }
