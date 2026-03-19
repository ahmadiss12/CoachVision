"""
Lunge counter using a finite state machine with hysteresis.
Tracks front knee angle to count repetitions and provide form feedback.
Handles alternating legs and checks for proper lunge form.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle, calculate_distance, calculate_midpoint
from .interface import ExerciseCounter, RepCounterConfig


class LungeState(Enum):
    """Lunge FSM states."""
    IDLE = "IDLE"           # No valid lunge pose detected
    UP = "UP"               # Standing (both knees extended)
    DESCENDING = "DESCENDING"  # In the descent phase
    DOWN = "DOWN"           # Bottom of lunge (front knee flexed)
    ASCENDING = "ASCENDING"    # In the ascent phase


class LeadingLeg(Enum):
    """Which leg is leading in the lunge."""
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"


@dataclass
class LungeConfig(RepCounterConfig):
    """
    Configuration for lunge counter.
    
    Extends base config with lunge-specific thresholds.
    """
    # Angle thresholds (in degrees)
    extension_threshold: float = 160.0  # Angle for standing position
    flexion_threshold: float = 80.0     # Angle for bottom position (front knee bent)
    
    # Hysteresis buffer (prevents false transitions)
    buffer: float = 10.0
    
    # Smoothing settings
    angle_buffer_size: int = 3
    
    # Feedback thresholds
    too_shallow_angle: float = 100.0    # Above this is too shallow
    too_deep_angle: float = 70.0        # Below this might be too deep
    
    # Knee tracking (for forward knee travel)
    knee_travel_threshold: float = 0.15  # Max forward knee travel (normalized)
    
    # Rear knee angle thresholds (should be close to ground)
    rear_knee_target: float = 90.0      # Ideal rear knee angle
    rear_knee_tolerance: float = 20.0   # Acceptable deviation
    
    # Alternating legs
    require_alternating: bool = True    # Require legs to alternate
    same_leg_timeout: float = 2.0       # Max seconds before forcing leg change
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.extension_threshold <= self.flexion_threshold:
            raise ValueError("extension_threshold must be greater than flexion_threshold")
        if self.buffer < 0:
            raise ValueError("buffer must be non-negative")

class LungeCounter(ExerciseCounter):
    """
    Lunge counter using finite state machine with dual thresholds.
    
    Tracks front knee angle through a complete lunge cycle:
    UP -> DESCENDING -> DOWN -> ASCENDING -> UP (count increments)
    
    Handles both left and right leg lunges and enforces proper form.
    """
    
    def __init__(self, config: Optional[LungeConfig] = None):
        """
        Initialize lunge counter.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or LungeConfig()
        
        # FSM state
        self.state = LungeState.IDLE
        self.count = 0
        
        # Leg tracking
        self.leading_leg = LeadingLeg.UNKNOWN
        self.last_leg = LeadingLeg.UNKNOWN
        self.last_leg_time = 0.0
        
        # Internal tracking
        self._last_front_knee_angle = 0.0
        self._last_rear_knee_angle = 0.0
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time: Optional[float] = None
        self._rep_end_time: Optional[float] = None
        self._rep_metrics: List[Dict[str, Any]] = []
        
        # Position tracking
        self._initial_foot_position: Optional[Tuple[float, float]] = None
        self._max_knee_travel = 0.0
        
        # Smoothing buffers
        self._front_knee_buffer = []
        self._rear_knee_buffer = []
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
    
    @property
    def name(self) -> str:
        """Return the name of the exercise."""
        return "lunge"
    
    @property
    def required_landmarks(self) -> list:
        """
        Return list of landmark names required for lunge.
        
        Returns:
            List of strings matching the keys expected in update() landmarks dict
        """
        return [
            # Hips
            'left_hip', 'right_hip',
            # Knees
            'left_knee', 'right_knee',
            # Ankles
            'left_ankle', 'right_ankle',
            # Feet positions (for tracking forward travel)
            'left_foot_index', 'right_foot_index'
        ]
    
    def _identify_leading_leg(self, landmarks: Dict[str, Tuple[float, float]]) -> LeadingLeg:
        """
        Identify which leg is leading based on foot position.
        
        The leading leg is the one with its foot further forward (smaller y-coordinate
        since origin is top-left in image coordinates).
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            LeadingLeg.LEFT, LeadingLeg.RIGHT, or LeadingLeg.UNKNOWN
        """
        try:
            left_foot = landmarks.get('left_foot_index', landmarks.get('left_ankle'))
            right_foot = landmarks.get('right_foot_index', landmarks.get('right_ankle'))
            
            if left_foot is None or right_foot is None:
                return LeadingLeg.UNKNOWN
            
            # In image coordinates, smaller y = higher in frame = forward relative to camera
            # Assuming user is facing the camera, forward foot will have smaller y
            if left_foot[1] < right_foot[1] - 0.05:  # Left foot is significantly forward
                return LeadingLeg.LEFT
            elif right_foot[1] < left_foot[1] - 0.05:  # Right foot is significantly forward
                return LeadingLeg.RIGHT
            else:
                return LeadingLeg.UNKNOWN
                
        except (KeyError, TypeError):
            return LeadingLeg.UNKNOWN
    
    def _check_alternating(self, current_leg: LeadingLeg, timestamp: float) -> bool:
        """
        Check if legs are alternating properly.
        
        Args:
            current_leg: Currently identified leading leg
            timestamp: Current timestamp
            
        Returns:
            True if leg alternation is valid, False otherwise
        """
        if not self.config.require_alternating:
            return True
        
        if current_leg == LeadingLeg.UNKNOWN:
            return True
        
        # First rep with identified leg
        if self.last_leg == LeadingLeg.UNKNOWN:
            self.last_leg = current_leg
            self.last_leg_time = timestamp
            return True
        
        # Same leg as last rep
        if current_leg == self.last_leg:
            # Check timeout
            if timestamp - self.last_leg_time > self.config.same_leg_timeout:
                self._form_warnings.append(f"Switch to your {self.last_leg.value.lower()} leg")
                return False
            return True
        
        # Different leg - valid alternation
        self.last_leg = current_leg
        self.last_leg_time = timestamp
        return True
    
    def _check_knee_travel(self, knee_pos: Tuple[float, float], ankle_pos: Tuple[float, float]) -> bool:
        """
        Check if knee travels too far forward past the ankle.
        
        Args:
            knee_pos: Current knee position
            ankle_pos: Current ankle position of front foot
            
        Returns:
            True if knee travel is acceptable, False otherwise
        """
        # Track initial foot position at start of descent
        if self.state == LungeState.DESCENDING and self._initial_foot_position is None:
            self._initial_foot_position = ankle_pos
        
        if self._initial_foot_position is not None:
            # Calculate how far knee has moved forward relative to foot
            # In normalized coordinates, we want knee x to be behind or over foot x
            # For forward lunge, knee should not extend past toes
            if knee_pos[0] > ankle_pos[0] + self.config.knee_travel_threshold:
                travel = knee_pos[0] - ankle_pos[0]
                self._max_knee_travel = max(self._max_knee_travel, travel)
                
                if travel > self.config.knee_travel_threshold * 2:
                    self._form_warnings.append("Knee too far forward - keep behind toes")
                    return False
                elif travel > self.config.knee_travel_threshold:
                    self._form_warnings.append("Watch your knee - don't let it pass your toes")
        
        return True
    
    def _check_rear_knee(self, rear_knee_angle: float) -> bool:
        """
        Check if rear knee is properly positioned (should be close to ground).
        
        Args:
            rear_knee_angle: Current rear knee angle
            
        Returns:
            True if rear knee position is acceptable
        """
        cfg = self.config
        target = cfg.rear_knee_target
        
        if abs(rear_knee_angle - target) > cfg.rear_knee_tolerance:
            if rear_knee_angle > target + cfg.rear_knee_tolerance:
                self._form_warnings.append("Lower your back knee toward the ground")
            else:
                self._form_warnings.append("Don't let back knee touch the ground")
            return False
        
        return True
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update counter state with new frame data.
        
        Args:
            landmarks: Dictionary containing required landmarks
            confidence: Overall confidence of pose detection
            
        Returns:
            Tuple of (count, state, front_knee_angle)
        """
        self._total_frames += 1
        self._form_warnings = []
        
        # Reset if confidence is too low
        if confidence < self.config.min_confidence:
            if self.state != LungeState.IDLE:
                self.state = LungeState.IDLE
            return self.count, self.state, self._last_front_knee_angle
        
        # Extract required landmarks
        try:
            # Hips
            left_hip = landmarks['left_hip']
            right_hip = landmarks['right_hip']
            
            # Knees
            left_knee = landmarks['left_knee']
            right_knee = landmarks['right_knee']
            
            # Ankles
            left_ankle = landmarks['left_ankle']
            right_ankle = landmarks['right_ankle']
            
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Identify which leg is leading
        current_leg = self._identify_leading_leg(landmarks)
        
        # Calculate angles based on leading leg
        if current_leg == LeadingLeg.LEFT:
            # Left leg is front
            front_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            rear_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            front_knee_pos = left_knee
            front_ankle_pos = left_ankle
        elif current_leg == LeadingLeg.RIGHT:
            # Right leg is front
            front_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            rear_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            front_knee_pos = right_knee
            front_ankle_pos = right_ankle
        else:
            # Can't determine leading leg - use minimum angle as fallback
            left_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_angle = calculate_angle(right_hip, right_knee, right_ankle)
            front_knee_angle = min(left_angle, right_angle)
            rear_knee_angle = max(left_angle, right_angle)
            front_knee_pos = left_knee if left_angle < right_angle else right_knee
            front_ankle_pos = left_ankle if left_angle < right_angle else right_ankle
        
        # Smooth angles with simple moving average
        self._front_knee_buffer.append(front_knee_angle)
        self._rear_knee_buffer.append(rear_knee_angle)
        
        if len(self._front_knee_buffer) > self.config.angle_buffer_size:
            self._front_knee_buffer.pop(0)
        if len(self._rear_knee_buffer) > self.config.angle_buffer_size:
            self._rear_knee_buffer.pop(0)
        
        smoothed_front = np.mean(self._front_knee_buffer)
        smoothed_rear = np.mean(self._rear_knee_buffer)
        
        self._last_front_knee_angle = smoothed_front
        self._last_rear_knee_angle = smoothed_rear
        
        # Check form before updating state
        timestamp = time.time()
        
        # Check leg alternation
        self._check_alternating(current_leg, timestamp)
        
        # Check knee travel (if in descending or down state)
        if self.state in [LungeState.DESCENDING, LungeState.DOWN]:
            self._check_knee_travel(front_knee_pos, front_ankle_pos)
        
        # Check rear knee position
        self._check_rear_knee(smoothed_rear)
        
        # Update FSM state
        self._update_state(smoothed_front, current_leg, timestamp)
        
        return self.count, self.state, smoothed_front
    
    def _update_state(self, angle: float, leading_leg: LeadingLeg, timestamp: float) -> None:
        """
        Update FSM based on current front knee angle.
        
        Args:
            angle: Current smoothed front knee angle
            leading_leg: Currently identified leading leg
            timestamp: Current timestamp
        """
        cfg = self.config
        
        # State transitions with hysteresis
        if self.state == LungeState.IDLE:
            # IDLE -> UP: Person standing with both legs extended
            if angle > cfg.extension_threshold:
                self.state = LungeState.UP
                self.leading_leg = leading_leg
                self._total_valid_frames += 1
        
        elif self.state == LungeState.UP:
            # UP -> DESCENDING: Start lunge (cross below extension threshold minus buffer)
            if angle < cfg.extension_threshold - cfg.buffer:
                self.state = LungeState.DESCENDING
                self.leading_leg = leading_leg
                self._rep_start_time = timestamp
                self._min_angle_in_rep = angle
                self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
                self._initial_foot_position = None
                self._max_knee_travel = 0.0
        
        elif self.state == LungeState.DESCENDING:
            # Update min angle
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # DESCENDING -> DOWN: Reached bottom position
            if angle < cfg.flexion_threshold:
                self.state = LungeState.DOWN
            
            # DESCENDING -> UP: Aborted descent (stood back up)
            elif angle > cfg.extension_threshold:
                self.state = LungeState.UP
                self._reset_rep_tracking()
            
            # Check if leading leg changed mid-rep
            if leading_leg != LeadingLeg.UNKNOWN and leading_leg != self.leading_leg:
                self._form_warnings.append("Keep same leg forward during the rep")
        
        elif self.state == LungeState.DOWN:
            # Update min angle
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # DOWN -> ASCENDING: Started ascent
            if angle > cfg.flexion_threshold + cfg.buffer:
                self.state = LungeState.ASCENDING
        
        elif self.state == LungeState.ASCENDING:
            # Update max angle
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # ASCENDING -> UP: Completed rep
            if angle > cfg.extension_threshold:
                # Check if this rep should count (alternating legs)
                should_count = self._check_alternating(self.leading_leg, timestamp)
                
                if should_count:
                    self.count += 1
                    self.state = LungeState.UP
                    self._rep_end_time = timestamp
                    self._store_rep_metrics()
                else:
                    # Don't count, but still transition
                    self.state = LungeState.UP
                
                self._reset_rep_tracking()
            
            # ASCENDING -> DOWN: User descended again without completing rep
            elif angle < cfg.flexion_threshold:
                self.state = LungeState.DOWN
        
        self._total_valid_frames += 1
    
    def _store_rep_metrics(self) -> None:
        """Store metrics for the completed repetition."""
        if self._rep_start_time and self._rep_end_time:
            duration = self._rep_end_time - self._rep_start_time
        else:
            duration = 0.0
        
        self._rep_metrics.append({
            'rep_number': len(self._rep_metrics) + 1,
            'leading_leg': self.leading_leg.value,
            'min_front_knee_angle': self._min_angle_in_rep,
            'max_front_knee_angle': self._max_angle_in_rep,
            'range_of_motion': self._max_angle_in_rep - self._min_angle_in_rep,
            'avg_rear_knee_angle': self._last_rear_knee_angle,
            'duration': duration,
            'max_knee_travel': self._max_knee_travel,
            'depth_quality': self._classify_depth_quality(),
            'form_issues': self._form_warnings.copy() if self._form_warnings else []
        })
    
    def _classify_depth_quality(self) -> str:
        """Classify the quality of lunge depth."""
        if self._min_angle_in_rep <= self.config.flexion_threshold:
            return "good"
        elif self._min_angle_in_rep <= self.config.too_shallow_angle:
            return "shallow"
        else:
            return "very_shallow"
    
    def _reset_rep_tracking(self) -> None:
        """Reset tracking variables for a new rep."""
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time = None
        self._rep_end_time = None
        self._initial_foot_position = None
        self._max_knee_travel = 0.0
    
    def reset(self) -> None:
        """Reset counter to initial state."""
        self.state = LungeState.IDLE
        self.count = 0
        self.leading_leg = LeadingLeg.UNKNOWN
        self.last_leg = LeadingLeg.UNKNOWN
        self.last_leg_time = 0.0
        self._last_front_knee_angle = 0.0
        self._last_rear_knee_angle = 0.0
        self._front_knee_buffer.clear()
        self._rear_knee_buffer.clear()
        self._reset_rep_tracking()
        self._rep_metrics = []
        self._form_warnings = []
        self._total_valid_frames = 0
        self._total_frames = 0
    
    def get_progress(self) -> float:
        """
        Get normalized progress of current repetition (0.0 to 1.0).
        
        Returns:
            0.0 = standing / just started
            0.25 = descending
            0.5 = at bottom
            0.75 = ascending
            1.0 = completed (but returns 0.0 after completion)
        """
        cfg = self.config
        angle = self._last_front_knee_angle
        
        if self.state == LungeState.IDLE or self.state == LungeState.UP:
            return 0.0
        
        elif self.state == LungeState.DESCENDING:
            # Progress from 0.0 to 0.5 as angle decreases
            span = cfg.extension_threshold - cfg.flexion_threshold
            if span <= 0:
                return 0.25
            prog = (cfg.extension_threshold - angle) / span
            return min(0.5, max(0.0, prog * 0.5))
        
        elif self.state == LungeState.DOWN:
            return 0.5
        
        elif self.state == LungeState.ASCENDING:
            # Progress from 0.5 to 1.0 as angle increases
            span = cfg.extension_threshold - cfg.flexion_threshold
            if span <= 0:
                return 0.75
            prog = (angle - cfg.flexion_threshold) / span
            return min(1.0, max(0.5, 0.5 + prog * 0.5))
        
        return 0.0
    
    def get_feedback(self) -> Optional[str]:
        """
        Get form feedback based on current state and angles.
        
        Returns:
            String with feedback message or None if form is correct
        """
        # Return form warnings first (higher priority)
        if self._form_warnings:
            return self._form_warnings[0]
        
        angle = self._last_front_knee_angle
        cfg = self.config
        
        # No feedback in IDLE state
        if self.state == LungeState.IDLE:
            if self.leading_leg == LeadingLeg.UNKNOWN:
                return "Step forward into lunge position"
            return "Stand tall to begin"
        
        # Feedback for UP state
        if self.state == LungeState.UP:
            leg = self.leading_leg.value.lower() if self.leading_leg != LeadingLeg.UNKNOWN else "front"
            return f"Lower into {leg} leg lunge"
        
        # Feedback for DESCENDING
        if self.state == LungeState.DESCENDING:
            if angle < cfg.flexion_threshold + 15:
                return "Almost there - go down further"
            elif angle > cfg.extension_threshold - 20:
                return "Bend your front knee more"
            return "Control your descent"
        
        # Feedback for DOWN
        if self.state == LungeState.DOWN:
            if angle > cfg.flexion_threshold + cfg.buffer:
                return "Go a bit deeper"
            elif angle < cfg.too_deep_angle:
                return "Don't let back knee touch the ground"
            return "Good depth - now push up"
        
        # Feedback for ASCENDING
        if self.state == LungeState.ASCENDING:
            if angle < cfg.flexion_threshold + 20:
                return "Push through your front heel"
            elif angle > cfg.extension_threshold - 15:
                return "Almost up - fully extend"
            return "Drive upward"
        
        return None
    
    def get_rep_metrics(self) -> List[Dict[str, Any]]:
        """
        Get metrics for all completed repetitions.
        
        Returns:
            List of dictionaries with per-rep metrics
        """
        return self._rep_metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the session.
        
        Returns:
            Dictionary with session summary
        """
        if not self._rep_metrics:
            return {
                'exercise': 'lunge',
                'total_reps': 0,
                'left_leg_reps': 0,
                'right_leg_reps': 0,
                'avg_range_of_motion': 0,
                'avg_duration': 0,
                'good_reps': 0,
                'shallow_reps': 0,
                'very_shallow_reps': 0,
                'reps_with_form_issues': 0,
                'detection_quality': 0
            }
        
        # Count by leg
        left_reps = sum(1 for r in self._rep_metrics if r['leading_leg'] == 'LEFT')
        right_reps = sum(1 for r in self._rep_metrics if r['leading_leg'] == 'RIGHT')
        
        # Count quality categories
        quality_counts = {'good': 0, 'shallow': 0, 'very_shallow': 0}
        reps_with_issues = 0
        
        for rep in self._rep_metrics:
            quality_counts[rep['depth_quality']] += 1
            if rep['form_issues']:
                reps_with_issues += 1
        
        # Calculate averages
        avg_rom = np.mean([r['range_of_motion'] for r in self._rep_metrics])
        avg_duration = np.mean([r['duration'] for r in self._rep_metrics])
        avg_knee_travel = np.mean([r['max_knee_travel'] for r in self._rep_metrics])
        
        # Detection quality (percentage of frames with valid pose)
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'lunge',
            'total_reps': len(self._rep_metrics),
            'left_leg_reps': left_reps,
            'right_leg_reps': right_reps,
            'avg_range_of_motion': round(avg_rom, 1),
            'avg_duration': round(avg_duration, 2),
            'avg_knee_travel': round(avg_knee_travel, 3),
            'good_reps': quality_counts['good'],
            'shallow_reps': quality_counts['shallow'],
            'very_shallow_reps': quality_counts['very_shallow'],
            'reps_with_form_issues': reps_with_issues,
            'detection_quality': round(detection_quality, 1)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export counter state to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'leading_leg': self.leading_leg.value,
            'last_leg': self.last_leg.value,
            'last_front_knee_angle': self._last_front_knee_angle,
            'last_rear_knee_angle': self._last_rear_knee_angle,
            'min_angle_in_rep': self._min_angle_in_rep if self._min_angle_in_rep != float('inf') else None,
            'max_angle_in_rep': self._max_angle_in_rep,
            'max_knee_travel': self._max_knee_travel,
            'form_warnings': self._form_warnings,
            'config': self.config.to_dict(),
            'rep_metrics': self._rep_metrics
        })
        return base_dict