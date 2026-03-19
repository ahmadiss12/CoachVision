"""
Wall sit counter (static hold) using finite state machine.
Tracks knee angle and back angle to ensure proper form.
Similar to plank but for lower body.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle
from .interface import ExerciseCounter, RepCounterConfig


class WallSitState(Enum):
    """Wall sit FSM states."""
    IDLE = "IDLE"           # No valid wall sit pose detected
    HOLDING = "HOLDING"     # Proper wall sit position maintained
    BROKEN = "BROKEN"       # Form broken (knees too bent/straight, back off wall)


@dataclass
class WallSitConfig(RepCounterConfig):
    """
    Configuration for wall sit form checker.
    """
    # Angle thresholds (in degrees)
    knee_angle_target: float = 90.0       # Ideal knee angle
    knee_angle_tolerance: float = 15.0    # Acceptable deviation
    
    # Back angle relative to wall (approximated by torso angle)
    back_angle_target: float = 90.0       # Back should be perpendicular to ground
    back_angle_tolerance: float = 10.0    # Acceptable lean
    
    # Hold time thresholds (in seconds)
    min_hold_time: float = 10.0           # Minimum hold for a "valid" wall sit
    good_hold_time: float = 30.0          # Time for "good" wall sit
    excellent_hold_time: float = 60.0     # Time for "excellent" wall sit
    
    # Smoothing settings
    angle_buffer_size: int = 5
    
    # Rest time after form break
    recovery_time: float = 2.0
    
    # Knee position threshold (prevent knees going over toes)
    knee_over_toe_threshold: float = 0.1  # Normalized distance
    
    def __post_init__(self):
        """Validate configuration."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.knee_angle_tolerance < 0:
            raise ValueError("knee_angle_tolerance must be non-negative")

class WallSitCounter(ExerciseCounter):
    """
    Wall sit counter for static hold exercise.
    
    Tracks:
    - Hold duration with proper form
    - Knee angle (should be ~90°)
    - Back angle (should be perpendicular to ground)
    - Knee position relative to ankles (should not pass toes)
    """
    
    def __init__(self, config: Optional[WallSitConfig] = None):
        self.config = config or WallSitConfig()
        
        # FSM state
        self.state = WallSitState.IDLE
        self.hold_count = 0  # Number of holds (sessions of continuous good form)
        
        # Timing tracking
        self._session_start_time: Optional[float] = None
        self._hold_start_time: Optional[float] = None
        self._break_start_time: Optional[float] = None
        self._total_hold_time = 0.0
        self._current_hold_duration = 0.0
        
        # Angle tracking
        self._last_knee_angle = 0.0
        self._last_back_angle = 0.0
        self._knee_history = []
        self._back_history = []
        
        # Position tracking
        self._initial_knee_pos: Optional[Tuple[float, float]] = None
        self._initial_ankle_pos: Optional[Tuple[float, float]] = None
        
        # Smoothing buffers
        self._knee_buffer = []
        self._back_buffer = []
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
        
        # Hold sessions tracking
        self._hold_sessions: List[Dict[str, Any]] = []
        
        # Best hold of session
        self._best_hold_duration = 0.0
        
        # Maximum deviations
        self._max_knee_deviation = 0.0
        self._max_back_deviation = 0.0
        self._max_knee_over_toe = 0.0
    
    @property
    def name(self) -> str:
        return "wall_sit"
    
    @property
    def required_landmarks(self) -> list:
        return [
            # Shoulders (for back angle)
            'left_shoulder', 'right_shoulder',
            # Hips
            'left_hip', 'right_hip',
            # Knees (primary)
            'left_knee', 'right_knee',
            # Ankles
            'left_ankle', 'right_ankle',
            # Feet (for knee over toe check)
            'left_foot_index', 'right_foot_index'
        ]
    
    def _calculate_knee_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """Calculate average knee angle."""
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
    
    def _calculate_back_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate back angle relative to vertical.
        Uses shoulder-hip line relative to vertical reference.
        """
        # Average shoulder and hip
        shoulder = (
            (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2,
            (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
        )
        hip = (
            (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2,
            (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
        )
        
        # Create vertical reference point (directly above hip)
        vertical_ref = (hip[0], hip[1] - 1.0)
        
        # Angle between shoulder-hip line and vertical
        angle = calculate_angle(shoulder, hip, vertical_ref)
        
        # 90° = back perpendicular to ground (parallel to wall)
        return angle
    
    def _check_knee_angle(self, knee_angle: float) -> bool:
        """Check if knee angle is within acceptable range."""
        deviation = abs(knee_angle - self.config.knee_angle_target)
        self._max_knee_deviation = max(self._max_knee_deviation, deviation)
        
        if deviation > self.config.knee_angle_tolerance:
            if knee_angle > self.config.knee_angle_target:
                self._form_warnings.append("Knees too straight - sit lower")
            else:
                self._form_warnings.append("Knees too bent - sit higher")
            return False
        return True
    
    def _check_back_angle(self, back_angle: float) -> bool:
        """Check if back is against wall properly."""
        deviation = abs(back_angle - self.config.back_angle_target)
        self._max_back_deviation = max(self._max_back_deviation, deviation)
        
        if deviation > self.config.back_angle_tolerance:
            if back_angle > self.config.back_angle_target:
                self._form_warnings.append("Lean back into the wall")
            else:
                self._form_warnings.append("Don't lean forward - back against wall")
            return False
        return True
    
    def _check_knee_over_toe(self, 
                            knee_pos: Tuple[float, float], 
                            ankle_pos: Tuple[float, float]) -> bool:
        """Check if knees extend past toes (bad form)."""
        # In normalized coordinates, knee should be behind ankle
        # For side view, we check x-coordinate difference
        if self._initial_knee_pos is None:
            self._initial_knee_pos = knee_pos
            self._initial_ankle_pos = ankle_pos
            return True
        
        # Calculate how far knee is in front of ankle
        knee_forward = knee_pos[0] - ankle_pos[0]
        self._max_knee_over_toe = max(self._max_knee_over_toe, knee_forward)
        
        if knee_forward > self.config.knee_over_toe_threshold:
            self._form_warnings.append("Knees past toes - sit back more")
            return False
        
        return True
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update counter state with new frame data.
        
        Returns:
            Tuple of (hold_count, state, knee_angle)
        """
        self._total_frames += 1
        self._form_warnings = []
        
        current_time = time.time()
        
        if confidence < self.config.min_confidence:
            if self.state != WallSitState.IDLE:
                self._end_hold_session(current_time)
                self.state = WallSitState.IDLE
            return self.hold_count, self.state, self._last_knee_angle
        
        try:
            knee_angle = self._calculate_knee_angle(landmarks)
            back_angle = self._calculate_back_angle(landmarks)
            
            # Average positions for knee-over-toe check
            knee_pos = (
                (landmarks['left_knee'][0] + landmarks['right_knee'][0]) / 2,
                (landmarks['left_knee'][1] + landmarks['right_knee'][1]) / 2
            )
            ankle_pos = (
                (landmarks['left_ankle'][0] + landmarks['right_ankle'][0]) / 2,
                (landmarks['left_ankle'][1] + landmarks['right_ankle'][1]) / 2
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Smooth angles
        self._knee_buffer.append(knee_angle)
        self._back_buffer.append(back_angle)
        
        for buffer in [self._knee_buffer, self._back_buffer]:
            if len(buffer) > self.config.angle_buffer_size:
                buffer.pop(0)
        
        smoothed_knee = np.mean(self._knee_buffer)
        smoothed_back = np.mean(self._back_buffer)
        
        self._last_knee_angle = smoothed_knee
        self._last_back_angle = smoothed_back
        
        # Track history for analysis
        self._knee_history.append(smoothed_knee)
        self._back_history.append(smoothed_back)
        if len(self._knee_history) > 30:
            self._knee_history.pop(0)
        if len(self._back_history) > 30:
            self._back_history.pop(0)
        
        # Check form
        knee_ok = self._check_knee_angle(smoothed_knee)
        back_ok = self._check_back_angle(smoothed_back)
        toe_ok = self._check_knee_over_toe(knee_pos, ankle_pos)
        
        form_ok = knee_ok and back_ok and toe_ok
        
        # Update FSM state
        self._update_state(form_ok, current_time)
        
        # Update timing if in HOLDING state
        if self.state == WallSitState.HOLDING:
            if self._hold_start_time is None:
                self._hold_start_time = current_time
            else:
                self._current_hold_duration = current_time - self._hold_start_time
                self._total_hold_time += (1.0 / 30)  # Approximate per frame
                
                if self._current_hold_duration > self._best_hold_duration:
                    self._best_hold_duration = self._current_hold_duration
        
        elif self.state == WallSitState.BROKEN:
            if self._break_start_time is None:
                self._break_start_time = current_time
        
        self._total_valid_frames += 1
        
        return self.hold_count, self.state, smoothed_knee
    
    def _update_state(self, form_ok: bool, current_time: float) -> None:
        """Update FSM based on form status."""
        
        if self.state == WallSitState.IDLE:
            # IDLE -> HOLDING: Person in good wall sit position
            if form_ok:
                self.state = WallSitState.HOLDING
                self._session_start_time = current_time
                self._hold_start_time = current_time
                self._total_valid_frames += 1
        
        elif self.state == WallSitState.HOLDING:
            if not form_ok:
                # Form broken - end this hold session
                self.state = WallSitState.BROKEN
                self._end_hold_session(current_time)
                self._break_start_time = current_time
        
        elif self.state == WallSitState.BROKEN:
            if form_ok:
                # Check if enough recovery time has passed
                if self._break_start_time:
                    recovery_time = current_time - self._break_start_time
                    if recovery_time >= self.config.recovery_time:
                        # Recovered - start new hold
                        self.state = WallSitState.HOLDING
                        self._hold_start_time = current_time
                        self._break_start_time = None
    
    def _end_hold_session(self, end_time: float) -> None:
        """End current hold session and record metrics."""
        if self._hold_start_time is not None:
            duration = end_time - self._hold_start_time
            
            # Only count holds that meet minimum duration
            if duration >= self.config.min_hold_time:
                self.hold_count += 1
                
                # Calculate average angles during hold
                avg_knee = np.mean(self._knee_history) if self._knee_history else 0
                avg_back = np.mean(self._back_history) if self._back_history else 0
                
                # Classify hold quality
                if duration >= self.config.excellent_hold_time:
                    quality = "excellent"
                elif duration >= self.config.good_hold_time:
                    quality = "good"
                else:
                    quality = "basic"
                
                # Store session data
                self._hold_sessions.append({
                    'session_number': len(self._hold_sessions) + 1,
                    'duration': duration,
                    'avg_knee_angle': avg_knee,
                    'avg_back_angle': avg_back,
                    'max_knee_deviation': self._max_knee_deviation,
                    'max_back_deviation': self._max_back_deviation,
                    'max_knee_over_toe': self._max_knee_over_toe,
                    'quality': quality,
                    'form_issues': self._form_warnings.copy() if self._form_warnings else []
                })
            
            # Reset tracking for next hold
            self._hold_start_time = None
            self._knee_history = []
            self._back_history = []
            self._max_knee_deviation = 0.0
            self._max_back_deviation = 0.0
            self._max_knee_over_toe = 0.0
            self._initial_knee_pos = None
            self._initial_ankle_pos = None
    
    def reset(self) -> None:
        """Reset counter to initial state."""
        current_time = time.time()
        
        if self.state == WallSitState.HOLDING and self._hold_start_time is not None:
            self._end_hold_session(current_time)
        
        self.state = WallSitState.IDLE
        self.hold_count = 0
        self._last_knee_angle = 0.0
        self._last_back_angle = 0.0
        self._session_start_time = None
        self._hold_start_time = None
        self._break_start_time = None
        self._total_hold_time = 0.0
        self._current_hold_duration = 0.0
        self._knee_buffer.clear()
        self._back_buffer.clear()
        self._knee_history.clear()
        self._back_history.clear()
        self._initial_knee_pos = None
        self._initial_ankle_pos = None
        self._form_warnings = []
        self._hold_sessions = []
        self._best_hold_duration = 0.0
        self._max_knee_deviation = 0.0
        self._max_back_deviation = 0.0
        self._max_knee_over_toe = 0.0
        self._total_valid_frames = 0
        self._total_frames = 0
    
    def get_progress(self) -> float:
        """
        Get normalized progress of current hold (0.0 to 1.0).
        """
        if self.state != WallSitState.HOLDING or self._hold_start_time is None:
            return 0.0
        
        current_duration = time.time() - self._hold_start_time
        
        # Progress to min hold time (0 to 0.5)
        if current_duration < self.config.min_hold_time:
            return (current_duration / self.config.min_hold_time) * 0.5
        
        # Progress from min to good (0.5 to 0.75)
        elif current_duration < self.config.good_hold_time:
            progress = (current_duration - self.config.min_hold_time) / (self.config.good_hold_time - self.config.min_hold_time)
            return 0.5 + (progress * 0.25)
        
        # Progress from good to excellent (0.75 to 1.0)
        elif current_duration < self.config.excellent_hold_time:
            progress = (current_duration - self.config.good_hold_time) / (self.config.excellent_hold_time - self.config.good_hold_time)
            return 0.75 + (progress * 0.25)
        
        return 1.0
    
    def get_feedback(self) -> Optional[str]:
        """Get form feedback based on current state."""
        if self._form_warnings:
            return self._form_warnings[0]
        
        if self.state == WallSitState.IDLE:
            return "Stand with back against wall"
        
        if self.state == WallSitState.HOLDING:
            if self._hold_start_time:
                duration = time.time() - self._hold_start_time
                
                if duration < self.config.min_hold_time:
                    remaining = self.config.min_hold_time - duration
                    return f"Hold for {remaining:.1f} more seconds"
                elif duration < self.config.good_hold_time:
                    return "Good form! Keep holding"
                elif duration < self.config.excellent_hold_time:
                    return "Excellent! Push for the full minute"
                else:
                    return "Amazing hold! You're crushing it"
            
            return None
        
        if self.state == WallSitState.BROKEN:
            return "Fix your form and hold again"
        
        return None
    
    def get_current_hold_duration(self) -> float:
        """Get duration of current hold."""
        if self.state == WallSitState.HOLDING and self._hold_start_time:
            return time.time() - self._hold_start_time
        return 0.0
    
    def get_total_hold_time(self) -> float:
        """Get total accumulated hold time for session."""
        return self._total_hold_time
    
    def get_hold_sessions(self) -> List[Dict[str, Any]]:
        """Get data for all completed hold sessions."""
        return self._hold_sessions.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        if not self._hold_sessions:
            return {
                'exercise': 'wall_sit',
                'total_holds': 0,
                'total_hold_time': round(self._total_hold_time, 1),
                'best_hold': round(self._best_hold_duration, 1),
                'avg_hold_duration': 0,
                'avg_knee_angle': 0,
                'avg_back_angle': 0,
                'max_knee_deviation': 0,
                'max_back_deviation': 0,
                'excellent_holds': 0,
                'good_holds': 0,
                'basic_holds': 0,
                'holds_with_issues': 0,
                'detection_quality': 0
            }
        
        # Count quality categories
        quality_counts = {'excellent': 0, 'good': 0, 'basic': 0}
        holds_with_issues = 0
        
        for session in self._hold_sessions:
            quality_counts[session['quality']] += 1
            if session['form_issues']:
                holds_with_issues += 1
        
        # Calculate averages
        avg_duration = np.mean([s['duration'] for s in self._hold_sessions])
        avg_knee = np.mean([s['avg_knee_angle'] for s in self._hold_sessions])
        avg_back = np.mean([s['avg_back_angle'] for s in self._hold_sessions])
        
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'wall_sit',
            'total_holds': len(self._hold_sessions),
            'total_hold_time': round(self._total_hold_time, 1),
            'best_hold': round(self._best_hold_duration, 1),
            'avg_hold_duration': round(avg_duration, 1),
            'avg_knee_angle': round(avg_knee, 1),
            'avg_back_angle': round(avg_back, 1),
            'max_knee_deviation': round(self._max_knee_deviation, 1),
            'max_back_deviation': round(self._max_back_deviation, 1),
            'excellent_holds': quality_counts['excellent'],
            'good_holds': quality_counts['good'],
            'basic_holds': quality_counts['basic'],
            'holds_with_issues': holds_with_issues,
            'detection_quality': round(detection_quality, 1)
        }