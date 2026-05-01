"""
Common interface for all exercise counters.
Defines the abstract base class that all exercise counters must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List
from enum import Enum


class ExerciseCounter(ABC):
    """
    Abstract base class for all exercise counters.
    
    All exercise-specific counters (SquatCounter, PushUpCounter, etc.) must inherit
    from this class and implement all abstract methods.
    """
    
    @abstractmethod
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update the counter state with new frame data.
        
        Args:
            landmarks: Dictionary mapping joint names to (x, y) coordinates
            confidence: Overall confidence score of pose detection (0.0 to 1.0)
            
        Returns:
            Tuple containing:
                - count: Current repetition count
                - state: Current FSM state (exercise-specific enum)
                - primary_angle: The main angle being tracked (for visualization)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the counter to its initial state (count=0, state=IDLE)."""
        pass
    
    @abstractmethod
    def get_progress(self) -> float:
        """
        Get the normalized progress of the current repetition.
        
        Returns:
            Float between 0.0 and 1.0 representing how complete the current rep is.
            0.0 = just started/standing, 0.5 = bottom position, 1.0 = completed
        """
        pass
    
    @abstractmethod
    def get_feedback(self) -> Optional[str]:
        """
        Get form feedback message based on current state and angles.
        
        Returns:
            String with feedback message (e.g., "Go deeper", "Straighten back")
            or None if form is correct.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the exercise."""
        pass
    
    @property
    @abstractmethod
    def required_landmarks(self) -> List[str]:
        """
        Return list of landmark names required for this exercise.
        
        Returns:
            List of strings like ['left_hip', 'left_knee', 'left_ankle', ...]
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export counter state to dictionary for logging/saving.
        
        Returns:
            Dictionary with current state information
        """
        return {
            'exercise': self.name,
            'count': getattr(self, 'count', 0),
            'state': str(getattr(self, 'state', 'IDLE')),
            'progress': self.get_progress(),
        }


from dataclasses import dataclass

@dataclass
class RepCounterConfig:
    """
    Base configuration class for exercise counters.
    Individual exercise configs can inherit from this and add specific fields.
    """
    min_confidence: float = 0.5
    buffer_degrees: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'min_confidence': self.min_confidence,
            'buffer_degrees': self.buffer_degrees,
        }