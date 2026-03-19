# Voice Feedback System - Integration Guide

## Overview

The voice feedback system provides real-time audio coaching for all 12 exercises in the system. It delivers:
- **Instructional cues** - How to perform the exercise
- **Form corrections** - When technique deviates  
- **Encouragement** - Progress updates and positive reinforcement
- **Milestones** - Time-based achievements for static exercises

## Features

✅ **12 Exercises Supported**
- Squat, Pushup, Lunge, Deadlift, Plank
- Bicep Curl, Shoulder Press, Sit-up
- Jumping Jack, High Knees, Mountain Climber, Wall Sit

✅ **3 Difficulty Levels**
- Beginner: Encouraging, detailed cues
- Intermediate: Standard coaching
- Advanced: Brief, technical feedback

✅ **Non-Blocking Speech**
- Speech runs in background thread
- Doesn't interfere with video processing

✅ **Intelligent Cooldowns**
- Prevents message spam
- Each cue type has independent cooldown
- Corrective feedback (1.5-2s) vs Encouragement (3-5s)

✅ **Priority Management**
- Critical corrections take precedence
- Smooth user experience

## Installation

### Prerequisites
```bash
pip install pyttsx3  # For offline text-to-speech
```

For online/higher quality speech (optional):
```bash
pip install gTTS      # Google Text-to-Speech
pip install edge-tts  # Microsoft Edge TTS
```

## Usage

### Basic Usage
```bash
python main.py --exercise squat --difficulty beginner --video video.mp4
```

### Keyboard Controls in Application
- **q** - Quit application
- **r** - Reset rep counter
- **m** - Cycle display mode (full/minimal/debug)
- **v** - Toggle voice feedback on/off

### Command Line Arguments
```bash
# List all available exercises
python main.py --help

# Run specific exercise
python main.py --exercise pushup --difficulty advanced

# Use webcam instead of video
python main.py --exercise plank --camera 0

# Save output video
python main.py --exercise lunge --save

# Change display mode
python main.py --exercise deadlift --display debug
```

## Architecture

### Component Structure
```
backend/voice/
├── __init__.py                 # Public API
├── voice_engine.py             # Low-level TTS engine
├── feedback_config.py          # Exercise cue database
├── feedback_manager.py         # Feedback orchestration
└── voice_integration.py        # Exercise-specific integration
```

### Data Flow
```
Exercise Counter State
       ↓
VoiceFeedbackIntegration.update()
       ↓
VoiceFeedbackManager.trigger_feedback()
       ↓
VoiceEngine.say()
       ↓
Background Thread → TTS Output → Audio
```

## Voice Feedback Triggers

### 1. State Transitions
Triggered when exercise state changes (e.g., entering DOWN state in squat)

**Example Cues:**
- Beginner: "Bend your knees and push your hips back"
- Advanced: "Control the descent"

### 2. Form Corrections
Triggered when angles/positions violate thresholds

**Example Cues:**
- "Go deeper" (shallow depth warning)
- "Keep your back straight" (form deviation)
- "Don't go too low" (excessive depth)

### 3. Rep/Hold Completion
Triggered after successful repetition or hold

**Example Cues:**
- "Great rep! Ready for the next one"
- "Solid lift"

### 4. Time Milestones
Triggered at time intervals for static exercises (Plank, Wall Sit)

**Example Cues:**
- "15 seconds, keep going"
- "30 seconds, you're doing great"

## Configuration

### Voice Settings

Configure in your application code:
```python
from backend.voice import get_voice_integration

# Create integration for squat at beginner level
voice_integration = get_voice_integration('squat', 'beginner')

# Adjust volume (0.0 = silent, 1.0 = max)
voice_integration.set_volume(0.8)

# Adjust speech rate (50-300 words per minute)
voice_integration.set_rate(150)
```

### Difficulty-Specific Phrases

Each cue has different phrases for each difficulty level:

```python
# Example: Squat shallow depth correction
{
    'shallow_depth': {
        'beginner':     'Go a little deeper',
        'intermediate': 'Go deeper',
        'advanced':     'Hit parallel',
    }
}
```

## Advanced Usage

### Custom Feedback Integration

Extend for with-custom exercise-specific logic:

```python
from backend.voice import VoiceFeedbackIntegration
from enum import Enum

class CustomExerciseVoice(VoiceFeedbackIntegration):
    def _check_angle_corrections(self, angle: float) -> None:
        """Custom angle-based corrections."""
        if angle < your_threshold:
            self.trigger_correction('your_custom_cue_id')
```

### Manual Feedback Triggering

```python
# Trigger specific cue manually
voice_feedback.trigger_custom_feedback('good_depth')

# Trigger correction
voice_feedback.trigger_correction('back_sagging')

# Trigger milestone
voice_feedback.trigger_milestone('milestone_30')
```

### Get Available Cues

```python
# List all available cues for current exercise
cues = voice_feedback.get_available_cues()
for cue_id, phrase in cues.items():
    print(f"{cue_id}: {phrase}")
```

## Exercise-Specific Cue Examples

### Squat
| Trigger | Beginner | Advanced |
|---------|----------|----------|
| Start UP | "Stand tall, ready to squat" | "Set your stance" |
| Start DOWN | "Bend your knees and push hips back" | "Control the descent" |
| Shallow | "Go a little deeper" | "Hit parallel" |
| Complete | "Great rep!" | "Good" |

### Push-up
| Trigger | Beginner | Advanced |
|---------|----------|----------|
| Start | "Get into plank position" | "Set your plank" |
| Descending | "Lower your chest to the ground" | "Lower down" |
| Back Sag | "Keep your back straight!" | "Straight back" |
| Complete | "Great push-up!" | "Good" |

### Plank
| Trigger | Beginner | Advanced |
|---------|----------|----------|
| Start | "Hold this position, keep body straight" | "Position set" |
| Hips Sag | "Lift your hips up!" | "Hips up" |
| 15s Mark | "15 seconds, keep going" | "Nice" |
| 30s Mark | "30 seconds, you're doing great" | "Solid" |

## Troubleshooting

### No Voice Output
1. **Check if pyttsx3 is installed:**
   ```bash
   python -c "import pyttsx3; print('OK')"
   ```

2. **If not installed:**
   ```bash
   pip install pyttsx3
   ```

3. **Check system audio:**
   - Ensure speakers/headphones are connected
   - Check volume is not muted
   - Test with: `python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('test'); engine.runAndWait()"`

### Speech Too Fast/Slow
Adjust speech rate in main.py:
```python
voice_feedback.set_volume(0.8)  # 0.0-1.0
voice_feedback.set_rate(150)    # 50-300 WPM
```

### Messages Repeating
- Cooldown period is working correctly
- Messages have minimum 1.5-5 seconds between repeats depending on type
- This prevents audio spam during exercise

## Performance Impact

- **CPU**: ~2-5% overhead (speech processing in background thread)
- **Memory**: ~20-30 MB (pyttsx3 + queued messages)
- **Video Processing**: No impact (non-blocking design)

## Testing

Run the comprehensive test suite:
```bash
python test_voice_system.py
```

This validates:
- Voice engine functionality
- Configuration for all exercises
- Feedback manager behavior
- Integration with counter states
- All 12 exercise support

## Future Enhancements

Possible improvements for Phase 2:
1. **Pre-recorded audio** - Lower latency if TTS is slow
2. **Rhythm beeps** - Cadence guidance for dynamic exercises
3. **Speech recognition** - User voice commands
4. **Custom voice profiles** - Save/load preferences
5. **Language support** - Multi-language cues
6. **Emotion analysis** - Adjust coaching based on user performance

## API Reference

### VoiceEngine
```python
engine = VoiceEngine(rate=150, volume=0.8, enabled=True)
engine.say("Text to speak")
engine.set_volume(0.5)  # 0.0-1.0
engine.set_rate(200)     # 50-300
engine.stop()
```

### VoiceFeedbackManager
```python
manager = VoiceFeedbackManager('squat', 'beginner')
manager.trigger_feedback('cue_id')
manager.trigger_state_feedback(new_state, old_state)
manager.trigger_rep_complete_feedback(count)
manager.trigger_correction_feedback('correction_type')
manager.list_cues()
manager.get_stats()
manager.stop()
```

### VoiceFeedbackIntegration
```python
integration = get_voice_integration('squat', 'beginner')
integration.update(count, state, angle, feedback_text)
integration.trigger_custom_feedback('cue_id')
integration.trigger_correction('type')
integration.set_volume(0.8)
integration.set_rate(150)
integration.get_available_cues()
integration.stop()
```

## Support

For issues or questions:
1. Check the test suite: `python test_voice_system.py`
2. Review feedback configuration: `backend/voice/feedback_config.py`
3. Check integration layer: `backend/voice/voice_integration.py`
4. Consult pyttsx3 documentation for TTS issues

---

**Version**: 1.0  
**Last Updated**: March 2026  
**Status**: Production Ready ✅
