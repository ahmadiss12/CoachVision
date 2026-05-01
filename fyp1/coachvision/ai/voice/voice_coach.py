"""
VoiceCoach: single-slot, latest-wins playback. No queue, no backlog.
Stops immediately on stop(); uses pre-generated audio for low latency.
"""

import threading
import time
import tempfile
import os
import asyncio
from typing import Optional, Dict

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

from .feedback_policy import FeedbackLabel
from .phrases import get_phrase


class VoiceCoach:
    """
    One current request at a time. New say() overwrites pending; playback can be stopped on exit.
    Uses pre-generated audio (edge-tts at startup) then only plays from cache.
    """

    VOICE = "en-US-AriaNeural"
    RATE = "+0%"

    def __init__(self, exercise_name: str, difficulty: str, enabled: bool = True):
        self.exercise_name = exercise_name.lower()
        self.difficulty = difficulty.lower()
        self.enabled = enabled and HAS_EDGE_TTS and HAS_AUDIO
        self.volume = 0.8
        self._lock = threading.Lock()
        self._current_request: Optional[str] = None  # label value
        self._running = True
        self._temp_dir = tempfile.mkdtemp(prefix="fyp_voice_")
        self._cache: Dict[str, str] = {}  # phrase text -> path
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        if not self.enabled:
            print("[WARNING] Voice coach disabled (edge-tts or sounddevice not available)")
        else:
            print("[OK] Voice coach initialized (single-slot, no queue)")

    def pregenerate(self) -> None:
        """Generate audio for all phrases used by this exercise/difficulty."""
        if not self.enabled:
            return
        for label in FeedbackLabel:
            if label == FeedbackLabel.NONE:
                continue
            text = get_phrase(self.exercise_name, self.difficulty, label)
            if not text or text.lower() in self._cache:
                continue
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                path = os.path.join(self._temp_dir, f"{label.value}.mp3")
                async def gen():
                    c = edge_tts.Communicate(text, self.VOICE, rate=self.RATE)
                    await c.save(path)
                loop.run_until_complete(gen())
                loop.close()
                self._cache[text.lower()] = path
            except Exception as e:
                print(f"[VOICE] Skip pregen {label.value}: {e}")
        print(f"[OK] Voice pre-generated {len(self._cache)} clips")

    def say(self, label: FeedbackLabel) -> None:
        """Set current request to this label (overwrites any pending). Latest wins."""
        if not self.enabled or label == FeedbackLabel.NONE:
            return
        with self._lock:
            self._current_request = label.value

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))

    def _run_worker(self) -> None:
        while self._running:
            with self._lock:
                req = self._current_request
                self._current_request = None
            if req and self.enabled and HAS_AUDIO:
                text = get_phrase(self.exercise_name, self.difficulty, FeedbackLabel(req))
                if not text:
                    time.sleep(0.05)
                    continue
                path = self._cache.get(text.lower())
                if not path or not os.path.exists(path):
                    time.sleep(0.05)
                    continue
                try:
                    data, sr = sf.read(path)
                    data = (data * self.volume).astype(data.dtype)
                    sd.play(data, sr, blocking=False)
                    duration = len(data) / float(sr)
                    deadline = time.time() + duration
                    while time.time() < deadline and self._running:
                        with self._lock:
                            if self._current_request is not None:
                                sd.stop()
                                break
                        time.sleep(0.03)
                    try:
                        sd.wait()
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[VOICE] Play error: {e}")
            else:
                time.sleep(0.05)

    def stop(self) -> None:
        """Stop immediately: no waiting for queue or playback."""
        self._running = False
        with self._lock:
            self._current_request = None
        try:
            sd.stop()
        except Exception:
            pass
        self._worker.join(timeout=1.0)
        try:
            import shutil
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
        except Exception:
            pass
        print("[OK] Voice coach stopped")
