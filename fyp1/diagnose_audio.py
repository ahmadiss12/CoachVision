#!/usr/bin/env python3
"""Diagnose Windows audio and SAPI configuration."""

import pyttsx3
import os
import subprocess

print("[DIAGNOSTIC] Checking Windows audio configuration...\n")

# 1. Check SAPI TTS engines
print("=" * 60)
print("1. SAPI TTS Engines Available:")
print("=" * 60)
engine = pyttsx3.init()
voices = engine.getProperty('voices')
print(f"Found {len(voices)} voice(s)")
for i, voice in enumerate(voices):
    print(f"  [{i}] {voice.name}")
    print(f"      -> {voice.id}")

# 2. Check current audio device
print("\n" + "=" * 60)
print("2. Current Audio Configuration:")
print("=" * 60)
try:
    result = subprocess.run(
        ['powershell', '-Command', 
         'Get-AudioDevice -List | Select-Object -First 5'],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("(Could not query audio devices)")
except Exception as e:
    print(f"(Audio device check skipped: {e})")

# 3. Check if system can produce ANY sound
print("\n" + "=" * 60)
print("3. Testing Direct Audio Output:")
print("=" * 60)
try:
    import numpy as np
    import sounddevice as sd
    
    print("sounddevice module available!")
    print(f"Default device: {sd.default.device}")
    
    # Generate a simple sine wave test tone
    sample_rate = 44100
    duration = 1  # 1 second
    frequency = 440  # A note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = 0.1 * np.sin(2 * np.pi * frequency * t)
    
    print(f"Playing test tone ({frequency}Hz for {duration}sec)...")
    sd.play(wave, sample_rate, blocking=True)
    print("[OK] Test tone finished (did you hear it?)")
    
except ImportError:
    print("sounddevice not available (NumPy/SoundDevice not installed)")
except Exception as e:
    print(f"Audio test failed: {e}")

# 4. Check system volume
print("\n" + "=" * 60)
print("4. System Volume Level:")
print("=" * 60)
try:
    result = subprocess.run(
        ['powershell', '-Command', 
         'Get-Volume | Select-Object Mute, Volume'],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0 and result.stdout.strip():
        print(result.stdout)
    else:
        print("(Could not query system volume)")
except Exception as e:
    print(f"(Volume check skipped: {e})")

print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
print("If you heard nothing from tests above:")
print("  1. Check system volume is not muted")
print("  2. Check speakers/headphones are connected")
print("  3. Try updating audio drivers")
print("  4. Restart the audio service or computer")
