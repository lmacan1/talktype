# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TalkType is a push-to-talk voice typing tool that works system-wide. Press F9, speak, press F9 — text is pasted into any focused window (terminals, browsers, IDEs). Uses local Whisper transcription via faster-whisper.

## Development Setup

```bash
# Linux dependencies
sudo apt install xdotool xclip portaudio19-dev

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Direct mode (loads model on startup)
python talktype.py

# Server mode (recommended for development - faster restarts)
python whisper_server.py --model base  # Terminal 1
python talktype.py --api http://localhost:8002/transcribe  # Terminal 2
```

## Architecture

Two main files:

- **talktype.py** — Main application: hotkey capture (pynput), audio recording (sounddevice), transcription, and paste simulation
- **whisper_server.py** — FastAPI server that keeps Whisper model loaded in memory

### talktype.py Flow

```
Hotkey → Recording → Stop → Transcribe → Focus original window → Paste
```

Key components:
- State machine: IDLE → RECORDING → TRANSCRIBING → IDLE
- OS-specific window management: `get_active_window()`, `focus_window()`, `is_terminal_window()`
- Smart paste: Ctrl+Shift+V for terminals, Ctrl+V for other apps
- Hallucination filtering to reject common Whisper false positives on silence

### Platform Differences

Linux uses xdotool/xclip. Windows/macOS use pyautogui. The `is_terminal_window()` function has OS-specific terminal detection to choose the correct paste shortcut.

## Testing Changes

No test suite. Manual testing workflow:
1. Run talktype.py
2. Focus a text field (terminal or browser)
3. Press F9, speak, press F9
4. Verify text appears correctly

## Troubleshooting

### "No speech detected" (error beep after recording)

**Symptom:** Start/stop beeps work, but always get error beep indicating no speech.

**Common causes:**

1. **Wrong audio input device** — PipeWire/PulseAudio default may not be your microphone
   ```bash
   # Check which device is actually capturing:
   pactl list sources short

   # Set correct input (e.g., GM300 USB mic):
   pactl set-default-source alsa_input.usb-YOUR_DEVICE_NAME
   ```

2. **Audio energy below threshold** — The `has_speech()` function in `talktype.py` rejects audio with energy < 0.01
   ```bash
   # Test your mic levels:
   python -c "
   import sounddevice as sd
   import numpy as np
   audio = sd.rec(32000, samplerate=16000, channels=1, dtype='float32')
   sd.wait()
   energy = np.sqrt(np.mean(audio ** 2))
   print(f'Energy: {energy:.4f} (threshold: 0.01)')
   "
   ```
   If energy is below 0.01, either increase mic gain or lower the threshold in code.

3. **Whisper server not running** — When using `--api` mode
   ```bash
   curl http://localhost:8002/health  # Should return JSON
   ```

### Broken venv

If you see "bad interpreter" errors, the venv has stale path references. Recreate it:
```bash
rm -rf venv && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
```
