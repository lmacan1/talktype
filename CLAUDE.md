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

## CLI Flags

**talktype.py:**
| Flag | Description |
|------|-------------|
| `--api URL` | Use external Whisper API instead of local model |
| `--model MODEL` | Whisper model: tiny, base, small, medium, large-v3 |
| `--hotkey KEY` | Hotkey to use (default: f9) |
| `--language CODE` | Language code (default: auto-detect) |
| `--minimal` | Minimal UI mode |

**whisper_server.py:**
| Flag | Description |
|------|-------------|
| `--model MODEL` | Whisper model (default: base) |
| `--device DEVICE` | cuda, cpu, or auto |
| `--compute TYPE` | float16, int8, or auto |
| `--port PORT` | Server port (default: 8002) |

### Server API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status and config |
| `/transcribe` | POST | Transcribe audio (multipart: `file`, `language`, `model`) |
| `/docs` | GET | Interactive Swagger UI |

## GPU Acceleration

For ~10x faster transcription on NVIDIA GPUs, install CUDA libraries:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

The server auto-detects these and configures CUDA paths. Defaults to `device=cuda` and `compute=float16`.

## Systemd Services (Linux)

For 24/7 operation, create user services:

**~/.config/systemd/user/whisper-server.service:**
```ini
[Unit]
Description=Whisper Transcription Server (GPU)
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=/path/to/talktype
ExecStart=/path/to/talktype/venv/bin/python whisper_server.py --model base
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

**~/.config/systemd/user/voice-dictation.service:**
```ini
[Unit]
Description=TalkType Voice Dictation
After=graphical-session.target whisper-server.service
Requires=whisper-server.service

[Service]
Type=simple
WorkingDirectory=/path/to/talktype
ExecStart=/path/to/talktype/venv/bin/python talktype.py --api http://localhost:8002/transcribe
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

[Install]
WantedBy=default.target
```

Enable and start:
```bash
systemctl --user daemon-reload
systemctl --user enable whisper-server.service voice-dictation.service
systemctl --user start whisper-server.service voice-dictation.service
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

### Low mic gain (standalone USB mics)

Some USB mics (like GM300) need gain boost beyond 100%:
```bash
# Boost to 200%
pactl set-source-volume YOUR_SOURCE_NAME 200%
```

### Wayland

pynput requires X11. On Wayland, run with `GDK_BACKEND=x11` or switch to X11 session.

### Broken venv

If you see "bad interpreter" errors, the venv has stale path references. Recreate it:
```bash
rm -rf venv && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
```
