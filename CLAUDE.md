# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TalkType is a push-to-talk voice typing tool that works system-wide. Press F9, speak, press F9 — text is pasted into any focused window (terminals, browsers, IDEs). Uses local Whisper transcription via faster-whisper.

## Development Setup

```bash
# Linux dependencies
sudo apt install xdotool xclip portaudio19-dev

# macOS dependencies
brew install portaudio

# Windows - no additional dependencies needed

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Direct mode (loads model on startup)
python talktype.py

# Server mode (recommended - faster restarts, GPU acceleration)
python whisper_server.py --model medium  # Terminal 1 (or base/small for faster)
python talktype.py --api http://localhost:8002/transcribe --language en  # Terminal 2
```

## First-Run Setup Wizard

On first launch (no config file exists), TalkType runs an interactive setup wizard:

```bash
python talktype.py  # Auto-launches wizard on first run
python talktype.py --setup  # Re-run wizard anytime
```

The wizard lets you:
- Choose API server or local model mode
- Press keys to set hotkeys (no typing "f9" — just press F9)
- Select Whisper model (tiny → large-v3)
- Set language preference

Settings are saved to `~/.config/talktype/config.yaml`. CLI flags override config file values.

## CLI Flags

**talktype.py:**
| Flag | Description |
|------|-------------|
| `--api URL` | Use external Whisper API instead of local model |
| `--model MODEL` | Whisper model: tiny, base, small, medium, large-v3 |
| `--hotkey KEY` | Hotkey to use (default: f9) |
| `--recovery-hotkey KEY` | Hotkey to recover/re-paste last transcription (default: f8) |
| `--retry-hotkey KEY` | Hotkey to retry failed transcription from saved audio (default: f7) |
| `--language CODE` | Language code (default: auto-detect) |
| `--minimal` | Minimal UI mode |
| `--history-limit N` | Max transcriptions to keep in history (default: 100) |
| `--setup` | Run setup wizard (reconfigure settings) |

**whisper_server.py:**
| Flag | Description |
|------|-------------|
| `--model MODEL` | Whisper model (default: base) |
| `--device DEVICE` | cuda, cpu, or auto |
| `--compute TYPE` | float16, int8, or auto |
| `--port PORT` | Server port (default: 8002) |
| `--timeout SECS` | Transcription timeout in seconds (default: 120) |
| `--no-vad` | Disable VAD (voice activity detection) filtering |
| `--log-level LEVEL` | Log level: DEBUG, INFO, WARNING, ERROR (default: INFO) |

### Server API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status, config, and actual CUDA device info |
| `/stats` | GET | Server metrics: request count, avg time, uptime, errors |
| `/transcribe` | POST | Transcribe audio (multipart: `file`, `language`, `model`) |
| `/docs` | GET | Interactive Swagger UI |

### OpenAI-Compatible APIs

TalkType auto-detects OpenAI-compatible endpoints by URL pattern. Use `--api-model` for custom model names:
```bash
python talktype.py --api https://api.groq.com/openai/v1/audio/transcriptions --api-model whisper-large-v3
```

## GPU Acceleration

For ~10x faster transcription on NVIDIA GPUs, install CUDA libraries:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

The server auto-detects these and configures CUDA paths. Defaults to `device=cuda` and `compute=float16`.

## Systemd Services (Linux)

For 24/7 operation, create user services for both the Whisper server and TalkType.

**~/.config/systemd/user/whisper-server.service:**
```ini
[Unit]
Description=Whisper Transcription Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/talktype
ExecStart=/path/to/talktype/venv/bin/python whisper_server.py --model medium
Restart=always
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
ExecStart=/path/to/talktype/venv/bin/python talktype.py --api http://localhost:8002/transcribe --language en
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
- Transcription history for recovery (see below)

### Transcription History & Recovery

TalkType has two recovery mechanisms:

**F8 - Re-paste last transcription:**
- Saves successful transcriptions to `~/.cache/talktype/history.jsonl`
- Press F8 to re-paste if the paste failed but transcription succeeded
- History is automatically trimmed to the last 100 entries

**F7 - Retry failed transcription:**
- Saves audio to `~/.cache/talktype/pending.wav` BEFORE transcription
- If transcription fails (API timeout, network error), press F7 to retry
- Pending audio auto-expires after 1 hour
- Deleted automatically on successful transcription

**Hotkey summary:**
| Key | Purpose | When to use |
|-----|---------|-------------|
| F9 | Record/stop | Normal operation |
| F8 | Re-paste text | Paste failed, transcription succeeded |
| F7 | Retry audio | Transcription failed (API error, timeout) |

**Clipboard race condition fix:** The clipboard restoration delay now scales with text length (1-3 seconds) to prevent the old clipboard from overwriting mid-paste on long transcriptions.

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

2. **Audio energy below threshold** — The `has_speech()` function uses segment-based detection (50ms chunks) with a 0.01 energy threshold. This handles quick phrases well, but very quiet speech may still be missed.
   ```bash
   # Test your mic levels (speak normally during recording):
   python -c "
   import sounddevice as sd
   import numpy as np
   audio = sd.rec(32000, samplerate=16000, channels=1, dtype='float32')
   sd.wait()
   # Check peak segment energy (same as has_speech uses)
   segment_size = 800  # 50ms at 16kHz
   max_energy = max(np.sqrt(np.mean(audio[i:i+segment_size]**2))
                    for i in range(0, len(audio), segment_size)
                    if len(audio[i:i+segment_size]) >= 400)
   print(f'Peak segment energy: {max_energy:.4f} (threshold: 0.01)')
   "
   ```
   If peak energy is below 0.01, increase mic gain in system settings.

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

### macOS Accessibility Permissions

macOS requires accessibility permissions for keyboard monitoring:
1. System Preferences → Security & Privacy → Privacy → Accessibility
2. Add your terminal app (Terminal, iTerm, etc.)

### Broken venv

If you see "bad interpreter" errors, the venv has stale path references. Recreate it:
```bash
rm -rf venv && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
```

### Port 8002 conflict

**Symptom:** whisper-server.service fails to start with "address already in use".

**Solution:** Check what's using the port and stop it:
```bash
# Check what's on port 8002
lsof -i :8002

# Kill the process or use a different port
python whisper_server.py --port 8003
```
