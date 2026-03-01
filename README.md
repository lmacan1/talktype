# TalkType

**Push-to-talk voice typing for your terminal.**

Press a hotkey, speak, press again — your words appear wherever you're typing. Works with any terminal, IDE, or text field. Local transcription using [Whisper](https://github.com/openai/whisper), no cloud services required.

![Demo](assets/demo.gif)

## Why TalkType?

When you type, you self-edit and truncate. When you speak, you explain naturally and fully. TalkType bridges that gap — letting you talk to your terminal, your AI assistant, or any app, and have your words appear instantly.

Built for developers who want:
- **Voice input for CLI tools** like Claude Code, aider, or any terminal app
- **System-wide dictation** that works anywhere — terminals, IDEs, browsers
- **Local, private transcription** — your voice never leaves your machine
- **Minimal latency** — GPU-accelerated transcription in under a second

## Features

- **Push-to-talk**: Press F9 to start, speak, press F9 to stop and paste
- **Works everywhere**: Browsers, IDEs, terminals — any text field that accepts paste
- **Cross-platform**: Linux, Windows, macOS
- **Local Whisper**: Uses faster-whisper for fast, private transcription
- **API mode**: Connect to any Whisper-compatible API server
- **Smart paste**: Auto-detects terminals vs other apps (Ctrl+Shift+V vs Ctrl+V)
- **Window focus**: Remembers where you started — switch apps while speaking
- **Configurable**: Choose your hotkey, model size, and language

## Installation

### Quick Install (Linux)

```bash
git clone https://github.com/lmacan1/talktype.git && cd talktype
sudo apt install xdotool xclip portaudio19-dev
python3 -m venv venv && source venv/bin/activate
pip install -e .
talktype  # Setup wizard launches automatically
```

### Manual Install - Linux (Ubuntu/Debian)

```bash
# System dependencies
sudo apt install xdotool xclip portaudio19-dev

# Clone and install
git clone https://github.com/lmacan1/talktype.git
cd talktype
python3 -m venv venv
source venv/bin/activate
pip install -e .  # Installs 'talktype' command
```

### Windows

```powershell
git clone https://github.com/lmacan1/talktype.git
cd talktype
python -m venv venv
.\venv\Scripts\activate
pip install -e .
talktype  # Setup wizard launches automatically
```

### macOS

```bash
brew install portaudio
git clone https://github.com/lmacan1/talktype.git
cd talktype
python3 -m venv venv && source venv/bin/activate
pip install -e .
talktype  # Setup wizard launches automatically
```

## Usage

### First Run — Setup Wizard

On first launch, TalkType runs an interactive setup wizard:

```bash
talktype
```

The wizard lets you:
- Choose transcription mode (local server, cloud API, or local model)
- Set hotkeys by pressing them (not typing)
- Select Whisper model and language
- Optionally install as a system service (runs on login)

Config is saved to `~/.config/talktype/config.yaml`. Re-run anytime with `talktype --setup`.

### Basic Usage

```bash
talktype  # Uses saved config
```

1. Press **F9** to start recording (beep)
2. Speak your text
3. Press **F9** again to stop and paste (beep)
4. Your words appear in the focused window

### Recovery Hotkeys

| Key | What it does |
|-----|-------------|
| **F9** | Record / Stop & Paste |
| **F8** | Re-paste last transcription (if paste failed) |
| **F7** | Retry transcription (if API timed out) |

### Options

```bash
# Use a different model (tiny, base, small, medium, large-v3)
python talktype.py --model small

# Use a different hotkey
python talktype.py --hotkey f8

# Connect to a Whisper API server (if you have one running)
python talktype.py --api http://localhost:8002/transcribe

# Change language
python talktype.py --language es  # Spanish
```

### OpenAI-Compatible APIs

TalkType supports any OpenAI-compatible transcription API, so you can use different backends like Whisper, Parakeet, or Whisper Turbo:

```bash
# OpenAI API
python talktype.py --api https://api.openai.com/v1/audio/transcriptions --api-model whisper-1

# Groq (super fast)
python talktype.py --api https://api.groq.com/openai/v1/audio/transcriptions --api-model whisper-large-v3

# Local OpenAI-compatible server (e.g., faster-whisper-server, whisper.cpp)
python talktype.py --api http://localhost:8080/v1/audio/transcriptions --api-model whisper-1

# Any custom server
python talktype.py --api http://localhost:8002/transcribe
```

TalkType auto-detects OpenAI-compatible endpoints by URL pattern. For custom servers, it uses a simpler format that works with most Whisper APIs.

### Model Sizes

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny | ~75MB | Fastest | Basic | ~1GB |
| base | ~150MB | Fast | Good | ~1GB |
| small | ~500MB | Medium | Better | ~2GB |
| medium | ~1.5GB | Slow | Great | ~5GB |
| large-v3 | ~3GB | Slowest | Best | ~10GB |

For most use cases, `base` or `small` is the sweet spot.

## Whisper API Server (Recommended for Power Users)

For faster startup and better performance, run the included Whisper API server. The model stays loaded in memory, so TalkType connects instantly.

### Why use the server?

| Mode | Startup | Memory | Best for |
|------|---------|--------|----------|
| Direct (`talktype.py`) | ~3-5s (loads model) | Uses RAM while running | Occasional use |
| Server (`whisper_server.py`) | Instant | Server keeps model loaded | Heavy use, multiple apps |

### Running the Server

**Terminal 1 - Start the server (once):**
```bash
source venv/bin/activate
python whisper_server.py --model base

# Or with GPU and larger model:
python whisper_server.py --model large-v3 --device cuda
```

**Terminal 2 - Run TalkType:**
```bash
source venv/bin/activate
python talktype.py --api http://localhost:8002/transcribe
```

### Server Options

```bash
python whisper_server.py --help

# Examples:
python whisper_server.py --model small        # Better accuracy
python whisper_server.py --port 8080          # Different port
python whisper_server.py --device cpu         # Force CPU
python whisper_server.py --device cuda        # Force GPU

# Environment variables also work:
WHISPER_MODEL=large-v3 WHISPER_DEVICE=cuda python whisper_server.py
```

### Running Server as a Service (Linux)

```bash
cat > ~/.config/systemd/user/whisper-server.service << 'EOF'
[Unit]
Description=Whisper API Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/talktype
ExecStart=/path/to/talktype/venv/bin/python whisper_server.py --model base
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable whisper-server
systemctl --user start whisper-server
```

### API Endpoints

The server exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check server status |
| `/transcribe` | POST | Transcribe audio file |
| `/docs` | GET | Interactive web UI — test transcription right in your browser |

**Example with curl:**
```bash
curl -X POST http://localhost:8002/transcribe \
  -F "file=@audio.wav" \
  -F "language=en"
```

## Running as a Service (Linux)

The setup wizard can install TalkType as a systemd service automatically — just select "Run at startup" when prompted.

Or install manually:

```bash
# Create systemd user service
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/talktype.service << 'EOF'
[Unit]
Description=TalkType Voice Dictation
After=graphical-session.target

[Service]
Type=simple
ExecStart=/path/to/talktype/venv/bin/talktype
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
EOF

# Enable and start
systemctl --user daemon-reload
systemctl --user enable talktype
systemctl --user start talktype
```

Manage with:
```bash
systemctl --user status talktype   # Check status
systemctl --user stop talktype     # Stop
systemctl --user restart talktype  # Restart
```

## Using with Claude Code

TalkType works seamlessly with [Claude Code](https://claude.ai/code) and similar terminal AI assistants:

1. Start TalkType in a separate terminal (or as a service)
2. Focus your Claude Code terminal
3. Press F9, describe what you want, press F9
4. Your detailed voice prompt appears in Claude Code

Voice lets you elaborate naturally without self-editing — often resulting in clearer, more detailed prompts.

## Using with Browsers

TalkType works in any browser text field — it's not just for terminals:

1. Focus a text field (Google Docs, ChatGPT, Slack, email composer, etc.)
2. Press F9, speak, press F9
3. Your words appear in the browser

Since TalkType uses clipboard + standard paste (Ctrl+V / Cmd+V), it works anywhere that accepts pasted text.

## Troubleshooting

### Linux: "No module named 'pynput'"
Make sure you activated the virtual environment: `source venv/bin/activate`

### Linux: Hotkey not working
pynput requires X11. If using Wayland, either:
- Switch to X11 session
- Run with `GDK_BACKEND=x11` environment variable

### macOS: Accessibility permissions
macOS requires accessibility permissions for keyboard monitoring:
1. Go to System Preferences → Security & Privacy → Privacy → Accessibility
2. Add your terminal app (Terminal, iTerm, etc.)

### Windows: No audio input
Make sure your microphone is set as the default input device in Windows Sound settings.

### Transcription is slow
- Try a smaller model: `--model tiny` or `--model base`
- If you have an NVIDIA GPU, ensure CUDA is installed for GPU acceleration
- Consider running a separate Whisper API server and using `--api`

## How It Works

1. **Global hotkey capture** (pynput) — works even when other apps are focused
2. **Audio recording** (sounddevice) — captures from your microphone
3. **Local transcription** (faster-whisper) — Whisper running on your machine
4. **Smart paste** (pyperclip + OS-specific) — detects terminal vs other apps

```
[F9 Press] → Start Recording → [Speak] → [F9 Press] → Stop Recording
                                                            ↓
                                                    Transcribe (Whisper)
                                                            ↓
                                                    Focus Original Window
                                                            ↓
                                                    Paste Text
```

## Contributing

Contributions welcome! Some ideas:
- [ ] Voice activity detection (auto-stop on silence)
- [ ] Wayland support (wtype instead of xdotool)
- [ ] Tray icon / visual indicator
- [ ] Custom vocabulary/prompts
- [ ] Streaming transcription

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — CTranslate2-based Whisper
- [OpenAI Whisper](https://github.com/openai/whisper) — the model itself
- [pynput](https://github.com/moses-palmer/pynput) — cross-platform input monitoring
