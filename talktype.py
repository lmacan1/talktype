#!/usr/bin/env python3
"""
TalkType - Push-to-talk voice typing for your terminal.

Press a hotkey, speak, press again - your words appear wherever you're typing.
Works on Linux, Windows, and macOS with local Whisper transcription.

Usage:
    python talktype.py [--api URL] [--model MODEL] [--hotkey KEY]

Examples:
    python talktype.py                          # Use faster-whisper locally
    python talktype.py --api http://localhost:8002/transcribe  # Use API
    python talktype.py --model small            # Use small model
    python talktype.py --hotkey f8              # Use F8 instead of F9
"""

import argparse
import io
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyperclip
import requests
import sounddevice as sd
from pynput import keyboard
from scipy.io import wavfile

# === Configuration ===
SAMPLE_RATE = 16000
DEFAULT_MODEL = "base"
SYSTEM = platform.system()  # "Linux", "Windows", "Darwin" (macOS)

# Terminal identifiers per OS
TERMINALS = {
    "Linux": [
        "gnome-terminal", "xterm", "konsole", "alacritty", "kitty",
        "terminator", "tilix", "xfce4-terminal", "urxvt", "st",
        "sakura", "guake", "tilda", "hyper", "wezterm"
    ],
    "Windows": [
        "WindowsTerminal", "cmd.exe", "powershell", "pwsh",
        "ConEmu", "mintty", "Hyper", "Terminus"
    ],
    "Darwin": [
        "Terminal", "iTerm", "iTerm2", "Hyper", "kitty",
        "alacritty", "wezterm"
    ]
}


# === State ===
class State:
    IDLE = 0
    RECORDING = 1
    TRANSCRIBING = 2


state = State.IDLE
state_lock = threading.Lock()
audio_chunks: list[np.ndarray] = []
stream: sd.InputStream | None = None
target_window = None
whisper_model = None
config = None


# === Argument Parsing ===
def parse_args():
    parser = argparse.ArgumentParser(
        description="Push-to-talk voice typing for your terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python talktype.py                     # Use local faster-whisper
  python talktype.py --api http://localhost:8002/transcribe
  python talktype.py --model small       # Use 'small' model for better accuracy
  python talktype.py --hotkey f8         # Use F8 instead of F9
        """
    )
    parser.add_argument(
        "--api", "-a",
        help="Whisper API URL (if not set, uses local faster-whisper)"
    )
    parser.add_argument(
        "--api-model",
        default=None,
        help="Model name for OpenAI-compatible APIs (default: whisper-1)"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Whisper model size: tiny, base, small, medium, large-v3 (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--hotkey", "-k",
        default="f9",
        help="Hotkey to use (default: f9). Examples: f8, f10, f12"
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code for transcription (default: auto-detect)"
    )
    parser.add_argument(
        "--minimal", "-M",
        action="store_true",
        help="Minimal UI - only show status (great for demos)"
    )
    return parser.parse_args()


# === Dependency Checks ===
def check_dependencies():
    """Verify system dependencies based on OS."""
    if SYSTEM == "Linux":
        missing = []
        for cmd in ("xdotool", "xclip"):
            try:
                subprocess.run(["which", cmd], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing.append(cmd)
        if missing:
            print(f"Missing Linux dependencies: {', '.join(missing)}")
            print(f"Install with: sudo apt install {' '.join(missing)}")
            sys.exit(1)

    # Check microphone
    try:
        devices = sd.query_devices()
        if not any(d['max_input_channels'] > 0 for d in devices):
            print("No microphone detected!")
            sys.exit(1)
    except Exception as e:
        print(f"Audio device error: {e}")
        sys.exit(1)


def load_whisper_model():
    """Load local Whisper model if not using API."""
    global whisper_model
    if config.api:
        # Test API connection
        try:
            health_url = config.api.rsplit('/', 1)[0] + "/health"
            resp = requests.get(health_url, timeout=2)
            info = resp.json()
            print(f"Using Whisper API: model={info.get('default_model', 'unknown')}")
        except:
            print(f"Using Whisper API: {config.api}")
    else:
        try:
            from faster_whisper import WhisperModel
            print(f"Loading Whisper model '{config.model}'... (first run downloads ~150MB)")
            whisper_model = WhisperModel(config.model, device="auto", compute_type="auto")
            print("Model loaded.")
        except ImportError:
            print("faster-whisper not installed!")
            print("Install with: pip install faster-whisper")
            print("Or use --api flag to connect to a Whisper API server")
            sys.exit(1)


# === Audio Feedback ===
def beep(freq: float, duration: float, volume: float = 0.12):
    """Play beep without blocking."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    try:
        sd.play(wave, SAMPLE_RATE)
    except:
        pass  # Ignore audio errors


def beep_start():
    beep(880, 0.08)

def beep_stop():
    beep(440, 0.12)

def beep_error():
    beep(220, 0.2)

def beep_success():
    beep(660, 0.08)


# === Terminal Title (visual status) ===
def set_terminal_title(title: str):
    """Set terminal window title for visual status."""
    # ANSI escape sequence to set terminal title
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.flush()


def show_status(status: str, detail: str = ""):
    """Show status in minimal mode (clears and centers)."""
    if not config.minimal:
        if detail:
            print(f"{status} {detail}")
        else:
            print(status)
        return

    # Clear screen and show centered status
    sys.stdout.write("\033[2J\033[H")  # Clear screen, move to top
    sys.stdout.write("\n" * 8)  # Padding from top
    sys.stdout.write(f"{'─' * 40}\n")
    sys.stdout.write(f"{status:^40}\n")
    if detail:
        # Truncate detail if too long
        detail = detail[:36] + "..." if len(detail) > 36 else detail
        sys.stdout.write(f"{detail:^40}\n")
    sys.stdout.write(f"{'─' * 40}\n")
    sys.stdout.flush()


# === Window Management (OS-specific) ===
def get_active_window():
    """Get the currently focused window identifier."""
    try:
        if SYSTEM == "Linux":
            return subprocess.check_output(
                ["xdotool", "getactivewindow"],
                stderr=subprocess.DEVNULL
            ).strip()
        elif SYSTEM == "Windows":
            import ctypes
            return ctypes.windll.user32.GetForegroundWindow()
        elif SYSTEM == "Darwin":
            script = 'tell application "System Events" to get name of first process whose frontmost is true'
            result = subprocess.check_output(["osascript", "-e", script], stderr=subprocess.DEVNULL)
            return result.strip()
    except:
        return None
    return None


def focus_window(window_id):
    """Focus a specific window."""
    if not window_id:
        return
    try:
        if SYSTEM == "Linux":
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", window_id],
                stderr=subprocess.DEVNULL
            )
        elif SYSTEM == "Windows":
            import ctypes
            ctypes.windll.user32.SetForegroundWindow(window_id)
        elif SYSTEM == "Darwin":
            # macOS: window_id is app name
            script = f'tell application "{window_id.decode()}" to activate'
            subprocess.run(["osascript", "-e", script], stderr=subprocess.DEVNULL)
    except:
        pass


def is_terminal_window(window_id) -> bool:
    """Check if the window is a terminal."""
    try:
        if SYSTEM == "Linux":
            wm_class = subprocess.check_output(
                ["xprop", "-id", window_id, "WM_CLASS"],
                stderr=subprocess.DEVNULL
            ).decode().lower()
            return any(t in wm_class for t in TERMINALS.get("Linux", []))

        elif SYSTEM == "Windows":
            import ctypes
            buffer = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetWindowTextW(window_id, buffer, 256)
            title = buffer.value.lower()
            class_buffer = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetClassNameW(window_id, class_buffer, 256)
            class_name = class_buffer.value
            return any(t.lower() in title or t.lower() in class_name.lower()
                      for t in TERMINALS.get("Windows", []))

        elif SYSTEM == "Darwin":
            # window_id is app name on macOS
            app_name = window_id.decode() if isinstance(window_id, bytes) else str(window_id)
            return any(t.lower() in app_name.lower() for t in TERMINALS.get("Darwin", []))
    except:
        pass
    return False


# === Recording ===
def audio_callback(indata, frames, time_info, status):
    """Accumulate audio chunks."""
    audio_chunks.append(indata.copy())


def start_recording():
    """Start recording from microphone."""
    global stream, audio_chunks, target_window
    target_window = get_active_window()
    audio_chunks = []
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()
    beep_start()
    set_terminal_title("🎤 RECORDING...")
    show_status("🎤 RECORDING", "Press hotkey to stop")


def stop_recording() -> np.ndarray:
    """Stop recording, return audio array."""
    global stream
    if stream:
        stream.stop()
        stream.close()
        stream = None
    beep_stop()
    set_terminal_title("⏳ Transcribing...")
    show_status("⏳ TRANSCRIBING", "Processing speech...")

    if not audio_chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(audio_chunks).flatten()


# === Transcription ===
# Common Whisper hallucinations on silence/noise
# Phrases that indicate Whisper is hallucinating on silence
HALLUCINATION_PHRASES = [
    "thanks for watching", "thank you for watching", "thanks for listening",
    "thank you for listening", "subscribe", "like and subscribe",
    "see you next time", "the end", "silence", "no speech",
    "inaudible", "[music]", "(music)",
]
# Single words that are hallucinations when they're the ENTIRE output
HALLUCINATION_WORDS = {"you", "i", "so", "uh", "um", "hmm", "huh", "ah", "oh", "bye", "goodbye"}


def is_hallucination(text: str) -> bool:
    """Check if text is likely a Whisper hallucination."""
    t = text.lower().strip()
    if len(t) < 3:
        return True
    # Check if entire text is just a hallucination word
    if t in HALLUCINATION_WORDS:
        return True
    # Check for hallucination phrases in short outputs
    if len(t) < 40:
        return any(phrase in t for phrase in HALLUCINATION_PHRASES)
    return False


def has_speech(audio: np.ndarray, threshold: float = 0.01, segment_ms: int = 50) -> bool:
    """Check if audio contains actual speech using segment-based detection.

    Instead of averaging energy over the entire recording (which dilutes
    short phrases surrounded by silence), this checks if ANY segment
    exceeds the threshold. This catches quick phrases much better.
    """
    segment_samples = int(SAMPLE_RATE * segment_ms / 1000)

    # Check each segment for speech
    for i in range(0, len(audio), segment_samples):
        segment = audio[i:i + segment_samples]
        if len(segment) < segment_samples // 2:
            continue  # Skip tiny trailing segments
        energy = np.sqrt(np.mean(segment ** 2))
        if energy > threshold:
            return True

    return False


def is_openai_api(url: str) -> bool:
    """Check if URL looks like an OpenAI-compatible API."""
    openai_patterns = ["/v1/audio/transcriptions", "/v1/audio/", "openai", "groq", "deepgram"]
    return any(p in url.lower() for p in openai_patterns)


def transcribe_api(wav_buffer: io.BytesIO) -> str:
    """Transcribe using API (supports OpenAI-compatible and custom APIs)."""
    wav_buffer.seek(0)

    if is_openai_api(config.api):
        # OpenAI-compatible API format
        files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
        data = {
            "model": config.api_model or "whisper-1",
            "language": config.language,
            "response_format": "json"
        }
    else:
        # Custom API format (e.g., local faster-whisper server)
        files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
        data = {"language": config.language}

    resp = requests.post(config.api, files=files, data=data, timeout=60)
    resp.raise_for_status()

    # Handle both JSON {"text": "..."} and plain text responses
    try:
        result = resp.json()
        return result.get("text", "").strip()
    except:
        return resp.text.strip()


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio to text."""
    if len(audio) < SAMPLE_RATE * 0.5:  # < 500ms
        return ""

    # Check if audio has enough energy (not just silence)
    if not has_speech(audio):
        return ""

    # Convert to int16 WAV
    audio_int16 = (audio * 32767).astype(np.int16)
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, SAMPLE_RATE, audio_int16)
    wav_buffer.seek(0)

    if config.api:
        return transcribe_api(wav_buffer)
    else:
        # Use local model
        wav_buffer.seek(0)
        audio_for_whisper = audio.astype(np.float32)
        segments, _ = whisper_model.transcribe(audio_for_whisper, language=config.language)
        return " ".join(seg.text for seg in segments).strip()


# === Paste ===
def paste_text(text: str):
    """Paste text into the target window."""
    # Save old clipboard
    try:
        old_clipboard = pyperclip.paste()
    except:
        old_clipboard = None

    # Set new clipboard
    pyperclip.copy(text)
    time.sleep(0.05)

    # Focus original window
    focus_window(target_window)
    time.sleep(0.05)

    # Determine paste shortcut
    is_terminal = is_terminal_window(target_window) if target_window else False

    if SYSTEM == "Linux":
        key = "ctrl+shift+v" if is_terminal else "ctrl+v"
        subprocess.run(["xdotool", "key", key], stderr=subprocess.DEVNULL)

    elif SYSTEM == "Windows":
        import pyautogui
        if is_terminal:
            # Windows Terminal and modern terminals use Ctrl+V
            pyautogui.hotkey('ctrl', 'v')
        else:
            pyautogui.hotkey('ctrl', 'v')

    elif SYSTEM == "Darwin":
        import pyautogui
        pyautogui.hotkey('command', 'v', interval=0.05)  # 50ms between keys for cold start reliability

    # Restore old clipboard
    if old_clipboard:
        def restore():
            time.sleep(0.5)
            try:
                pyperclip.copy(old_clipboard)
            except:
                pass
        threading.Thread(target=restore, daemon=True).start()


# === Main Logic ===
def transcribe_and_paste(audio: np.ndarray):
    """Background thread: transcribe and paste."""
    global state
    try:
        text = transcribe(audio)
        if text and not is_hallucination(text):
            paste_text(" " + text)  # Space to separate from previous
            beep_success()
            set_terminal_title("TalkType ✅")
            show_status("✅ DONE", text[:50])
        else:
            beep_error()
            set_terminal_title("TalkType")
            show_status("❌ NO SPEECH", "Nothing detected")
    except Exception as e:
        beep_error()
        set_terminal_title("TalkType ❌")
        show_status("❌ ERROR", str(e)[:30])
    finally:
        with state_lock:
            state = State.IDLE
        # Reset to ready after a moment
        time.sleep(1.5)
        set_terminal_title("TalkType - Ready")
        show_status("● READY", "Press F9 to record")


def get_hotkey(key_name: str):
    """Convert key name string to pynput key."""
    key_name = key_name.lower().strip()
    key_map = {
        "f1": keyboard.Key.f1, "f2": keyboard.Key.f2, "f3": keyboard.Key.f3,
        "f4": keyboard.Key.f4, "f5": keyboard.Key.f5, "f6": keyboard.Key.f6,
        "f7": keyboard.Key.f7, "f8": keyboard.Key.f8, "f9": keyboard.Key.f9,
        "f10": keyboard.Key.f10, "f11": keyboard.Key.f11, "f12": keyboard.Key.f12,
    }
    return key_map.get(key_name, keyboard.Key.f9)


def create_hotkey_handler(hotkey):
    """Create the hotkey handler function."""
    def on_press(key):
        global state
        if key != hotkey:
            return

        with state_lock:
            if state == State.IDLE:
                state = State.RECORDING
                start_recording()
            elif state == State.RECORDING:
                state = State.TRANSCRIBING
                audio = stop_recording()
                threading.Thread(
                    target=transcribe_and_paste,
                    args=(audio,),
                    daemon=True
                ).start()
            # TRANSCRIBING: ignore

    return on_press


def main():
    global config
    config = parse_args()

    print("TalkType - Voice Typing for Your Terminal")
    print("=" * 45)
    print(f"System: {SYSTEM}")

    check_dependencies()
    load_whisper_model()

    hotkey = get_hotkey(config.hotkey)
    set_terminal_title("TalkType - Ready")

    if config.minimal:
        show_status("● READY", f"Press {config.hotkey.upper()} to record")
    else:
        print(f"\nReady! Press {config.hotkey.upper()} to record.")
        print("Press Ctrl+C to exit.\n")

    handler = create_hotkey_handler(hotkey)
    with keyboard.Listener(on_press=handler) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\nBye!")


if __name__ == "__main__":
    main()
