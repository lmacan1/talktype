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
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyperclip
import requests
import sounddevice as sd
from pynput import keyboard
from scipy.io import wavfile
import yaml

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
history = None  # TranscriptionHistory instance


class TranscriptionHistory:
    """Persists transcriptions to ~/.cache/talktype/history.jsonl for recovery."""

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.cache_dir = Path.home() / ".cache" / "talktype"
        self.history_file = self.cache_dir / "history.jsonl"
        self.pending_audio = self.cache_dir / "pending.wav"
        self._last: dict | None = None
        self._ensure_dir()

    def _ensure_dir(self):
        """Create cache directory if needed."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def add(self, text: str):
        """Add transcription to history."""
        entry = {"timestamp": datetime.now().isoformat(), "text": text}
        self._last = entry
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self._maybe_trim()
        except Exception:
            pass  # Don't crash on history write failure

    def get_last(self) -> str | None:
        """Get last transcription text."""
        if self._last:
            return self._last["text"]
        # Fallback: read from file
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    return json.loads(lines[-1])["text"]
        except Exception:
            pass
        return None

    def _maybe_trim(self):
        """Trim history file if it exceeds max_entries."""
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > self.max_entries * 1.5:  # Only trim when 50% over
                with open(self.history_file, "w", encoding="utf-8") as f:
                    f.writelines(lines[-self.max_entries:])
        except Exception:
            pass

    def save_pending_audio(self, wav_buffer: io.BytesIO):
        """Save audio before transcription attempt."""
        try:
            wav_buffer.seek(0)
            with open(self.pending_audio, "wb") as f:
                f.write(wav_buffer.read())
            wav_buffer.seek(0)  # Reset for transcription
        except Exception:
            pass

    def clear_pending_audio(self):
        """Delete pending audio after successful transcription."""
        try:
            self.pending_audio.unlink(missing_ok=True)
        except Exception:
            pass

    def get_pending_audio(self) -> bytes | None:
        """Get pending audio for retry (expires after 1 hour)."""
        try:
            if self.pending_audio.exists():
                # Check if not too old (1 hour max)
                age = time.time() - self.pending_audio.stat().st_mtime
                if age > 3600:
                    self.clear_pending_audio()
                    return None
                return self.pending_audio.read_bytes()
        except Exception:
            pass
        return None


# === Config File Loading ===
CONFIG_PATH = Path.home() / ".config" / "talktype" / "config.yaml"


def load_config_file() -> dict:
    """Load config from YAML file if it exists."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


# === Argument Parsing ===
def parse_args():
    # Load config file first (CLI args will override)
    file_config = load_config_file()
    hotkeys = file_config.get("hotkeys", {})
    trans = file_config.get("transcription", {})
    ui = file_config.get("ui", {})
    hist = file_config.get("history", {})

    # Determine API default from config
    api_default = None
    if trans.get("mode") == "api":
        api_default = trans.get("api_url")

    parser = argparse.ArgumentParser(
        description="Push-to-talk voice typing for your terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python talktype.py                     # Use local faster-whisper
  python talktype.py --api http://localhost:8002/transcribe
  python talktype.py --model small       # Use 'small' model for better accuracy
  python talktype.py --hotkey f8         # Use F8 instead of F9
  python talktype.py --setup             # Run setup wizard
        """
    )
    parser.add_argument(
        "--api", "-a",
        default=api_default,
        help="Whisper API URL (if not set, uses local faster-whisper)"
    )
    parser.add_argument(
        "--api-model",
        default=None,
        help="Model name for OpenAI-compatible APIs (default: whisper-1)"
    )
    parser.add_argument(
        "--model", "-m",
        default=trans.get("model", DEFAULT_MODEL),
        help=f"Whisper model size: tiny, base, small, medium, large-v3 (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--hotkey", "-k",
        default=hotkeys.get("record", "f9"),
        help="Hotkey to use (default: f9). Examples: f8, f10, f12"
    )
    parser.add_argument(
        "--language", "-l",
        default=trans.get("language"),
        help="Language code for transcription (default: auto-detect)"
    )
    parser.add_argument(
        "--minimal", "-M",
        action="store_true",
        default=ui.get("minimal", False),
        help="Minimal UI - only show status (great for demos)"
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=hist.get("limit", 100),
        help="Maximum transcriptions to keep in history (default: 100)"
    )
    parser.add_argument(
        "--recovery-hotkey",
        default=hotkeys.get("recovery", "f8"),
        help="Hotkey to recover/re-paste last transcription (default: f8)"
    )
    parser.add_argument(
        "--retry-hotkey",
        default=hotkeys.get("retry", "f7"),
        help="Hotkey to retry failed transcription from saved audio (default: f7)"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run setup wizard (reconfigure settings)"
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

    # Save pending audio BEFORE transcription (for retry on failure)
    if history:
        history.save_pending_audio(wav_buffer)

    wav_buffer.seek(0)

    if config.api:
        text = transcribe_api(wav_buffer)
    else:
        # Use local model
        wav_buffer.seek(0)
        audio_for_whisper = audio.astype(np.float32)
        segments, _ = whisper_model.transcribe(audio_for_whisper, language=config.language)
        text = " ".join(seg.text for seg in segments).strip()

    # Clear pending audio on success
    if history and text:
        history.clear_pending_audio()

    return text


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

    # Restore old clipboard (scale delay by text length to avoid race condition)
    if old_clipboard:
        def restore():
            # Base 1.0s + 10ms per 100 chars, capped at 3.0s
            delay = min(3.0, max(1.0, 1.0 + len(text) * 0.0001))
            time.sleep(delay)
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
            # Save to history for recovery
            if history:
                history.add(text)
            beep_success()
            set_terminal_title("TalkType ✅")
            show_status("✅ DONE", text[:50])
        else:
            beep_error()
            set_terminal_title("TalkType")
            show_status("❌ NO SPEECH", "Nothing detected")
            # Clear pending audio - no point retrying silence
            if history:
                history.clear_pending_audio()
    except Exception as e:
        beep_error()
        set_terminal_title("TalkType ❌")
        show_status("❌ FAILED", "F7 to retry")
        # Keep pending audio for retry - don't clear it
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


def create_recovery_handler(recovery_key):
    """Create the recovery hotkey handler (re-paste last transcription)."""
    def on_press(key):
        if key != recovery_key:
            return

        with state_lock:
            if state != State.IDLE:
                return  # Only recover when idle

        if not history:
            beep_error()
            return

        last_text = history.get_last()
        if not last_text:
            beep_error()
            show_status("❌ NO HISTORY", "Nothing to recover")
            return

        # Re-paste the last transcription
        paste_text(" " + last_text)
        beep_success()
        set_terminal_title("TalkType ↩️")
        show_status("↩️ RECOVERED", last_text[:50])

    return on_press


def create_retry_handler(retry_key):
    """Create the retry hotkey handler (re-transcribe from saved audio)."""
    def on_press(key):
        if key != retry_key:
            return

        with state_lock:
            if state != State.IDLE:
                return  # Only retry when idle

        if not history:
            beep_error()
            return

        pending = history.get_pending_audio()
        if not pending:
            beep_error()
            show_status("❌ NO PENDING", "Nothing to retry")
            return

        # Re-transcribe from saved WAV
        set_terminal_title("TalkType 🔄")
        show_status("🔄 RETRYING", "Re-transcribing...")

        try:
            wav_buffer = io.BytesIO(pending)
            if config.api:
                text = transcribe_api(wav_buffer)
            else:
                # Load audio from WAV for local transcription
                wav_buffer.seek(0)
                # Skip WAV header (44 bytes) and convert to float32
                audio = np.frombuffer(wav_buffer.read()[44:], dtype=np.int16).astype(np.float32) / 32767
                segments, _ = whisper_model.transcribe(audio, language=config.language)
                text = " ".join(seg.text for seg in segments).strip()

            if text and not is_hallucination(text):
                paste_text(" " + text)
                if history:
                    history.add(text)
                    history.clear_pending_audio()
                beep_success()
                set_terminal_title("TalkType ✅")
                show_status("✅ RETRIED", text[:50])
            else:
                beep_error()
                show_status("❌ NO SPEECH", "")
                if history:
                    history.clear_pending_audio()
        except Exception as e:
            beep_error()
            set_terminal_title("TalkType ❌")
            show_status("❌ RETRY FAILED", str(e)[:30])
            # Keep pending audio for another retry attempt

    return on_press


def main():
    global config, history

    # Check for first run or --setup flag
    if "--setup" in sys.argv or not CONFIG_PATH.exists():
        try:
            from setup_wizard import run_wizard
            result = run_wizard()
            # run_wizard returns (config, should_run)
            if isinstance(result, tuple):
                _, should_run = result
                if not should_run:
                    sys.exit(0)
        except ImportError:
            print("Setup wizard not available. Using defaults.")
        except KeyboardInterrupt:
            print("\nSetup cancelled.")
            sys.exit(0)

    config = parse_args()
    history = TranscriptionHistory(max_entries=config.history_limit)

    print("TalkType - Voice Typing for Your Terminal")
    print("=" * 45)
    print(f"System: {SYSTEM}")

    check_dependencies()
    load_whisper_model()

    hotkey = get_hotkey(config.hotkey)
    recovery_key = get_hotkey(config.recovery_hotkey)
    retry_key = get_hotkey(config.retry_hotkey)
    set_terminal_title("TalkType - Ready")

    if config.minimal:
        show_status("● READY", f"Press {config.hotkey.upper()} to record")
    else:
        print(f"\nReady! Press {config.hotkey.upper()} to record, {config.recovery_hotkey.upper()} to recover, {config.retry_hotkey.upper()} to retry.")
        print("Press Ctrl+C to exit.\n")

    # Create handlers for all hotkeys
    record_handler = create_hotkey_handler(hotkey)
    recovery_handler = create_recovery_handler(recovery_key)
    retry_handler = create_retry_handler(retry_key)

    def combined_handler(key):
        record_handler(key)
        recovery_handler(key)
        retry_handler(key)

    # Use signal handler for clean Ctrl+C exit
    import signal
    def signal_handler(sig, frame):
        print("\nBye!")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    with keyboard.Listener(on_press=combined_handler) as listener:
        listener.join()


if __name__ == "__main__":
    main()
