#!/usr/bin/env python3
"""First-run setup wizard for TalkType."""

from pathlib import Path
import shutil
import sys
from pynput import keyboard
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from beaupy import select, Config
from beaupy.spinners import Spinner, DOTS
import yaml
import time

console = Console()
CONFIG_PATH = Path.home() / ".config" / "talktype" / "config.yaml"

# Configure beaupy styling
Config.raise_on_interrupt = True


def capture_hotkey(prompt: str, default: str) -> str:
    """Capture a single keypress from the user."""
    import termios
    import tty

    console.print(f"\n  [bold]{prompt}[/bold]")
    console.print(f"  [dim]Press any key... (default: {default.upper()})[/dim]")

    captured = [None]

    # Save terminal settings and switch to raw mode (no echo)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def on_press(key):
        try:
            if hasattr(key, 'name'):
                captured[0] = key.name.lower()
            elif hasattr(key, 'char') and key.char:
                captured[0] = key.char.lower()
            else:
                captured[0] = default
        except Exception:
            captured[0] = default
        return False

    try:
        tty.setraw(fd)
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    result = captured[0] or default
    console.print(f"  [green]\u2713[/green] Set to [bold cyan]{result.upper()}[/bold cyan]")
    return result


def select_option(title: str, options: list[str], cursor_index: int = 0) -> int:
    """Arrow-navigable selection menu using beaupy."""
    console.print(f"\n  [bold]{title}[/bold]\n")

    result = select(
        options=options,
        cursor="  \u203a ",
        cursor_style="cyan bold",
        cursor_index=cursor_index,
        return_index=True,
    )

    if result is None:
        raise KeyboardInterrupt

    return result


def run_wizard() -> dict:
    """Run the interactive setup wizard."""
    console.clear()

    # Welcome banner
    console.print()
    console.print(Panel(
        "[bold cyan]TalkType Setup[/bold cyan]\n\n"
        "Voice typing that works everywhere.\n"
        "Let's configure your preferences.",
        border_style="cyan",
        padding=(1, 3),
        width=50
    ))

    config = {"hotkeys": {}, "transcription": {}, "ui": {}, "history": {}}

    # Step 1: Mode
    console.print()
    console.rule("[cyan]Step 1 of 4: Mode[/cyan]", style="dim")

    mode_options = [
        "Local Server    \u2192 Your whisper_server.py",
        "Cloud API       \u2192 Groq, OpenAI, etc.",
        "Local Model     \u2192 Standalone (slower startup)",
    ]
    mode_choice = select_option("How do you want to run transcription?", mode_options, 0)

    if mode_choice == 0:  # Local server
        config["transcription"]["mode"] = "api"
        config["transcription"]["api_url"] = "http://localhost:8002/transcribe"

    elif mode_choice == 1:  # Cloud API
        config["transcription"]["mode"] = "api"
        console.print()

        provider_options = [
            "Groq      \u2192 Free tier, very fast",
            "OpenAI    \u2192 Official Whisper API",
            "Custom    \u2192 Enter your own URL",
        ]
        provider = select_option("Select provider:", provider_options, 0)

        if provider == 0:
            config["transcription"]["api_url"] = "https://api.groq.com/openai/v1/audio/transcriptions"
            config["transcription"]["api_model"] = "whisper-large-v3"
            console.print("\n  [yellow]\u26a0[/yellow]  Set [bold]GROQ_API_KEY[/bold] env variable")
        elif provider == 1:
            config["transcription"]["api_url"] = "https://api.openai.com/v1/audio/transcriptions"
            config["transcription"]["api_model"] = "whisper-1"
            console.print("\n  [yellow]\u26a0[/yellow]  Set [bold]OPENAI_API_KEY[/bold] env variable")
        else:
            console.print()
            url = Prompt.ask("  [bold]API URL[/bold]")
            config["transcription"]["api_url"] = url
            model = Prompt.ask("  [bold]Model name[/bold]", default="whisper-1")
            config["transcription"]["api_model"] = model

    else:  # Local model
        config["transcription"]["mode"] = "local"

    # Step 2: Hotkeys
    console.print()
    console.rule("[cyan]Step 2 of 4: Hotkeys[/cyan]", style="dim")
    config["hotkeys"]["record"] = capture_hotkey("Press your RECORD hotkey:", "f9")
    config["hotkeys"]["retry"] = capture_hotkey("Press your RETRY hotkey:", "f7")
    config["hotkeys"]["recovery"] = capture_hotkey("Press your RECOVERY hotkey:", "f8")

    # Step 3: Model
    console.print()
    console.rule("[cyan]Step 3 of 4: Model[/cyan]", style="dim")

    models = ["tiny", "base", "small", "medium", "large-v3"]
    model_options = [
        "tiny      \u2192 ~1GB  Fastest",
        "base      \u2192 ~1GB  Recommended",
        "small     \u2192 ~2GB  Better accuracy",
        "medium    \u2192 ~5GB  High accuracy",
        "large-v3  \u2192 ~10GB Best accuracy",
    ]
    model_choice = select_option("Select Whisper model:", model_options, 1)
    config["transcription"]["model"] = models[model_choice]

    # Step 4: Language
    console.print()
    console.rule("[cyan]Step 4 of 4: Language[/cyan]", style="dim")

    langs = [None, "en", "es", "fr", "de", None]
    lang_options = [
        "Auto-detect",
        "English",
        "Spanish",
        "French",
        "German",
        "Other...",
    ]
    lang_choice = select_option("Language preference:", lang_options, 0)

    if lang_choice == 5:  # Other
        console.print()
        code = Prompt.ask("  [bold]Language code[/bold]", default="en")
        config["transcription"]["language"] = code
    else:
        config["transcription"]["language"] = langs[lang_choice]

    # Defaults
    config["ui"]["minimal"] = False
    config["history"]["limit"] = 100

    # Save
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Summary
    console.print()
    console.rule("[green]Setup Complete[/green]", style="dim")

    lang_display = config["transcription"]["language"] or "Auto-detect"
    mode_map = {"api": "API", "local": "Local Model"}

    lines = [
        f"[green]\u2713[/green] [bold]TalkType Configured[/bold]",
        "",
        f"  Mode:      {mode_map.get(config['transcription']['mode'], 'API')}",
    ]
    if config["transcription"].get("api_url"):
        url = config["transcription"]["api_url"]
        # Shorten URL for display
        if "localhost" in url:
            lines.append(f"  Server:    localhost:8002")
        elif "groq" in url:
            lines.append(f"  Provider:  Groq")
        elif "openai" in url:
            lines.append(f"  Provider:  OpenAI")
        else:
            lines.append(f"  API:       {url[:40]}...")

    lines.extend([
        f"  Model:     {config['transcription']['model']}",
        f"  Language:  {lang_display}",
        "",
        f"  [dim]Record:[/dim]   {config['hotkeys']['record'].upper()}",
        f"  [dim]Retry:[/dim]    {config['hotkeys']['retry'].upper()}",
        f"  [dim]Recovery:[/dim] {config['hotkeys']['recovery'].upper()}",
    ])

    console.print(Panel("\n".join(lines), border_style="green", padding=(1, 2), width=50))

    # PATH tip
    if not shutil.which("talktype"):
        venv_bin = Path(__file__).parent / "venv" / "bin"
        console.print()
        console.print(Panel(
            f"[yellow]Tip:[/yellow] To run [bold]talktype[/bold] from anywhere:\n\n"
            f"  echo 'export PATH=\"$PATH:{venv_bin}\"' >> ~/.zshrc\n"
            f"  source ~/.zshrc",
            title="Add to PATH",
            title_align="left",
            border_style="yellow",
            padding=(1, 2),
            width=60
        ))

    # Ask what to do next
    console.print()
    run_options = [
        "Run now         \u2192 Start in this terminal",
        "Run at startup  \u2192 Install as system service (always on)",
        "Exit            \u2192 Just save config, run later",
    ]
    run_choice = select_option("What would you like to do?", run_options, 0)

    if run_choice == 0:
        console.print("\n  [bold]Starting TalkType...[/bold]\n")
        time.sleep(0.3)
        return config, True  # Run now

    elif run_choice == 1:
        # Install systemd user service
        install_systemd_service(config)
        return config, False  # Don't run (service handles it)

    else:
        console.print("\n  [green]\u2713[/green] Config saved! Run [bold cyan]talktype[/bold cyan] anytime.\n")
        return config, False


def install_systemd_service(config: dict):
    """Install TalkType as a systemd user service."""
    import subprocess

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_dir.mkdir(parents=True, exist_ok=True)

    # Find talktype command - works whether pip installed or run from source
    talktype_cmd = shutil.which("talktype")
    if talktype_cmd:
        cmd_parts = [talktype_cmd]
    else:
        # Fallback: use current Python + talktype.py
        talktype_path = Path(__file__).parent / "talktype.py"
        cmd_parts = [sys.executable, str(talktype_path)]

    # Build command based on config
    if config["transcription"].get("api_url"):
        cmd_parts.extend(["--api", config["transcription"]["api_url"]])
    if config["transcription"].get("language"):
        cmd_parts.extend(["--language", config["transcription"]["language"]])

    service_content = f"""[Unit]
Description=TalkType Voice Typing
After=graphical-session.target

[Service]
Type=simple
ExecStart={' '.join(cmd_parts)}
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
"""

    service_file = service_dir / "talktype.service"
    service_file.write_text(service_content)

    # Enable and start the service
    console.print("\n  [dim]Installing systemd service...[/dim]")

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True)
        subprocess.run(["systemctl", "--user", "enable", "talktype.service"], check=True, capture_output=True)
        subprocess.run(["systemctl", "--user", "start", "talktype.service"], check=True, capture_output=True)

        console.print()
        console.print(Panel(
            "[green]\u2713[/green] [bold]TalkType installed as system service![/bold]\n\n"
            "  It will start automatically on login.\n\n"
            "  [dim]Manage with:[/dim]\n"
            "    systemctl --user status talktype\n"
            "    systemctl --user stop talktype\n"
            "    systemctl --user restart talktype",
            border_style="green",
            padding=(1, 2),
            width=55
        ))
    except subprocess.CalledProcessError as e:
        console.print(f"\n  [red]\u2717[/red] Failed to install service: {e}")
        console.print(f"  [dim]Service file saved to: {service_file}[/dim]")
    except FileNotFoundError:
        console.print("\n  [yellow]\u26a0[/yellow] systemctl not found. Service file saved to:")
        console.print(f"  [dim]{service_file}[/dim]")


def check_first_run() -> bool:
    """Return True if this is first run (no config file)."""
    return not CONFIG_PATH.exists()


def load_config() -> dict:
    """Load config from YAML file."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    try:
        run_wizard()
    except KeyboardInterrupt:
        console.print("\n  [dim]Setup cancelled.[/dim]\n")
        sys.exit(0)
