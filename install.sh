#!/usr/bin/env bash
# TalkType installer for Linux

set -e

echo "Installing TalkType..."

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "This installer is for Linux. See README.md for other platforms."
    exit 1
fi

# Detect distribution
DISTRO=""
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
elif type lsb_release >/dev/null 2>&1; then
    DISTRO=$(lsb_release -si)
elif [ -f /etc/redhat-release ]; then
    DISTRO=$(awk '{print tolower($1)}' /etc/redhat-release)
elif [ -f /etc/arch-release ]; then
    DISTRO="arch"
fi

# Convert distro name to lowercase for easier matching
DISTRO=$(echo "$DISTRO" | tr '[:upper:]' '[:lower:]')

echo "Detected distribution: $DISTRO"

# Install system dependencies based on distribution
echo "Installing system dependencies..."
case "$DISTRO" in
    ubuntu|debian|linuxmint)
        sudo apt-get update -qq
        sudo apt-get install -y -qq xautomation wmctrl x11-utils portaudio19-dev python3-venv
        ;;
    fedora|centos|rhel)
        sudo dnf check-update || sudo yum check-update # Check if dnf is available, otherwise try yum
        sudo dnf install -y xautomation wmctrl xorg-x11-utils portaudio-devel python3-venv || \
        sudo yum install -y xautomation wmctrl xorg-x11-utils portaudio-devel python3-venv
        ;;
    arch|manjaro)
        sudo pacman -Sy --noconfirm
        sudo pacman -S --noconfirm xautomation wmctrl xorg-xprop portaudio python-venv
        ;;
    suse|opensuse|sles)
        sudo zypper refresh
        sudo zypper install -y xautomation wmctrl xprop portaudio-devel # venv comes with python in arch
        ;;
    *)
        echo "Unsupported distribution: $DISTRO"
        echo "Attempting to install common dependencies. This might fail."
        echo "Please install 'xautomation' (for xte), 'wmctrl', 'xprop', 'portaudio-dev' (or equivalent), and 'python3-venv' manually."
        ;;
esac

# Create venv and install
echo "Setting up Python environment..."
# Check if python3-venv was successfully installed or is already present
if command -v python3 >/dev/null 2>&1; then
    python3 -m venv venv
else
    echo "Error: python3 is not available or python3-venv could not be installed."
    echo "Please ensure Python 3 and its venv module are installed on your system."
    exit 1
fi

source venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "Installation complete!"
echo ""
echo "To run TalkType:"
echo "  source venv/bin/activate"
echo "  python talktype.py"
echo ""
echo "Or for faster startup, run the Whisper server first:"
echo "  python whisper_server.py &"
echo "  python talktype.py --api http://localhost:8002/transcribe"
