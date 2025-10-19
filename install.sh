#!/bin/bash
# Installation script for Voice TTS on macOS/Linux

set -e

echo "===================================="
echo "Voice TTS Installation - macOS"
echo "===================================="
echo ""

echo "[1/4] Checking Python..."
python3 --version || {
    echo "ERROR: Python 3 not found. Please install Python 3.9 or higher."
    exit 1
}
echo ""

echo "[2/4] Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo ""
echo "[3/4] Installing voice package..."
python3 -m pip install -e .

echo ""
echo "[4/4] Verifying installation..."
voice --version

echo ""
echo "===================================="
echo "Installation complete!"
echo "===================================="
echo ""
echo "Quick start:"
echo "  voice --list"
echo '  voice glados "Hello, test subject."'
echo ""
