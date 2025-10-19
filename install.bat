@echo off
REM Installation script for Voice TTS on Windows

echo ====================================
echo Voice TTS Installation - Windows
echo ====================================
echo.

echo [1/4] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.9 or higher.
    pause
    exit /b 1
)
echo.

echo [2/4] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo [3/4] Installing voice package...
python -m pip install -e .

echo.
echo [4/4] Verifying installation...
voice --version

echo.
echo ====================================
echo Installation complete!
echo ====================================
echo.
echo Quick start:
echo   voice --list
echo   voice glados "Hello, test subject."
echo.
echo For GPU support (NVIDIA), run:
echo   pip install onnxruntime-gpu
echo.

pause
