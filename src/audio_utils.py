"""Audio utilities for playback and file saving."""

import os
from typing import Optional
import numpy as np
import time

from .timing import log

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

# Removed pydub dependency - only supporting WAV format


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    """
    Play audio through the default audio device.
    
    Args:
        audio: Audio data as numpy array (float32, range -1.0 to 1.0)
        sample_rate: Sample rate in Hz
    """
    if sd is None:
        raise ImportError("sounddevice is required for audio playback. Install with: pip install sounddevice")
    
    # Get default output device info
    default_device = sd.default.device[1]  # Output device
    device_info = sd.query_devices(default_device)
    
    log(f"[Voice] Playing audio ({len(audio)/sample_rate:.2f}s) on: {device_info['name']}")
    log(f"[Voice] Audio shape: {audio.shape}, dtype: {audio.dtype}, range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    try:
        # Ensure audio is the right shape and type
        if len(audio.shape) == 1:
            # Mono audio - keep as 1D for sounddevice
            audio_to_play = audio
        elif len(audio.shape) == 2 and audio.shape[1] == 1:
            # Convert (N, 1) to (N,) - flatten
            audio_to_play = audio.flatten()
        else:
            audio_to_play = audio
            
        # Play audio with non-blocking mode first to allow driver to initialize
        sd.play(audio_to_play, samplerate=sample_rate, device=default_device, blocking=False)
        
        # Small delay to let playback start
        time.sleep(0.01)
        
        # Now wait for completion
        sd.wait()
        log(f"[Voice] Playback complete")
    except Exception as e:
        print(f"Error playing audio: {e}")
        import traceback
        traceback.print_exc()


def save_audio_wav(audio: np.ndarray, sample_rate: int, output_path: str) -> None:
    """
    Save audio to WAV file.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        output_path: Output file path
    """
    if sf is None:
        raise ImportError("soundfile is required for saving audio. Install with: pip install soundfile")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as WAV
    sf.write(output_path, audio, sample_rate)
    log(f"[Voice] Saved to: {output_path}")


# Removed OGG support to avoid ffmpeg dependency
# Removed append_audio_wav function (unused)


def save_audio(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str,
    format: Optional[str] = None
) -> None:
    """
    Save audio to file as WAV format.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        output_path: Output file path
        format: Output format (only 'wav' is supported)
    """
    # Force WAV format regardless of extension
    if format and format.lower() != 'wav':
        print(f"Warning: Only WAV format is supported, ignoring format '{format}'")
    
    # If output path has a different extension, warn and change to .wav
    if not output_path.lower().endswith('.wav'):
        print(f"Warning: Changing output extension to .wav")
        output_path = output_path.rsplit('.', 1)[0] + '.wav' if '.' in output_path else output_path + '.wav'
    
    save_audio_wav(audio, sample_rate, output_path)
