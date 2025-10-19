"""Audio utilities for playback and file saving."""

import os
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    """
    Play audio through the default audio device.
    
    Args:
        audio: Audio data as numpy array (float32, range -1.0 to 1.0)
        sample_rate: Sample rate in Hz
    """
    if sd is None:
        raise ImportError("sounddevice is required for audio playback. Install with: pip install sounddevice")
    
    print(f"[Audio] Playing audio ({len(audio)/sample_rate:.2f}s)")
    
    try:
        # Play audio and wait for completion
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")


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
    print(f"[Audio] Saved to: {output_path}")


def save_audio_ogg(audio: np.ndarray, sample_rate: int, output_path: str) -> None:
    """
    Save audio to OGG file.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        output_path: Output file path
    """
    if sf is None:
        raise ImportError("soundfile is required for saving audio. Install with: pip install soundfile")
    
    if AudioSegment is None:
        raise ImportError("pydub is required for OGG conversion. Install with: pip install pydub")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # First save as temporary WAV
    temp_wav = output_path.rsplit('.', 1)[0] + '_temp.wav'
    
    try:
        # Save as WAV first
        sf.write(temp_wav, audio, sample_rate)
        
        # Convert to OGG
        audio_segment = AudioSegment.from_wav(temp_wav)
        audio_segment.export(output_path, format="ogg")
        
        print(f"[Audio] Saved to: {output_path}")
        
    finally:
        # Clean up temporary WAV file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def save_audio(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str,
    format: Optional[str] = None
) -> None:
    """
    Save audio to file with automatic format detection.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        output_path: Output file path
        format: Output format ('wav', 'ogg', or None for auto-detect from extension)
    """
    # Determine format from extension if not specified
    if format is None:
        ext = Path(output_path).suffix.lower()
        format = ext[1:] if ext else 'wav'
    
    format = format.lower()
    
    # Save based on format
    if format == 'ogg':
        save_audio_ogg(audio, sample_rate, output_path)
    elif format == 'wav':
        save_audio_wav(audio, sample_rate, output_path)
    else:
        # Default to WAV for unknown formats
        print(f"Warning: Unknown format '{format}', saving as WAV")
        save_audio_wav(audio, sample_rate, output_path)
