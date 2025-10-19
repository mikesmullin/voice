"""Audio utilities for playback and file saving."""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import time

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
    
    # Get default output device info
    default_device = sd.default.device[1]  # Output device
    device_info = sd.query_devices(default_device)
    
    print(f"[Audio] Playing audio ({len(audio)/sample_rate:.2f}s) on: {device_info['name']}")
    print(f"[Audio] Audio shape: {audio.shape}, dtype: {audio.dtype}, range: [{audio.min():.3f}, {audio.max():.3f}]")
    
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
        
        # Add a small silence buffer at the start to prevent cutoff
        # This helps with audio driver latency
        silence_duration = 0.05  # 50ms
        silence_samples = int(sample_rate * silence_duration)
        silence = np.zeros(silence_samples, dtype=audio_to_play.dtype)
        audio_with_buffer = np.concatenate([silence, audio_to_play])
            
        # Play audio with non-blocking mode first to allow driver to initialize
        sd.play(audio_with_buffer, samplerate=sample_rate, device=default_device, blocking=False)
        
        # Small delay to let playback start
        time.sleep(0.01)
        
        # Now wait for completion
        sd.wait()
        print(f"[Audio] Playback complete")
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
    
    # Add silence buffer at start to prevent cutoff when playing the file
    silence_duration = 0.05  # 50ms
    silence_samples = int(sample_rate * silence_duration)
    silence = np.zeros(silence_samples, dtype=audio.dtype)
    audio_with_buffer = np.concatenate([silence, audio])
    
    # Save as WAV
    sf.write(output_path, audio_with_buffer, sample_rate)
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
