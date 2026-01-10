"""Piper TTS engine - fast CPU-based text-to-speech."""

import os
import time
import wave
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional
import numpy as np

from .timing import log


class PiperEngine:
    """
    Piper TTS engine - fast, CPU-optimized text-to-speech.
    
    Piper uses ONNX models optimized for edge devices and runs
    efficiently on CPU without requiring a GPU.
    """
    
    BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize Piper engine.
        
        Args:
            model_dir: Directory to store downloaded models (default: ~/.cache/voice/piper-models)
        """
        cache_dir = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        self.model_dir = Path(model_dir or cache_dir) / "voice" / "piper-models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_voices = {}
        self.last_voice_id = None
    
    def _parse_voice_id(self, voice_id: str) -> tuple[str, str, str]:
        """
        Parse a Piper voice ID into components.
        
        Args:
            voice_id: Voice ID like "en_US-lessac-high"
            
        Returns:
            Tuple of (locale, voice_name, quality)
        """
        # Format: en_US-lessac-high
        parts = voice_id.split("-")
        locale = parts[0]  # en_US or en_GB
        quality = parts[-1]  # low, medium, high
        voice_name = "-".join(parts[1:-1])  # voice name (may have hyphens)
        return locale, voice_name, quality
    
    def _get_model_url(self, voice_id: str) -> tuple[str, str]:
        """Get download URLs for a voice model."""
        locale, voice_name, quality = self._parse_voice_id(voice_id)
        
        # URL structure: /en/en_US/lessac/medium/en_US-lessac-medium.onnx
        model_url = f"{self.BASE_URL}/en/{locale}/{voice_name}/{quality}/{voice_id}.onnx"
        config_url = f"{model_url}.json"
        return model_url, config_url
    
    def _get_model_path(self, voice_id: str) -> tuple[Path, Path]:
        """Get local paths for a voice model."""
        model_path = self.model_dir / f"{voice_id}.onnx"
        config_path = self.model_dir / f"{voice_id}.onnx.json"
        return model_path, config_path
    
    def _download_model(self, voice_id: str) -> tuple[Path, Path]:
        """Download a voice model if not cached."""
        model_path, config_path = self._get_model_path(voice_id)
        model_url, config_url = self._get_model_url(voice_id)
        
        if not model_path.exists():
            log(f"[Piper] Downloading model {voice_id}...")
            start = time.time()
            try:
                urllib.request.urlretrieve(model_url, model_path)
                log(f"[Piper] Downloaded in {time.time() - start:.1f}s")
            except Exception as e:
                log(f"[Piper] ERROR downloading {voice_id}: {e}")
                raise
        
        if not config_path.exists():
            urllib.request.urlretrieve(config_url, config_path)
        
        return model_path, config_path
    
    def _load_voice(self, voice_id: str):
        """Load a Piper voice model."""
        if voice_id in self.loaded_voices:
            return self.loaded_voices[voice_id]
        
        from piper import PiperVoice
        
        model_path, config_path = self._download_model(voice_id)
        
        log(f"[Piper] Loading voice {voice_id}...")
        start = time.time()
        
        piper_voice = PiperVoice.load(
            model_path=str(model_path),
            config_path=str(config_path),
            use_cuda=False  # Always CPU
        )
        
        log(f"[Piper] Loaded in {time.time() - start:.2f}s")
        
        self.loaded_voices[voice_id] = piper_voice
        return piper_voice
    
    def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_id: Piper voice ID (e.g., "en_US-lessac-high")
            speed: Speech speed multiplier (note: Piper has limited speed control)
            
        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        from .timing import get_elapsed
        from piper.config import SynthesisConfig
        
        synth_start = get_elapsed()
        
        # Load voice if needed
        piper_voice = self._load_voice(voice_id)
        self.last_voice_id = voice_id
        
        # Configure synthesis with speed control
        # Piper uses length_scale: 1.0 = normal, <1.0 = faster, >1.0 = slower
        # We invert the speed multiplier so speed=1.5 means 1.5x faster (length_scale=0.67)
        length_scale = 1.0 / speed if speed > 0 else 1.0
        syn_config = SynthesisConfig(length_scale=length_scale)
        
        if speed != 1.0:
            log(f"[Piper] Using speed={speed} (length_scale={length_scale:.2f})")
        
        # Synthesize to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with wave.open(tmp_path, "wb") as wav_file:
                piper_voice.synthesize_wav(text, wav_file, syn_config=syn_config)
            
            # Read the audio data
            with wave.open(tmp_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)
            
            # Convert to numpy float32 array
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            synth_time = get_elapsed() - synth_start
            audio_duration = len(audio_float) / sample_rate
            rtf = synth_time / audio_duration if audio_duration > 0 else 0
            
            log(f"[Piper] Synthesized {len(text)} chars in {synth_time:.2f}s "
                f"(RTF: {rtf:.2f}x, audio: {audio_duration:.1f}s)")
            
            return audio_float, sample_rate
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def get_sample_rate(self, voice_id: str) -> int:
        """Get the sample rate for a voice (typically 22050 for Piper)."""
        # Most Piper models use 22050 Hz
        return 22050
