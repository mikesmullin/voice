"""ElevenLabs TTS engine implementation."""

import os
from typing import Optional
import numpy as np

try:
    from elevenlabs import generate, set_api_key, Voice
except ImportError:
    generate = None
    set_api_key = None
    Voice = None

from . import TTSEngine


class ElevenLabsTTSEngine(TTSEngine):
    """ElevenLabs cloud-based TTS engine."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model: str = "eleven_monolingual_v1"
    ):
        """
        Initialize ElevenLabs TTS engine.
        
        Args:
            api_key: ElevenLabs API key (or use ELEVENLABS_API_KEY env var)
            voice_id: Voice ID to use
            model: Model to use for generation
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id
        self.model = model
        
    def initialize(self) -> None:
        """Initialize the ElevenLabs API."""
        if generate is None:
            raise ImportError(
                "elevenlabs package is required for ElevenLabs TTS. "
                "Install with: pip install elevenlabs"
            )
        
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key is required. "
                "Set it in config.yaml or via ELEVENLABS_API_KEY environment variable."
            )
        
        set_api_key(self.api_key)
        print("[ElevenLabs TTS] Initialized")
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech using ElevenLabs API.
        
        Args:
            text: Input text
            
        Returns:
            Audio waveform as numpy array
        """
        if generate is None:
            raise RuntimeError("ElevenLabs package not available")
        
        print(f"[ElevenLabs TTS] Synthesizing: {text}")
        
        # Generate audio using ElevenLabs API
        audio_data = generate(
            text=text,
            voice=self.voice_id,
            model=self.model
        )
        
        # Convert bytes to numpy array
        # ElevenLabs returns PCM audio data
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize to float32 range [-1.0, 1.0]
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        return audio_float
    
    def get_sample_rate(self) -> int:
        """Get sample rate for ElevenLabs audio."""
        return 44100  # ElevenLabs typically uses 44.1kHz
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
