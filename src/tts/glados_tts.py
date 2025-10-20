"""GLaDOS TTS engine implementation."""

import os
from pathlib import Path
from typing import Optional
import numpy as np

from . import TTSEngine
from .glados_synthesizer import GladosSynthesizer


class GladosTTSEngine(TTSEngine):
    """GLaDOS voice TTS engine using ONNX models."""
    
    def __init__(self, model_path: str, phonemizer_path: Optional[str] = None, speed: float = 1.0):
        """
        Initialize GLaDOS TTS engine.
        
        Args:
            model_path: Path to GLaDOS ONNX model
            phonemizer_path: Path to phonemizer ONNX model (not used, kept for compatibility)
            speed: Speech speed multiplier
        """
        self.model_path = model_path
        self.speed = speed
        self.synthesizer = None
        
    def initialize(self) -> None:
        """Initialize the GLaDOS synthesizer."""
        # Check if model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"GLaDOS model not found at {self.model_path}")
        
        # Determine phoneme_to_id path
        model_dir = Path(self.model_path).parent
        phoneme_path = model_dir / "phoneme_to_id.pkl"
        
        if not phoneme_path.exists():
            raise FileNotFoundError(f"phoneme_to_id.pkl not found at {phoneme_path}")
        
        print(f"[GLaDOS TTS] Initializing with model: {self.model_path}")
        self.synthesizer = GladosSynthesizer(
            model_path=Path(self.model_path),
            phoneme_path=phoneme_path
        )
        
        # Warm up the synthesizer with a dummy synthesis to ensure all buffers are ready
        # This prevents audio cutoff on the first real synthesis
        try:
            _ = self.synthesizer.generate_speech_audio(".")
            print(f"[GLaDOS TTS] Synthesizer warm-up complete")
        except Exception as e:
            print(f"[GLaDOS TTS] Warm-up synthesis failed (non-fatal): {e}")
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech using GLaDOS model.
        
        Args:
            text: Input text
            
        Returns:
            Audio waveform as numpy array
        """
        if self.synthesizer is None:
            raise RuntimeError("TTS engine not initialized. Call initialize() first.")
        
        print(f"[GLaDOS TTS] Synthesizing: {text}")
        audio = self.synthesizer.generate_speech_audio(text)
        
        return audio
    
    def get_sample_rate(self) -> int:
        """Get sample rate for GLaDOS voice."""
        if self.synthesizer:
            return self.synthesizer.sample_rate
        return 22050
    
    def cleanup(self) -> None:
        """Clean up synthesizer."""
        self.synthesizer = None

