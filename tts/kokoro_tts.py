"""Kokoro TTS engine implementation."""

import os
from typing import Optional
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from . import TTSEngine


class KokoroTTSEngine(TTSEngine):
    """Kokoro multi-voice TTS engine using ONNX models."""
    
    SUPPORTED_VOICES = [
        "af_bella", "af_sarah", "af_nicole",  # Female voices
        "am_adam", "am_michael", "bf_emma", "bf_isabella",  # Male and other voices
        "bm_george", "bm_lewis"
    ]
    
    def __init__(
        self,
        model_path: str,
        phonemizer_path: Optional[str] = None,
        voice: str = "af_bella",
        speed: float = 1.0
    ):
        """
        Initialize Kokoro TTS engine.
        
        Args:
            model_path: Path to Kokoro ONNX model
            phonemizer_path: Path to phonemizer ONNX model
            voice: Voice variant to use
            speed: Speech speed multiplier
        """
        self.model_path = model_path
        self.phonemizer_path = phonemizer_path
        self.voice = voice
        self.speed = speed
        self.session = None
        self.phonemizer_session = None
        
        if voice not in self.SUPPORTED_VOICES:
            print(f"Warning: Voice '{voice}' may not be supported. Available: {self.SUPPORTED_VOICES}")
    
    def initialize(self) -> None:
        """Initialize the ONNX runtime sessions."""
        if ort is None:
            raise ImportError("onnxruntime is required for Kokoro TTS. Install with: pip install onnxruntime")
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Kokoro model not found at {self.model_path}")
        
        # Configure ONNX runtime
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Try to use GPU if available, fall back to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Load main TTS model
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Load phonemizer if provided
        if self.phonemizer_path and os.path.exists(self.phonemizer_path):
            self.phonemizer_session = ort.InferenceSession(
                self.phonemizer_path,
                sess_options=sess_options,
                providers=providers
            )
        
        print(f"[Kokoro TTS] Initialized with voice: {self.voice}")
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech using Kokoro model.
        
        Args:
            text: Input text
            
        Returns:
            Audio waveform as numpy array
        """
        if self.session is None:
            raise RuntimeError("TTS engine not initialized. Call initialize() first.")
        
        # TODO: Implement proper Kokoro synthesis
        # This is a placeholder - you need to implement the actual Kokoro pipeline
        print(f"[Kokoro TTS - {self.voice}] Synthesizing: {text}")
        
        # Placeholder: Generate silence
        sample_rate = 22050
        duration = len(text) * 0.1
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        return audio
    
    def get_sample_rate(self) -> int:
        """Get sample rate for Kokoro voice."""
        return 22050
    
    def cleanup(self) -> None:
        """Clean up ONNX sessions."""
        self.session = None
        self.phonemizer_session = None


# Helper function to integrate with existing GLaDOS codebase
def create_kokoro_engine_from_existing(
    model_path: str,
    voice: str = "af_bella",
    speed: float = 1.0
) -> 'KokoroTTSEngine':
    """
    Create Kokoro engine using the existing GLaDOS project's TTS implementation.
    """
    try:
        # Try to import from the parent GLaDOS project
        import sys
        from pathlib import Path
        
        # Add parent src to path
        parent_src = Path(__file__).parent.parent.parent / "src"
        if parent_src.exists() and str(parent_src) not in sys.path:
            sys.path.insert(0, str(parent_src))
        
        from glados.TTS import tts_kokoro
        
        class KokoroWrapper(TTSEngine):
            """Wrapper around existing Kokoro TTS."""
            
            def __init__(self, model_path: str, voice: str, speed: float = 1.0):
                self.model_path = model_path
                self.voice = voice
                self.speed = speed
                self.synthesizer = None
                
            def initialize(self) -> None:
                self.synthesizer = tts_kokoro.Synthesizer(model_path=self.model_path)
                
            def synthesize(self, text: str) -> np.ndarray:
                if self.synthesizer is None:
                    raise RuntimeError("TTS engine not initialized")
                
                audio = self.synthesizer.generate_speech_audio(text, voice=self.voice)
                return audio
                
            def get_sample_rate(self) -> int:
                return 22050
        
        return KokoroWrapper(model_path, voice, speed)
        
    except ImportError:
        # Fall back to standalone implementation
        return KokoroTTSEngine(model_path, voice=voice, speed=speed)
