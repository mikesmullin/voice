"""GLaDOS TTS engine implementation."""

import os
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from . import TTSEngine


class GladosTTSEngine(TTSEngine):
    """GLaDOS voice TTS engine using ONNX models."""
    
    def __init__(self, model_path: str, phonemizer_path: Optional[str] = None, speed: float = 1.0):
        """
        Initialize GLaDOS TTS engine.
        
        Args:
            model_path: Path to GLaDOS ONNX model
            phonemizer_path: Path to phonemizer ONNX model
            speed: Speech speed multiplier
        """
        self.model_path = model_path
        self.phonemizer_path = phonemizer_path
        self.speed = speed
        self.session = None
        self.phonemizer_session = None
        
    def initialize(self) -> None:
        """Initialize the ONNX runtime sessions."""
        if ort is None:
            raise ImportError("onnxruntime is required for GLaDOS TTS. Install with: pip install onnxruntime")
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"GLaDOS model not found at {self.model_path}")
        
        # Configure ONNX runtime
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Try to use GPU if available
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
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech using GLaDOS model.
        
        This is a simplified implementation. For production, you'd need to:
        1. Convert text to phonemes using the phonemizer
        2. Process phonemes through the GLaDOS model
        3. Apply vocoder to generate audio
        
        Args:
            text: Input text
            
        Returns:
            Audio waveform as numpy array
        """
        if self.session is None:
            raise RuntimeError("TTS engine not initialized. Call initialize() first.")
        
        # TODO: Implement proper text preprocessing and phoneme conversion
        # This is a placeholder implementation
        # In reality, you need to:
        # 1. Preprocess text (expand numbers, etc.)
        # 2. Convert to phonemes
        # 3. Run inference
        
        # For now, return a simple placeholder
        # You'll need to integrate the actual GLaDOS synthesis pipeline
        print(f"[GLaDOS TTS] Synthesizing: {text}")
        
        # Placeholder: Generate silence (you need to implement actual synthesis)
        sample_rate = 22050
        duration = len(text) * 0.1  # Rough estimate
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        return audio
    
    def get_sample_rate(self) -> int:
        """Get sample rate for GLaDOS voice."""
        return 22050
    
    def cleanup(self) -> None:
        """Clean up ONNX sessions."""
        self.session = None
        self.phonemizer_session = None


# Helper function to integrate with existing GLaDOS codebase
def create_glados_engine_from_existing(model_path: str, speed: float = 1.0) -> 'GladosTTSEngine':
    """
    Create GLaDOS engine using the existing GLaDOS project's TTS implementation.
    
    This is a bridge to use the existing glados.TTS.tts_glados module.
    """
    try:
        # Try to import from the parent GLaDOS project
        import sys
        from pathlib import Path
        
        # Add parent src to path
        parent_src = Path(__file__).parent.parent.parent / "src"
        if parent_src.exists() and str(parent_src) not in sys.path:
            sys.path.insert(0, str(parent_src))
        
        from glados.TTS import tts_glados
        
        class GladosWrapper(TTSEngine):
            """Wrapper around existing GLaDOS TTS."""
            
            def __init__(self, model_path: str, speed: float = 1.0):
                self.model_path = model_path
                self.speed = speed
                self.synthesizer = None
                
            def initialize(self) -> None:
                self.synthesizer = tts_glados.Synthesizer()
                
            def synthesize(self, text: str) -> np.ndarray:
                if self.synthesizer is None:
                    raise RuntimeError("TTS engine not initialized")
                
                audio = self.synthesizer.generate_speech_audio(text)
                return audio
                
            def get_sample_rate(self) -> int:
                return 22050
        
        return GladosWrapper(model_path, speed)
        
    except ImportError:
        # Fall back to standalone implementation
        return GladosTTSEngine(model_path, speed=speed)
