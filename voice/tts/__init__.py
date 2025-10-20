"""Base TTS interface and implementations."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the TTS engine and load models."""
        pass
    
    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Audio data as numpy array (float32, range -1.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get the sample rate of the generated audio."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources (optional)."""
        pass
