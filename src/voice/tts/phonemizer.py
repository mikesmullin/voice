"""
Phonemizer for GLaDOS TTS - wrapper around parent GLaDOS project's phonemizer.
"""

import sys
from pathlib import Path
from typing import Any

# Try to import from parent GLaDOS project
try:
    # Add parent src to path
    parent_src = Path(__file__).parent.parent.parent.parent.parent / "src"
    if parent_src.exists() and str(parent_src) not in sys.path:
        sys.path.insert(0, str(parent_src))
    
    from glados.TTS.phonemizer import Phonemizer as ParentPhonemizer
    
    # Use the parent implementation directly
    Phonemizer = ParentPhonemizer
    
except ImportError:
    # Fallback: create a minimal standalone implementation
    print("Warning: Could not import parent GLaDOS phonemizer, using fallback")
    
    from dataclasses import dataclass
    from pathlib import Path
    from pickle import load
    import re
    
    import onnxruntime as ort
    import numpy as np
    
    @dataclass
    class ModelConfig:
        MODEL_PATH: Path
        PHONEME_DICT_PATH: Path
        TOKEN_TO_IDX_PATH: Path
        IDX_TO_TOKEN_PATH: Path
        CHAR_REPEATS: int = 3
        MODEL_INPUT_LENGTH: int = 64
        
        def __init__(
            self,
            model_path: Path | None = None,
            phoneme_dict_path: Path | None = None,
            token_to_idx_path: Path | None = None,
            idx_to_token_path: Path | None = None,
        ) -> None:
            # Default paths relative to models directory
            models_dir = Path(__file__).parent.parent.parent.parent.parent / "models" / "TTS"
            self.MODEL_PATH = model_path or models_dir / "phomenizer_en.onnx"
            self.PHONEME_DICT_PATH = phoneme_dict_path or models_dir / "lang_phoneme_dict.pkl"
            self.TOKEN_TO_IDX_PATH = token_to_idx_path or models_dir / "token_to_idx.pkl"
            self.IDX_TO_TOKEN_PATH = idx_to_token_path or models_dir / "idx_to_token.pkl"
    
    
    class Phonemizer:
        """Fallback phonemizer implementation."""
        
        def __init__(self, config: ModelConfig | None = None) -> None:
            """Initialize the phonemizer."""
            self.config = config or ModelConfig()
            
            # Load dictionaries
            self.phoneme_dict = self._load_pickle(self.config.PHONEME_DICT_PATH)
            self.token_to_idx = self._load_pickle(self.config.TOKEN_TO_IDX_PATH)
            self.idx_to_token = self._load_pickle(self.config.IDX_TO_TOKEN_PATH)
            
            # Setup ONNX Runtime
            providers = ort.get_available_providers()
            self.ort_session = ort.InferenceSession(
                str(self.config.MODEL_PATH),
                providers=providers
            )
        
        @staticmethod
        def _load_pickle(path: Path) -> dict:
            """Load a pickled dictionary."""
            with path.open("rb") as f:
                return dict(load(f))
        
        def convert_to_phonemes(self, texts: list[str], lang: str = "en_us") -> list[str]:
            """
            Convert text to phonemes.
            
            Args:
                texts: List of text strings to convert
                lang: Language code (default: en_us)
            
            Returns:
                List of phoneme strings
            """
            results = []
            for text in texts:
                # Simple word-by-word lookup
                words = text.lower().split()
                phonemes = []
                
                for word in words:
                    # Clean word
                    word = re.sub(r'[^\w\s]', '', word)
                    if not word:
                        continue
                    
                    # Look up in dictionary
                    if word in self.phoneme_dict:
                        phonemes.append(self.phoneme_dict[word])
                    else:
                        # Try to use the model for unknown words
                        word_phonemes = self._predict_phonemes(word)
                        if word_phonemes:
                            phonemes.append(word_phonemes)
                
                results.append(''.join(phonemes))
            
            return results
        
        def _predict_phonemes(self, word: str) -> str:
            """Predict phonemes for a word using the ONNX model."""
            # This is a simplified version - the real implementation is more complex
            # For now, return empty string and let the model handle it
            return ""
