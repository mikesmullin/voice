"""
Phonemizer for voice TTS - standalone implementation.
Ported from the original GLaDOS project with punctuation support.
"""

from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from pickle import load
import re
from typing import Any

import onnxruntime as ort
import numpy as np
from numpy.typing import NDArray

@dataclass
class ModelConfig:
    """Configuration for the phonemizer model."""
    
    MODEL_PATH: Path
    PHONEME_DICT_PATH: Path
    TOKEN_TO_IDX_PATH: Path
    IDX_TO_TOKEN_PATH: Path
    CHAR_REPEATS: int = 3
    MODEL_INPUT_LENGTH: int = 64
    EXPAND_ACRONYMS: bool = False
    USE_CUDA: bool = True
    
    def __init__(
        self,
        model_path: Path | None = None,
        phoneme_dict_path: Path | None = None,
        token_to_idx_path: Path | None = None,
        idx_to_token_path: Path | None = None,
    ) -> None:
        # Default paths relative to voice package
        # This resolves to voice/models/TTS/
        package_dir = Path(__file__).parent.parent.parent
        models_dir = package_dir / "models" / "TTS"
        self.MODEL_PATH = model_path or models_dir / "phomenizer_en.onnx"
        self.PHONEME_DICT_PATH = phoneme_dict_path or models_dir / "lang_phoneme_dict.pkl"
        self.TOKEN_TO_IDX_PATH = token_to_idx_path or models_dir / "token_to_idx.pkl"
        self.IDX_TO_TOKEN_PATH = idx_to_token_path or models_dir / "idx_to_token.pkl"


class Punctuation(Enum):
    """Punctuation and special characters for phonemization."""
    PUNCTUATION = "().,:?!/–"
    HYPHEN = "-"
    SPACE = " "

    @classmethod
    @cache
    def get_punc_set(cls) -> set[str]:
        """Get set of all punctuation characters to preserve."""
        return set(cls.PUNCTUATION.value + cls.HYPHEN.value + cls.SPACE.value)

    @classmethod
    @cache
    def get_punc_pattern(cls) -> re.Pattern[str]:
        """
        Compile a regular expression pattern to match punctuation and space characters.

        Returns:
            re.Pattern[str]: A compiled regex pattern that matches any punctuation or space character.
        """
        return re.compile(f"([{cls.PUNCTUATION.value + cls.SPACE.value}])")


class Phonemizer:
    """Phonemizer implementation for converting text to phonemes.
    
    Supports punctuation preservation for proper prosody and pausing.
    Ported from the original GLaDOS project.
    """
    
    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the phonemizer with proper punctuation support."""
        self.config = config or ModelConfig()
        
        # Load dictionaries
        self.phoneme_dict = self._load_pickle(self.config.PHONEME_DICT_PATH)
        self.token_to_idx = self._load_pickle(self.config.TOKEN_TO_IDX_PATH)
        self.idx_to_token = self._load_pickle(self.config.IDX_TO_TOKEN_PATH)
        
        # Add GLaDOS to phoneme dictionary
        self.phoneme_dict["glados"] = "ɡlˈɑːdɑːs"
        
        # Setup ONNX Runtime
        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")
        
        self.ort_session = ort.InferenceSession(
            str(self.config.MODEL_PATH),
            sess_options=ort.SessionOptions(),
            providers=providers
        )
    
    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        """Load a pickled dictionary from the specified file path."""
        with path.open("rb") as f:
            return dict(load(f))
    
    def _clean_and_split_texts(
        self, texts: list[str], punc_set: set[str], punc_pattern: re.Pattern[str]
    ) -> tuple[list[list[str]], set[str]]:
        """
        Clean and split input texts into words while preserving specified punctuation.

        This method preserves punctuation as separate tokens for proper prosody.

        Parameters:
            texts (list[str]): List of input text strings to be cleaned and split.
            punc_set (set[str]): Set of punctuation characters to preserve during cleaning.
            punc_pattern (re.Pattern[str]): Regular expression pattern for splitting text.

        Returns:
            tuple[list[list[str]], set[str]]: A tuple containing:
                - A list of lists, where each inner list represents tokens (words and punctuation)
                - A set of unique cleaned tokens across all input texts
        """
        split_text, cleaned_words = [], set[str]()
        for text in texts:
            # PRESERVE punctuation by not stripping it
            cleaned_text = "".join(t for t in text if t.isalnum() or t in punc_set)
            # Split by punctuation pattern BUT keep punctuation as separate tokens
            split = [s for s in re.split(punc_pattern, cleaned_text) if len(s) > 0]
            split_text.append(split)
            cleaned_words.update(split)
        return split_text, cleaned_words
    
    def _get_dict_entry(self, word: str, punc_set: set[str]) -> str | None:
        """
        Retrieves the phoneme entry for a given word from the phoneme dictionary.

        This method handles different word variations by checking the dictionary with original, lowercase, and
        title-cased versions of the word. It also handles punctuation and empty strings as special cases.

        Args:
            word (str): The word to look up in the phoneme dictionary.
            punc_set (set[str]): A set of punctuation characters.

        Returns:
            str | None: The phoneme entry for the word if found, the word itself if it's a punctuation or
            empty string, or None if no entry exists.
        """
        if word in punc_set or len(word) == 0:
            return word
        if word in self.phoneme_dict:
            return self.phoneme_dict[word]
        elif word.lower() in self.phoneme_dict:
            return self.phoneme_dict[word.lower()]
        elif word.title() in self.phoneme_dict:
            return self.phoneme_dict[word.title()]
        else:
            return None
    
    def convert_to_phonemes(self, texts: list[str], lang: str = "en_us") -> list[str]:
        """
        Convert text to phonemes while preserving punctuation.
        
        Args:
            texts: List of text strings to convert
            lang: Language code (default: en_us)
        
        Returns:
            List of phoneme strings with punctuation preserved as separate tokens
        """
        punc_set = Punctuation.get_punc_set()
        punc_pattern = Punctuation.get_punc_pattern()
        
        # Add punctuation entries to dictionary for lookup
        for punct in punc_set:
            self.phoneme_dict[punct] = punct
        
        # Split text while preserving punctuation
        split_text, cleaned_words = self._clean_and_split_texts(texts, punc_set, punc_pattern)
        
        # Get phonemes for each token
        word_phonemes = {word: self._get_dict_entry(word, punc_set) for word in cleaned_words}
        
        # Build phoneme lists for each input text
        phoneme_lists = []
        for text_tokens in split_text:
            text_phons = []
            for token in text_tokens:
                # Get phoneme for this token
                phons = word_phonemes.get(token)
                if phons is not None:
                    text_phons.append(phons)
                elif token.lower() in word_phonemes:
                    text_phons.append(word_phonemes[token.lower()])
                else:
                    # Skip unknown tokens
                    pass
            phoneme_lists.append(text_phons)
        
        # Join phonemes into strings
        return ["".join(phoneme_list) for phoneme_list in phoneme_lists]
    
    def _predict_phonemes(self, word: str) -> str:
        """Predict phonemes for a word using the ONNX model."""
        # This is a simplified version - the real implementation is more complex
        # For now, return empty string and let the model handle it
        return ""
