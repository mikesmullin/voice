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
    PUNCTUATION = "().,:?!/–"  # Note: apostrophe removed - handled separately for contractions
    HYPHEN = "-"
    SPACE = " "
    APOSTROPHE = "'"

    @classmethod
    @cache
    def get_punc_set(cls) -> set[str]:
        """Get set of all punctuation characters to preserve."""
        # Include apostrophe in the set but don't split it in patterns
        return set(cls.PUNCTUATION.value + cls.HYPHEN.value + cls.SPACE.value + cls.APOSTROPHE.value)

    @classmethod
    @cache
    def get_punc_pattern(cls) -> re.Pattern[str]:
        """
        Compile a regular expression pattern to match punctuation and space characters.
        
        Note: Apostrophes are NOT included in the split pattern to preserve contractions
        like "I'm", "don't", "we're", etc.

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
        For unknown words, uses the ONNX model to predict phonemes. For contractions, expands them first.

        Args:
            word (str): The word to look up in the phoneme dictionary.
            punc_set (set[str]): A set of punctuation characters.

        Returns:
            str | None: The phoneme entry for the word if found, or predicted phonemes for unknown words.
        """
        if word in punc_set or len(word) == 0:
            return word
        
        # Check if this is a contraction and expand it
        expanded = self._expand_contractions(word)
        if len(expanded) > 1:
            # This was a contraction - phonemize each part and concatenate
            phonemes = []
            for part in expanded:
                phon = self._get_dict_entry(part, punc_set)
                if phon:
                    phonemes.append(phon)
            return "".join(phonemes)
        
        # Not a contraction, proceed with normal lookup
        if word in self.phoneme_dict:
            return self.phoneme_dict[word]
        elif word.lower() in self.phoneme_dict:
            return self.phoneme_dict[word.lower()]
        elif word.title() in self.phoneme_dict:
            return self.phoneme_dict[word.title()]
        else:
            # Use ONNX model to predict phonemes for unknown words
            return self._predict_phonemes(word)
    
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
                # Get phoneme for this token (will be the word itself if not in dict)
                if token in word_phonemes:
                    text_phons.append(word_phonemes[token])
            phoneme_lists.append(text_phons)
        
        # Join phonemes into strings
        return ["".join(phoneme_list) for phoneme_list in phoneme_lists]
    
    def _predict_phonemes(self, word: str) -> str:
        """
        Predict phonemes for a word using the ONNX model.
        
        Args:
            word: The word to phonemize
            
        Returns:
            Predicted phoneme string
        """
        # Convert word to lowercase for processing
        word_lower = word.lower()
        
        # Convert characters to token indices
        char_indices = []
        for char in word_lower:
            if char in self.token_to_idx:
                char_indices.append(self.token_to_idx[char])
            else:
                # Skip unknown characters
                continue
        
        if not char_indices:
            # If no valid characters, return the word as-is
            return word
        
        # Remember the actual number of characters for later
        actual_len = len(char_indices)
        
        # Pad or truncate to model input length (64)
        max_len = self.config.MODEL_INPUT_LENGTH
        if len(char_indices) < max_len:
            # Pad with 0 (underscore token)
            char_indices = char_indices + [0] * (max_len - len(char_indices))
        else:
            # Truncate if too long
            char_indices = char_indices[:max_len]
        
        # Prepare input for ONNX model
        input_array = np.array([char_indices], dtype=np.int64)
        
        try:
            # Run inference
            outputs = self.ort_session.run(None, {"modelInput": input_array})
            phoneme_output = outputs[0][0]  # (batch_size=1, 64, 64) -> (64, 64)
            
            # Convert model output to phoneme indices
            # The model outputs logits for each position and phoneme class
            # Take argmax to get the most likely phoneme for each position
            phoneme_indices = np.argmax(phoneme_output, axis=1)
            
            # Convert indices to phoneme strings, only for actual input length
            phonemes = []
            for i in range(actual_len):
                idx = phoneme_indices[i]
                if idx in self.idx_to_token:
                    token = self.idx_to_token[idx]
                    # Skip special tokens and separators
                    if token not in ['<start>', '<end>', '_', ' ']:
                        phonemes.append(token)
            
            result = "".join(phonemes)
            return result if result else word
            
        except Exception as e:
            # If inference fails, return the word as-is
            return word

    def _expand_contractions(self, word: str) -> list[str]:
        """
        Expand English contractions into their full forms for phonemization.
        
        For example: "I'm" -> ["I", "am"], "We're" -> ["We", "are"]
        
        Args:
            word: The word that may be a contraction
            
        Returns:
            A list of phoneme-ready words
        """
        # Common contractions mapping (all lowercase)
        contractions_map = {
            "i'm": ["i", "am"],
            "he's": ["he", "is"],
            "she's": ["she", "is"],
            "it's": ["it", "is"],
            "we're": ["we", "are"],
            "they're": ["they", "are"],
            "you're": ["you", "are"],
            "don't": ["do", "not"],
            "doesn't": ["does", "not"],
            "didn't": ["did", "not"],
            "won't": ["will", "not"],
            "can't": ["can", "not"],
            "couldn't": ["could", "not"],
            "shouldn't": ["should", "not"],
            "wouldn't": ["would", "not"],
            "isn't": ["is", "not"],
            "aren't": ["are", "not"],
            "wasn't": ["was", "not"],
            "weren't": ["were", "not"],
            "haven't": ["have", "not"],
            "hasn't": ["has", "not"],
            "hadn't": ["had", "not"],
            "i'll": ["i", "will"],
            "he'll": ["he", "will"],
            "she'll": ["she", "will"],
            "we'll": ["we", "will"],
            "they'll": ["they", "will"],
            "you'll": ["you", "will"],
            "i've": ["i", "have"],
            "we've": ["we", "have"],
            "they've": ["they", "have"],
            "you've": ["you", "have"],
        }
        
        # Check lowercase version
        word_lower = word.lower()
        if word_lower in contractions_map:
            expanded = contractions_map[word_lower]
            # If the original word was capitalized, capitalize the first expanded word
            if word[0].isupper() and len(expanded) > 0:
                expanded = [expanded[0].capitalize()] + expanded[1:]
            return expanded
        else:
            return [word]