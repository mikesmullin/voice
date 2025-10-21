"""Kokoro TTS engine implementation using ONNX models."""

from pathlib import Path
from typing import Optional, Generator
import numpy as np
from numpy.typing import NDArray
from pickle import load
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cache
import re

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from . import TTSEngine

# Default OnnxRuntime is way to verbose, only show fatal errors
if ort is not None:
    ort.set_default_logger_severity(4)


def get_model_path(relative_path: str) -> Path:
    """Get the absolute path to a model file relative to the models directory."""
    current_dir = Path(__file__).parent.parent.parent
    return current_dir / relative_path


@dataclass
class PhonemizerConfig:
    MODEL_PATH: Path
    PHONEME_DICT_PATH: Path
    TOKEN_TO_IDX_PATH: Path
    IDX_TO_TOKEN_PATH: Path
    CHAR_REPEATS: int = 3
    MODEL_INPUT_LENGTH: int = 64
    EXPAND_ACRONYMS: bool = False

    def __init__(
        self,
        model_path: Optional[Path] = None,
        phoneme_dict_path: Optional[Path] = None,
        token_to_idx_path: Optional[Path] = None,
        idx_to_token_path: Optional[Path] = None,
    ) -> None:
        self.MODEL_PATH = model_path if model_path is not None else get_model_path("models/TTS/phomenizer_en.onnx")
        self.PHONEME_DICT_PATH = (
            phoneme_dict_path if phoneme_dict_path is not None else get_model_path("models/TTS/lang_phoneme_dict.pkl")
        )
        self.TOKEN_TO_IDX_PATH = (
            token_to_idx_path if token_to_idx_path is not None else get_model_path("models/TTS/token_to_idx.pkl")
        )
        self.IDX_TO_TOKEN_PATH = (
            idx_to_token_path if idx_to_token_path is not None else get_model_path("models/TTS/idx_to_token.pkl")
        )


class SpecialTokens(Enum):
    PAD = "_"
    START = "<start>"
    END = "<end>"
    EN_US = "<en_us>"


class Punctuation(Enum):
    PUNCTUATION = "().,:?!/–"
    HYPHEN = "-"
    SPACE = " "

    @classmethod
    @cache
    def get_punc_set(cls) -> set[str]:
        return set(cls.PUNCTUATION.value + cls.HYPHEN.value + cls.SPACE.value)

    @classmethod
    @cache
    def get_punc_pattern(cls) -> re.Pattern[str]:
        return re.compile(f"([{cls.PUNCTUATION.value + cls.SPACE.value}])")


class Phonemizer:
    """Phonemizer for converting text to phonemes using ONNX model."""

    def __init__(self, config: Optional[PhonemizerConfig] = None) -> None:
        if config is None:
            config = PhonemizerConfig()
        self.config = config
        self.phoneme_dict: dict[str, str] = self._load_pickle(self.config.PHONEME_DICT_PATH)
        self.token_to_idx = self._load_pickle(self.config.TOKEN_TO_IDX_PATH)
        self.idx_to_token = self._load_pickle(self.config.IDX_TO_TOKEN_PATH)

        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")

        self.ort_session = ort.InferenceSession(
            str(self.config.MODEL_PATH),
            sess_options=ort.SessionOptions(),
            providers=providers,
        )

        self.special_tokens: set[str] = {
            SpecialTokens.PAD.value,
            SpecialTokens.END.value,
            SpecialTokens.EN_US.value,
        }

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        with open(path, "rb") as f:
            return load(f)

    @staticmethod
    def _unique_consecutive(arr: list[NDArray]) -> list[NDArray]:
        result = []
        for row in arr:
            if len(row) == 0:
                result.append(row)
            else:
                mask = np.concatenate(([True], row[1:] != row[:-1]))
                result.append(row[mask])
        return result

    @staticmethod
    def _remove_padding(arr: list[NDArray], padding_value: int = 0) -> list[NDArray]:
        return [row[row != padding_value] for row in arr]

    @staticmethod
    def _trim_to_stop(arr: list[NDArray], end_index: int = 2) -> list[NDArray]:
        result = []
        for row in arr:
            stop_index = np.where(row == end_index)[0]
            if len(stop_index) > 0:
                result.append(row[: stop_index[0] + 1])
            else:
                result.append(row)
        return result

    def _process_model_output(self, arr: list[NDArray]) -> list[NDArray]:
        arr_processed = np.argmax(arr[0], axis=2)
        arr_processed = self._unique_consecutive(arr_processed)
        arr_processed = self._remove_padding(arr_processed)
        arr_processed = self._trim_to_stop(arr_processed)
        return arr_processed

    def encode(self, sentence: Iterable[str]) -> list[int]:
        sentence = [item for item in sentence for _ in range(self.config.CHAR_REPEATS)]
        sentence = [s.lower() for s in sentence]
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        return [
            self.token_to_idx[SpecialTokens.START.value],
            *sequence,
            self.token_to_idx[SpecialTokens.END.value],
        ]

    def decode(self, sequence: NDArray) -> str:
        decoded = []
        for t in sequence:
            idx = t.item() if hasattr(t, 'item') else t
            token = self.idx_to_token[idx]
            decoded.append(token)
        result = "".join(d for d in decoded if d not in self.special_tokens)
        return result

    @staticmethod
    def pad_sequence_fixed(v: list[list[int]], target_length: int) -> NDArray:
        result: NDArray = np.zeros((len(v), target_length), dtype=np.int64)
        for i, seq in enumerate(v):
            length = min(len(seq), target_length)
            result[i, :length] = seq[:length]
        return result

    def _get_dict_entry(self, word: str, punc_set: set[str]) -> Optional[str]:
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

    @staticmethod
    def _get_phonemes(word: str, word_phonemes: dict, word_splits: dict) -> str:
        phons = word_phonemes[word]
        if phons is None:
            subwords = word_splits[word]
            subphons_converted = [word_phonemes[w] for w in subwords]
            phons = "".join([subphon for subphon in subphons_converted if subphon is not None])
        return phons

    def _clean_and_split_texts(self, texts: list[str], punc_set: set[str], punc_pattern) -> tuple:
        split_text, cleaned_words = [], set()
        for text in texts:
            cleaned_text = "".join(t for t in text if t.isalnum() or t in punc_set)
            split = [s for s in re.split(punc_pattern, cleaned_text) if len(s) > 0]
            split_text.append(split)
            cleaned_words.update(split)
        return split_text, cleaned_words

    def convert_to_phonemes(self, texts: list[str], lang: str = "en_us") -> list[str]:
        punc_set = Punctuation.get_punc_set()
        punc_pattern = Punctuation.get_punc_pattern()

        split_text, cleaned_words = self._clean_and_split_texts(texts, punc_set, punc_pattern)

        for punct in punc_set:
            self.phoneme_dict[punct] = punct
        word_phonemes = {word: self.phoneme_dict.get(word.lower()) for word in cleaned_words}

        words_to_split = [w for w in cleaned_words if word_phonemes[w] is None]
        word_splits = {
            key: re.split(r"([-])", word)
            for key, word in zip(words_to_split, words_to_split, strict=False)
        }

        subwords = {w for values in word_splits.values() for w in values if w not in word_phonemes}
        for subword in subwords:
            word_phonemes[subword] = self._get_dict_entry(word=subword, punc_set=punc_set)

        words_to_predict = [
            word for word, phons in word_phonemes.items() if phons is None and len(word_splits.get(word, [])) <= 1
        ]

        if words_to_predict:
            input_batch = [self.encode(word) for word in words_to_predict]
            input_batch_padded = self.pad_sequence_fixed(input_batch, self.config.MODEL_INPUT_LENGTH)

            ort_inputs = {self.ort_session.get_inputs()[0].name: input_batch_padded}
            ort_outs = self.ort_session.run(None, ort_inputs)

            ids = self._process_model_output(ort_outs)
            for id, word in zip(ids, words_to_predict, strict=False):
                word_phonemes[word] = self.decode(id)

        phoneme_lists = []
        for text in split_text:
            text_phons = [
                self._get_phonemes(word=word, word_phonemes=word_phonemes, word_splits=word_splits) for word in text
            ]
            phoneme_lists.append(text_phons)

        return ["".join(phoneme_list) for phoneme_list in phoneme_lists]

    def __del__(self) -> None:
        if hasattr(self, "ort_session"):
            del self.ort_session


class KokoroSynthesizer:
    """Kokoro speech synthesizer using ONNX model."""

    MODEL_PATH: Path = get_model_path("models/TTS/kokoro-v1.0.fp16.onnx")
    VOICES_PATH: Path = get_model_path("models/TTS/kokoro-voices-v1.0.bin")
    DEFAULT_VOICE: str = "af_bella"
    MAX_PHONEME_LENGTH: int = 510
    SAMPLE_RATE: int = 24000

    def __init__(self, model_path: Optional[Path] = None, voice: str = DEFAULT_VOICE) -> None:
        self.sample_rate = self.SAMPLE_RATE
        voices_file = np.load(str(self.VOICES_PATH), allow_pickle=True)
        self.voices: dict = {k: voices_file[k] for k in voices_file.files}
        self.vocab = self._get_vocab()
        self.set_voice(voice)

        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")

        model_to_use = str(model_path if model_path is not None else self.MODEL_PATH)
        self.ort_sess = ort.InferenceSession(
            model_to_use,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        self.phonemizer = Phonemizer()

    def set_voice(self, voice: str) -> None:
        if voice not in self.voices:
            raise ValueError(f"Voice '{voice}' not found. Available voices: {list(self.voices.keys())}")
        self.voice = voice

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences, preserving sentence-ending punctuation."""
        # Split on sentence boundaries but keep the delimiter
        sentences = re.split(r'([.!?]+(?:\s+|$))', text)
        
        # Combine the sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if sentences[i].strip():
                sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
                result.append(sentence.strip())
        
        # Handle last sentence if it doesn't end with punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result

    def _chunk_text_by_phonemes(self, text: str) -> list[str]:
        """
        Split text into chunks that fit within the phoneme limit.
        Tries to split at sentence boundaries when possible.
        
        Returns:
            List of text chunks, each guaranteed to be under the phoneme limit
        """
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            # Convert sentence to phonemes to check length
            test_text = ' '.join(current_chunk + [sentence])
            phonemes = self.phonemizer.convert_to_phonemes([test_text], "en_us")[0]
            
            if len(phonemes) <= self.MAX_PHONEME_LENGTH:
                # Sentence fits, add it to current chunk
                current_chunk.append(sentence)
            else:
                # Sentence doesn't fit
                if current_chunk:
                    # Save current chunk and start new one with this sentence
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    
                    # Check if single sentence is too long
                    phonemes = self.phonemizer.convert_to_phonemes([sentence], "en_us")[0]
                    if len(phonemes) > self.MAX_PHONEME_LENGTH:
                        # Single sentence is too long, need to split it further
                        # Split by words as a fallback
                        chunks.pop() if not chunks or chunks[-1] != ' '.join(current_chunk[:-1]) else None
                        chunks.extend(self._chunk_long_sentence(sentence))
                        current_chunk = []
                else:
                    # First sentence is too long, need to split it
                    chunks.extend(self._chunk_long_sentence(sentence))
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _chunk_long_sentence(self, sentence: str) -> list[str]:
        """
        Split a single long sentence into smaller chunks by words.
        Used when a sentence exceeds the phoneme limit.
        """
        words = sentence.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            test_text = ' '.join(current_chunk + [word])
            phonemes = self.phonemizer.convert_to_phonemes([test_text], "en_us")[0]
            
            if len(phonemes) <= self.MAX_PHONEME_LENGTH:
                current_chunk.append(word)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                else:
                    # Single word is too long (rare), just include it
                    chunks.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def generate_speech_audio_chunked(self, text: str) -> Generator[NDArray[np.float32], None, None]:
        """
        Generate speech audio in chunks for long text.
        Yields audio chunks one at a time for memory efficiency.
        
        Args:
            text: Input text (can be any length)
            
        Yields:
            Audio waveform chunks as numpy arrays
        """
        # Check if text needs chunking
        phonemes = self.phonemizer.convert_to_phonemes([text], "en_us")[0]
        
        if len(phonemes) <= self.MAX_PHONEME_LENGTH:
            # Text is short enough, process as single chunk
            phoneme_ids = self._phonemes_to_ids(phonemes)
            audio = self._synthesize_ids_to_audio(phoneme_ids)
            yield np.array(audio, dtype=np.float32)
        else:
            # Text is too long, split into chunks
            chunks = self._chunk_text_by_phonemes(text)
            print(f"[Kokoro] Text split into {len(chunks)} chunks for processing")
            
            for i, chunk in enumerate(chunks, 1):
                print(f"[Kokoro] Processing chunk {i}/{len(chunks)}: {chunk[:50]}...")
                phonemes = self.phonemizer.convert_to_phonemes([chunk], "en_us")[0]
                phoneme_ids = self._phonemes_to_ids(phonemes)
                audio = self._synthesize_ids_to_audio(phoneme_ids)
                yield np.array(audio, dtype=np.float32)

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """
        Generate speech audio (legacy method for backward compatibility).
        For long texts, this concatenates all chunks in memory.
        Consider using generate_speech_audio_chunked() for memory efficiency.
        """
        chunks = list(self.generate_speech_audio_chunked(text))
        if len(chunks) == 1:
            return chunks[0]
        else:
            # Concatenate all chunks
            return np.concatenate(chunks)

    @staticmethod
    def _get_vocab() -> dict[str, int]:
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = (
            "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻ"
            "ʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        )
        symbols = [_pad, *_punctuation, *_letters, *_letters_ipa]
        dicts = {}
        for i in range(len(symbols)):
            dicts[symbols[i]] = i
        return dicts

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        # Note: Chunking is handled at a higher level, but keep validation for safety
        if len(phonemes) > self.MAX_PHONEME_LENGTH:
            raise ValueError(
                f"text chunk is too long ({len(phonemes)} phonemes), "
                f"must be less than {self.MAX_PHONEME_LENGTH} phonemes. "
                f"This should not happen if chunking is working correctly."
            )
        return [i for i in map(self.vocab.get, phonemes) if i is not None]

    def _synthesize_ids_to_audio(self, ids: list[int]) -> NDArray[np.float32]:
        voice_vector = self.voices[self.voice]
        voice_array = voice_vector[len(ids)]

        tokens = [[0, *ids, 0]]
        speed = 1.0
        audio = self.ort_sess.run(
            None,
            {
                "tokens": tokens,
                "style": voice_array,
                "speed": np.ones(1, dtype=np.float32) * speed,
            },
        )[0]
        return np.array(audio[:-8000], dtype=np.float32)

    def __del__(self) -> None:
        if hasattr(self, "ort_sess"):
            del self.ort_sess


class KokoroTTSEngine(TTSEngine):
    """Kokoro multi-voice TTS engine using ONNX models."""

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
            phonemizer_path: Path to phonemizer ONNX model (not used, kept for compatibility)
            voice: Voice variant to use
            speed: Speech speed multiplier (not used in current Kokoro implementation)
        """
        self.model_path = Path(model_path)
        self.voice = voice
        self.speed = speed
        self.synthesizer = None

    def initialize(self) -> None:
        """Initialize the Kokoro synthesizer."""
        if ort is None:
            raise ImportError("onnxruntime is required for Kokoro TTS. Install with: pip install onnxruntime")

        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Kokoro model not found at {self.model_path}")

        try:
            self.synthesizer = KokoroSynthesizer(model_path=self.model_path, voice=self.voice)
            print(f"[Kokoro TTS] Initialized with voice: {self.voice}")
        except Exception as e:
            print(f"[Kokoro TTS] Error initializing: {e}")
            raise

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech using Kokoro model.
        
        Args:
            text: Input text
            
        Returns:
            Audio waveform as numpy array
        """
        if self.synthesizer is None:
            raise RuntimeError("TTS engine not initialized. Call initialize() first.")

        print(f"[Kokoro TTS - {self.voice}] Synthesizing: {text}")
        audio = self.synthesizer.generate_speech_audio(text)
        return audio

    def synthesize_streaming(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Synthesize speech in chunks for memory efficiency.
        
        Args:
            text: Input text
            
        Yields:
            Audio waveform chunks as numpy arrays
        """
        if self.synthesizer is None:
            raise RuntimeError("TTS engine not initialized. Call initialize() first.")

        print(f"[Kokoro TTS - {self.voice}] Synthesizing (streaming): {text[:100]}...")
        for audio_chunk in self.synthesizer.generate_speech_audio_chunked(text):
            yield audio_chunk

    def get_sample_rate(self) -> int:
        """Get sample rate for Kokoro voice."""
        return 24000

    def cleanup(self) -> None:
        """Clean up ONNX sessions."""
        if self.synthesizer is not None:
            self.synthesizer = None
