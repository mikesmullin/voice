"""
GLaDOS TTS synthesizer - integrated from parent GLaDOS project.
This module provides the SpeechSynthesizer class for GLaDOS voice synthesis.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from pickle import load
from typing import Any

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

# Import phonemizer from the tts package
from .phonemizer import Phonemizer

# Suppress ONNX Runtime verbosity
ort.set_default_logger_severity(4)


@dataclass
class PiperConfig:
    """Piper configuration for GLaDOS TTS."""

    num_symbols: int
    num_speakers: int
    sample_rate: int
    espeak_voice: str
    length_scale: float
    noise_scale: float
    noise_w: float
    phoneme_id_map: Mapping[str, Sequence[int]]
    speaker_id_map: dict[str, int] | None = None

    @staticmethod
    def from_dict(config: dict[str, Any]) -> "PiperConfig":
        """Create a PiperConfig instance from a configuration dictionary."""
        inference = config.get("inference", {})

        return PiperConfig(
            num_symbols=config["num_symbols"],
            num_speakers=config["num_speakers"],
            sample_rate=config["audio"]["sample_rate"],
            noise_scale=inference.get("noise_scale", 0.667),
            length_scale=inference.get("length_scale", 1.0),
            noise_w=inference.get("noise_w", 0.8),
            espeak_voice=config["espeak"]["voice"],
            phoneme_id_map=config["phoneme_id_map"],
            speaker_id_map=config.get("speaker_id_map", {}),
        )


class GladosSynthesizer:
    """
    GLaDOS Text-to-Speech Synthesizer based on the VITS model.
    Trained using the Piper project (https://github.com/rhasspy/piper)
    """

    # Constants
    MAX_WAV_VALUE = 32767.0
    
    # Special tokens
    PAD = "_"  # padding (0)
    BOS = "^"  # beginning of sentence
    EOS = "$"  # end of sentence

    def __init__(self, model_path: Path, phoneme_path: Path, speaker_id: int | None = None) -> None:
        """
        Initialize the GLaDOS TTS synthesizer.

        Args:
            model_path: Path to the ONNX model file
            phoneme_path: Path to the phoneme-to-ID mapping file
            speaker_id: Optional speaker ID for multi-speaker models
        """
        # Setup ONNX Runtime providers
        providers = ort.get_available_providers()
        # Remove problematic providers
        for provider in ["TensorrtExecutionProvider", "CoreMLExecutionProvider"]:
            if provider in providers:
                providers.remove(provider)

        self.ort_sess = ort.InferenceSession(
            str(model_path),
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        
        self.phonemizer = Phonemizer()
        self.id_map = self._load_pickle(phoneme_path)

        # Load configuration
        config_file_path = model_path.with_suffix(".json")
        try:
            with open(config_file_path, encoding="utf-8") as config_file:
                config_dict = json.load(config_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at path: {config_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Configuration file at {config_file_path} is not valid JSON. Error: {e}")

        self.config = PiperConfig.from_dict(config_dict)
        self.sample_rate = self.config.sample_rate
        self.speaker_id = (
            self.config.speaker_id_map.get(str(speaker_id), 0)
            if self.config.num_speakers > 1 and self.config.speaker_id_map is not None
            else None
        )

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        """Load a pickled dictionary from the specified file path."""
        with path.open("rb") as f:
            return dict(load(f))

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """
        Convert input text to synthesized speech audio.

        Args:
            text: The text to be converted to speech

        Returns:
            An array of audio samples representing the synthesized speech
        """
        phonemes = self._phonemizer(text)
        phoneme_ids_list = [self._phonemes_to_ids(sentence) for sentence in phonemes]
        audio_chunks = [self._synthesize_ids_to_audio(phoneme_ids) for phoneme_ids in phoneme_ids_list]

        if audio_chunks:
            # Concatenate all 1D audio chunks
            audio: NDArray[np.float32] = np.concatenate(audio_chunks)
            return audio
        return np.array([], dtype=np.float32)

    def _phonemizer(self, input_text: str) -> list[str]:
        """Convert input text to phonemes using espeak-ng phonemization."""
        phonemes = self.phonemizer.convert_to_phonemes([input_text], "en_us")
        return phonemes

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        """Convert a sequence of phonemes to their corresponding integer IDs."""
        ids: list[int] = list(self.id_map[self.BOS])

        for phoneme in phonemes:
            if phoneme not in self.id_map:
                continue
            ids.extend(self.id_map[phoneme])
            ids.extend(self.id_map[self.PAD])
            
        ids.extend(self.id_map[self.EOS])
        return ids

    def _synthesize_ids_to_audio(
        self,
        phoneme_ids: list[int],
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> NDArray[np.float32]:
        """Synthesize raw audio from phoneme IDs using the VITS model."""
        if length_scale is None:
            length_scale = self.config.length_scale
        if noise_scale is None:
            noise_scale = self.config.noise_scale
        if noise_w is None:
            noise_w = self.config.noise_w

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)

        scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

        sid = None
        if self.speaker_id is not None:
            sid = np.array([self.speaker_id], dtype=np.int64)

        # Synthesize through ONNX
        audio: NDArray[np.float32] = self.ort_sess.run(
            None,
            {
                "input": phoneme_ids_array,
                "input_lengths": phoneme_ids_lengths,
                "scales": scales,
                "sid": sid,
            },
        )[0].squeeze()

        return audio

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "ort_sess"):
            del self.ort_sess
