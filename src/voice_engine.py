"""Simple voice engine using Kokoro TTS library."""

import os
import warnings
import logging
import random
from pathlib import Path
from typing import Optional
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

# Suppress huggingface_hub logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from kokoro import KPipeline
except ImportError:
    raise ImportError("kokoro package not installed. Install with: pip install kokoro")

from .audio_utils import play_audio, save_audio
from .timing import log


class VoiceEngine:
    """Main voice engine using Kokoro TTS."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.pipeline = None
        
    def _get_default_config_path(self) -> str:
        return str(Path(__file__).parent / "config.yaml")
    
    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _initialize_pipeline(self) -> None:
        if self.pipeline is None:
            self.pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')  # English
    
    def _add_filler_prefix(self, text: str) -> str:
        """Add a random filler word prefix to ease into speech naturally."""
        fillers = [
            #"Um,", "Well,", "Uh,", "So,", "Like,", 
            #"You know,", "Hmm,", "Okay,", "Alright,", "Er,"
            "... ... "
        ]
        filler = random.choice(fillers)
        return f"{filler} {text}"
    
    def synthesize(self, text: str, voice_name: str, output_file: Optional[str] = None) -> None:
        self._initialize_pipeline()
        
        voices = self.config.get("voices", {})
        if voice_name not in voices:
            available = ", ".join(voices.keys())
            raise ValueError(
                f"Voice preset '{voice_name}' not found in config. "
                f"Available voices: {available}"
            )
        
        voice_config = voices[voice_name]
        voice_id = voice_config.get("voice")
        speed = voice_config.get("speed", 1.0)
        
        # Add filler prefix to prevent clipping
        prefixed_text = self._add_filler_prefix(text)
        
        log(f"[Kokoro TTS] Generating speech with voice '{voice_name}' ({voice_id})...")
        
        # Generate speech using Kokoro
        audio_generator = self.pipeline(prefixed_text, voice=voice_id, speed=speed)
        
        # Collect audio chunks
        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(audio_generator):
            audio_chunks.append(audio)
        
        if not audio_chunks:
            raise RuntimeError("No audio generated")
        
        # Concatenate audio chunks
        import numpy as np
        audio_data = np.concatenate(audio_chunks)
        sample_rate = 24000  # Kokoro uses 24kHz
        
        audio_config = self.config.get("audio", {})
        
        if output_file:
            audio_format = audio_config.get("format", "wav")
            save_audio(audio_data, sample_rate, output_file, audio_format)
        else:
            if audio_config.get("auto_play", True):
                play_audio(audio_data, sample_rate, speed)
    
    def list_voices(self) -> list:
        voices = self.config.get("voices", {})
        return sorted(voices.keys())
    
    def get_voice_info(self, voice_name: str) -> dict:
        voices = self.config.get("voices", {})
        if voice_name not in voices:
            raise ValueError(f"Voice '{voice_name}' not found")
        
        return voices[voice_name]
