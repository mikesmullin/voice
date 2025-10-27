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
    
    def __init__(self, config_path: Optional[str] = None, force_cpu: bool = False):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.pipeline = None
        self.last_voice_id = None  # Track last loaded voice
        self.force_cpu = force_cpu  # Force CPU usage
        
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
            import torch
            if self.force_cpu:
                device = "cpu"
                log(f"[Kokoro TTS] Initializing pipeline on device: cpu (forced)")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                log(f"[Kokoro TTS] Initializing pipeline on device: {device}")
            
            self.pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')  # English
            
            # Force CPU if requested
            if self.force_cpu and hasattr(self.pipeline, 'model'):
                self.pipeline.model = self.pipeline.model.cpu()
            
            log(f"[Kokoro TTS] Pipeline initialized")
    
    def _add_filler_prefix(self, text: str) -> str:
        """Add a random filler word prefix to ease into speech naturally."""
        fillers = [
            #"Um,", "Well,", "Uh,", "So,", "Like,", 
            #"You know,", "Hmm,", "Okay,", "Alright,", "Er,"
            "... ... "
        ]
        filler = random.choice(fillers)
        return f"{filler} {text}"
    
    def synthesize(self, text: str, voice_name: str, output_file: Optional[str] = None, stinger: Optional[str] = None) -> None:
        from .timing import get_elapsed
        
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
        
        # Resolve and load stinger early (but don't play yet)
        stinger_audio_data = None
        stinger_sample_rate = None
        if stinger or voice_config.get("default_stinger"):
            # Use CLI-provided stinger if available, otherwise use default_stinger from config
            stinger_name = stinger or voice_config.get("default_stinger")
            stingers = voice_config.get("stingers", {})
            
            if stinger_name in stingers:
                # Resolve path relative to project root (parent of config directory)
                config_dir = Path(self.config_path).parent
                project_root = config_dir.parent  # Go up one level from src/
                stinger_path = project_root / stingers[stinger_name]
                log(f"[Voice] Stinger '{stinger_name}' resolved to: {stinger_path}")
                
                # Load stinger now (but don't play yet) if not saving to file
                if not output_file:
                    try:
                        from .audio_utils import load_stinger
                        stinger_audio_data, stinger_sample_rate = load_stinger(str(stinger_path))
                    except Exception as e:
                        log(f"[Voice] Warning: Could not load stinger: {e}")
            elif stinger:
                # CLI provided a stinger name that doesn't exist in config - no-op (ignore)
                log(f"[Voice] Stinger '{stinger_name}' not found in config, ignoring")
        
        # Check if we need to load a different voice
        if voice_id != self.last_voice_id:
            log(f"[Kokoro TTS] Loading voice '{voice_name}' ({voice_id})...")
            load_start = get_elapsed()
            # Kokoro loads the voice on first use, we'll see the timing
            self.last_voice_id = voice_id
            log(f"[Kokoro TTS] Voice load preparation took {get_elapsed() - load_start:.2f}s")
        
        # Add filler prefix to prevent clipping
        prefixed_text = self._add_filler_prefix(text)
        
        log(f"[Kokoro TTS] Generating speech with voice '{voice_name}' ({voice_id})...")
        gen_start = get_elapsed()
        
        # Generate speech using Kokoro
        audio_generator = self.pipeline(prefixed_text, voice=voice_id, speed=speed)
        
        # Collect audio chunks
        audio_chunks = []
        chunk_start = get_elapsed()
        for i, (gs, ps, audio) in enumerate(audio_generator):
            audio_chunks.append(audio)
        
        log(f"[Kokoro TTS] Audio collection took {get_elapsed() - chunk_start:.2f}s")
        log(f"[Kokoro TTS] Total generation took {get_elapsed() - gen_start:.2f}s")
        
        if not audio_chunks:
            raise RuntimeError("No audio generated")
        
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
            # Play stinger right before playing synthesized audio
            if stinger_audio_data is not None:
                import sounddevice as sd
                default_device = sd.default.device[1]  # Output device
                device_info = sd.query_devices(default_device)
                
                log(f"[Voice] Playing stinger on: {device_info['name']}")
                
                try:
                    # Ensure audio is the right shape
                    if len(stinger_audio_data.shape) == 2 and stinger_audio_data.shape[1] == 1:
                        stinger_audio_data = stinger_audio_data.flatten()
                    
                    # Play stinger and wait for completion
                    sd.play(stinger_audio_data, samplerate=stinger_sample_rate, device=default_device, blocking=True)
                    log(f"[Voice] Stinger playback complete")
                except Exception as e:
                    log(f"[Voice] Warning: Could not play stinger: {e}")
            
            # Now play the synthesized audio
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
