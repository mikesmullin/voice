"""Kokoro TTS engine wrapper."""

import os
import warnings
import logging
import random
from typing import Optional
import numpy as np

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


class KokoroEngine:
    """Kokoro TTS engine."""
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize Kokoro engine.
        
        Args:
            force_cpu: Force CPU usage instead of GPU
        """
        self.force_cpu = force_cpu
        self.pipeline = None
        self.last_voice_id = None  # Track last loaded voice
    
    def _initialize_pipeline(self) -> None:
        """Initialize the Kokoro pipeline (lazy loading)."""
        if self.pipeline is None:
            from .timing import log
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
    
    def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text using Kokoro TTS.
        
        Args:
            text: Text to synthesize
            voice_id: Kokoro voice ID (e.g., "af_bella", "am_adam")
            speed: Speech speed multiplier (default: 1.0)
            
        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        from .timing import log, get_elapsed
        
        self._initialize_pipeline()
        
        # Check if we need to load a different voice
        if voice_id != self.last_voice_id:
            log(f"[Kokoro TTS] Loading voice '{voice_id}'...")
            load_start = get_elapsed()
            # Kokoro loads the voice on first use, we'll see the timing
            self.last_voice_id = voice_id
            log(f"[Kokoro TTS] Voice load preparation took {get_elapsed() - load_start:.2f}s")
        
        # Add filler prefix to prevent clipping
        prefixed_text = self._add_filler_prefix(text)
        
        log(f"[Kokoro TTS] Generating speech with voice '{voice_id}' (speed={speed})...")
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
        
        # Concatenate audio chunks
        audio_data = np.concatenate(audio_chunks)
        sample_rate = 24000  # Kokoro uses 24kHz
        
        return audio_data, sample_rate
