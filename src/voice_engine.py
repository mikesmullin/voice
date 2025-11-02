"""Voice engine adapter - routes to different TTS model implementations."""

import os
from pathlib import Path
from typing import Optional
import yaml

from .audio_utils import play_audio, save_audio
from .timing import log


class VoiceEngine:
    """
    Main voice engine adapter that routes to different TTS model implementations.
    
    Supports:
    - Kokoro TTS (default): Fast, preset voices
    - StyleTTS2: Zero-shot voice cloning with reference audio
    """
    
    def __init__(self, config_path: Optional[str] = None, force_cpu: bool = False):
        """
        Initialize the voice engine adapter.
        
        Args:
            config_path: Path to config.yaml file
            force_cpu: Force CPU usage instead of GPU
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.force_cpu = force_cpu
        
        # Lazy-loaded engine instances
        self.kokoro_engine = None
        self.styletts2_engine = None
        
    def _get_default_config_path(self) -> str:
        """Get default path to config.yaml."""
        return str(Path(__file__).parent / "config.yaml")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_kokoro_engine(self):
        """Get or create Kokoro engine instance (lazy loading)."""
        if self.kokoro_engine is None:
            from .kokoro_engine import KokoroEngine
            self.kokoro_engine = KokoroEngine(force_cpu=self.force_cpu)
        return self.kokoro_engine
    
    def _get_styletts2_engine(self):
        """Get or create StyleTTS2 engine instance (lazy loading)."""
        if self.styletts2_engine is None:
            from .styletts2_engine import StyleTTS2Engine
            log(f"[StyleTTS2] Initializing StyleTTS2 engine...")
            self.styletts2_engine = StyleTTS2Engine(force_cpu=self.force_cpu)
        return self.styletts2_engine
    
    def _resolve_path(self, relative_path: str) -> Path:
        """
        Resolve a path relative to the project root.
        
        Args:
            relative_path: Path relative to project root
            
        Returns:
            Absolute Path object
        """
        config_dir = Path(self.config_path).parent
        project_root = config_dir.parent  # Go up one level from src/
        return project_root / relative_path
    
    def _load_stinger(self, voice_config: dict, stinger: Optional[str], output_file: Optional[str]) -> tuple[Optional[object], Optional[int]]:
        """
        Load stinger audio if needed.
        
        Args:
            voice_config: Voice configuration dictionary
            stinger: CLI-provided stinger name (optional)
            output_file: Output file path (stingers not used when saving)
            
        Returns:
            Tuple of (stinger_audio_data, stinger_sample_rate) or (None, None)
        """
        # Don't load stinger if saving to file
        if output_file:
            return None, None
        
        # Check if stinger should be used
        if not (stinger or voice_config.get("default_stinger")):
            return None, None
        
        stinger_name = stinger or voice_config.get("default_stinger")
        stingers = voice_config.get("stingers", {})
        
        if stinger_name in stingers:
            stinger_path = self._resolve_path(stingers[stinger_name])
            log(f"[Voice] Stinger '{stinger_name}' resolved to: {stinger_path}")
            
            try:
                from .audio_utils import load_stinger
                return load_stinger(str(stinger_path))
            except Exception as e:
                log(f"[Voice] Warning: Could not load stinger: {e}")
                return None, None
        elif stinger:
            log(f"[Voice] Stinger '{stinger_name}' not found in config, ignoring")
        
        return None, None
    
    def _play_stinger(self, stinger_audio_data, stinger_sample_rate):
        """Play stinger audio."""
        if stinger_audio_data is None:
            return
        
        import sounddevice as sd
        from .audio_utils import with_interrupt_handler, _playback_interrupted
        
        default_device = sd.default.device[1]  # Output device
        device_info = sd.query_devices(default_device)
        
        log(f"[Voice] Playing stinger on: {device_info['name']}")
        
        @with_interrupt_handler
        def _do_playback():
            # Ensure audio is the right shape
            if len(stinger_audio_data.shape) == 2 and stinger_audio_data.shape[1] == 1:
                audio_to_play = stinger_audio_data.flatten()
            else:
                audio_to_play = stinger_audio_data
            
            # Play stinger and wait for completion
            sd.play(audio_to_play, samplerate=stinger_sample_rate, device=default_device, blocking=True)
            
            if not _playback_interrupted:
                log(f"[Voice] Stinger playback complete")
        
        try:
            _do_playback()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"[Voice] Warning: Could not play stinger: {e}")
    
    def synthesize(self, text: str, voice_name: str, output_file: Optional[str] = None, stinger: Optional[str] = None) -> None:
        """
        Synthesize speech from text using the configured TTS model.
        
        Args:
            text: Text to synthesize
            voice_name: Voice preset name from config
            output_file: Optional output file path
            stinger: Optional stinger sound effect name
        """
        # Get voice configuration
        voices = self.config.get("voices", {})
        if voice_name not in voices:
            available = ", ".join(voices.keys())
            raise ValueError(
                f"Voice preset '{voice_name}' not found in config. "
                f"Available voices: {available}"
            )
        
        voice_config = voices[voice_name]
        model_type = voice_config.get("model", "kokoro")
        
        # Route to appropriate engine
        if model_type == "styletts2":
            self._synthesize_styletts2(text, voice_name, voice_config, output_file, stinger)
        elif model_type == "kokoro":
            self._synthesize_kokoro(text, voice_name, voice_config, output_file, stinger)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _synthesize_kokoro(self, text: str, voice_name: str, voice_config: dict, output_file: Optional[str], stinger: Optional[str]) -> None:
        """Synthesize using Kokoro TTS engine."""
        engine = self._get_kokoro_engine()
        
        # Get Kokoro-specific parameters
        voice_id = voice_config.get("voice")
        if not voice_id:
            raise ValueError(f"Voice preset '{voice_name}' missing 'voice' parameter")
        
        speed = voice_config.get("speed", 1.0)
        
        # Load stinger if needed
        stinger_audio_data, stinger_sample_rate = self._load_stinger(voice_config, stinger, output_file)
        
        # Synthesize audio
        audio_data, sample_rate = engine.synthesize(text, voice_id, speed)
        
        # Output or playback
        audio_config = self.config.get("audio", {})
        
        if output_file:
            audio_format = audio_config.get("format", "wav")
            save_audio(audio_data, sample_rate, output_file, audio_format)
        else:
            self._play_stinger(stinger_audio_data, stinger_sample_rate)
            
            if audio_config.get("auto_play", True):
                play_audio(audio_data, sample_rate, speed)
    
    def _synthesize_styletts2(self, text: str, voice_name: str, voice_config: dict, output_file: Optional[str], stinger: Optional[str]) -> None:
        """Synthesize using StyleTTS2 engine with zero-shot voice cloning."""
        engine = self._get_styletts2_engine()
        
        # Get reference audio path
        reference_audio = voice_config.get("reference_audio")
        if not reference_audio:
            raise ValueError(
                f"Voice preset '{voice_name}' is configured to use StyleTTS2 but "
                "no 'reference_audio' path is specified in config"
            )
        
        ref_audio_path = self._resolve_path(reference_audio)
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")
        
        log(f"[StyleTTS2] Using reference audio: {ref_audio_path}")
        
        # Load stinger if needed
        stinger_audio_data, stinger_sample_rate = self._load_stinger(voice_config, stinger, output_file)
        
        # Get StyleTTS2 parameters (with defaults)
        alpha = voice_config.get("alpha", 0.3)
        beta = voice_config.get("beta", 0.7)
        diffusion_steps = voice_config.get("diffusion_steps", 5)
        embedding_scale = voice_config.get("embedding_scale", 1.0)
        
        # Synthesize audio
        audio_data = engine.synthesize(
            text=text,
            ref_audio_path=str(ref_audio_path),
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale
        )
        
        sample_rate = 24000  # StyleTTS2 uses 24kHz
        audio_config = self.config.get("audio", {})
        
        if output_file:
            audio_format = audio_config.get("format", "wav")
            save_audio(audio_data, sample_rate, output_file, audio_format)
        else:
            self._play_stinger(stinger_audio_data, stinger_sample_rate)
            
            if audio_config.get("auto_play", True):
                # Note: speed parameter not used with StyleTTS2
                play_audio(audio_data, sample_rate, speed=1.0)
    
    def list_voices(self) -> list:
        """List all available voice presets."""
        voices = self.config.get("voices", {})
        return sorted(voices.keys())
    
    def get_voice_info(self, voice_name: str) -> dict:
        """Get configuration for a specific voice preset."""
        voices = self.config.get("voices", {})
        if voice_name not in voices:
            raise ValueError(f"Voice '{voice_name}' not found")
        
        return voices[voice_name]
