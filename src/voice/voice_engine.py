"""Core voice engine that orchestrates TTS, LLM, and audio output."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from .text_processor import TextProcessor, create_processor_from_config
from .tts import TTSEngine
from .tts.glados_tts import GladosTTSEngine, create_glados_engine_from_existing
from .tts.kokoro_tts import KokoroTTSEngine, create_kokoro_engine_from_existing
from .tts.elevenlabs_tts import ElevenLabsTTSEngine
from .audio_utils import play_audio, save_audio


class VoiceEngine:
    """Main voice engine coordinating all components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize voice engine.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default config.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.text_processor = create_processor_from_config(self.config)
        self.tts_engines: Dict[str, TTSEngine] = {}
        
    def _get_default_config_path(self) -> str:
        """Get the default config path."""
        return str(Path(__file__).parent / "config.yaml")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths relative to config file location."""
        if os.path.isabs(path):
            return path
        
        # Resolve relative to config file directory
        config_dir = Path(self.config_path).parent
        resolved = config_dir / path
        return str(resolved.resolve())
    
    def _create_tts_engine(self, voice_name: str, voice_config: dict) -> TTSEngine:
        """
        Create a TTS engine based on voice configuration.
        
        Args:
            voice_name: Name of the voice preset
            voice_config: Voice configuration dictionary
            
        Returns:
            Initialized TTSEngine instance
        """
        engine_type = voice_config.get("tts_engine", "").lower()
        
        if engine_type == "glados":
            model_path = self._resolve_path(voice_config["model_path"])
            phonemizer_path = voice_config.get("phonemizer_path")
            if phonemizer_path:
                phonemizer_path = self._resolve_path(phonemizer_path)
            
            speed = voice_config.get("speed", 1.0)
            
            # Try to use existing GLaDOS implementation if available
            try:
                engine = create_glados_engine_from_existing(model_path, speed)
            except Exception:
                engine = GladosTTSEngine(model_path, phonemizer_path, speed)
            
        elif engine_type == "kokoro":
            model_path = self._resolve_path(voice_config["model_path"])
            phonemizer_path = voice_config.get("phonemizer_path")
            if phonemizer_path:
                phonemizer_path = self._resolve_path(phonemizer_path)
            
            voice = voice_config.get("voice", "af_bella")
            speed = voice_config.get("speed", 1.0)
            
            # Try to use existing Kokoro implementation if available
            try:
                engine = create_kokoro_engine_from_existing(model_path, voice, speed)
            except Exception:
                engine = KokoroTTSEngine(model_path, phonemizer_path, voice, speed)
            
        elif engine_type == "elevenlabs":
            api_key = voice_config.get("api_key")
            voice_id = voice_config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
            model = voice_config.get("model", "eleven_monolingual_v1")
            
            engine = ElevenLabsTTSEngine(api_key, voice_id, model)
            
        else:
            raise ValueError(f"Unknown TTS engine type: {engine_type}")
        
        # Initialize the engine
        engine.initialize()
        
        return engine
    
    def _get_or_create_engine(self, voice_name: str) -> TTSEngine:
        """
        Get cached TTS engine or create a new one.
        
        Args:
            voice_name: Name of the voice preset
            
        Returns:
            TTSEngine instance
        """
        # Check if engine is already cached
        if voice_name in self.tts_engines:
            return self.tts_engines[voice_name]
        
        # Get voice config
        voices = self.config.get("voices", {})
        if voice_name not in voices:
            available = ", ".join(voices.keys())
            raise ValueError(
                f"Voice preset '{voice_name}' not found in config. "
                f"Available voices: {available}"
            )
        
        voice_config = voices[voice_name]
        
        # Create and cache engine
        engine = self._create_tts_engine(voice_name, voice_config)
        self.tts_engines[voice_name] = engine
        
        return engine
    
    def synthesize(
        self,
        text: str,
        voice_name: str,
        output_file: Optional[str] = None
    ) -> None:
        """
        Synthesize speech from text using specified voice.
        
        Args:
            text: Input text to synthesize
            voice_name: Name of the voice preset to use
            output_file: Optional output file path. If None, plays audio.
        """
        # Get voice configuration
        voices = self.config.get("voices", {})
        if voice_name not in voices:
            available = ", ".join(voices.keys())
            raise ValueError(
                f"Voice preset '{voice_name}' not found. Available: {available}"
            )
        
        voice_config = voices[voice_name]
        
        # Process text with LLM if enabled
        enable_llm = voice_config.get("enable_llm", False)
        system_prompt = voice_config.get("system_prompt")
        
        if enable_llm and system_prompt:
            processed_text = self.text_processor.process(text, system_prompt)
        else:
            processed_text = text
        
        # Get or create TTS engine
        engine = self._get_or_create_engine(voice_name)
        
        # Synthesize speech
        print(f"[Voice Engine] Synthesizing with voice: {voice_name}")
        audio_data = engine.synthesize(processed_text)
        sample_rate = engine.get_sample_rate()
        
        # Output audio
        audio_config = self.config.get("audio", {})
        
        if output_file:
            # Save to file (format auto-detected from extension)
            save_audio(audio_data, sample_rate, output_file, format=None)
        else:
            # Play audio if auto_play is enabled
            auto_play = audio_config.get("auto_play", True)
            if auto_play:
                play_audio(audio_data, sample_rate)
            else:
                print("[Voice Engine] Audio generated but auto_play is disabled")
    
    def list_voices(self) -> list:
        """
        Get list of available voice presets.
        
        Returns:
            List of voice preset names
        """
        voices = self.config.get("voices", {})
        return list(voices.keys())
    
    def get_voice_info(self, voice_name: str) -> dict:
        """
        Get information about a specific voice preset.
        
        Args:
            voice_name: Name of the voice preset
            
        Returns:
            Voice configuration dictionary
        """
        voices = self.config.get("voices", {})
        if voice_name not in voices:
            raise ValueError(f"Voice preset '{voice_name}' not found")
        
        return voices[voice_name]
    
    def cleanup(self) -> None:
        """Clean up all TTS engines."""
        for engine in self.tts_engines.values():
            engine.cleanup()
        self.tts_engines.clear()
