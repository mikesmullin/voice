#!/usr/bin/env python3
"""
Piper Voice Comparison Test - Hear ALL available English voices.

This test downloads and plays samples from all available Piper English voices
(US and GB accents) across all quality levels (low, medium, high).

Usage:
    # Play all voices (takes a while - downloads ~2GB of models)
    uv run python test/test_voice_comparison.py

    # Play only US voices
    uv run python test/test_voice_comparison.py --us-only

    # Play only GB voices
    uv run python test/test_voice_comparison.py --gb-only

    # Play only a specific quality
    uv run python test/test_voice_comparison.py --quality medium

    # Play a specific voice
    uv run python test/test_voice_comparison.py --voice lessac

    # Skip download confirmation
    uv run python test/test_voice_comparison.py --yes

    # List all available voices without playing
    uv run python test/test_voice_comparison.py --list
"""

import os
import sys
import time
import wave
import json
import tempfile
from pathlib import Path
from typing import Optional
import urllib.request

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Piper Voice Manager
# ============================================================================

class PiperVoiceManager:
    """Manage and compare all Piper voices."""
    
    VOICES_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json"
    BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize voice manager."""
        cache_dir = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        self.model_dir = Path(model_dir or cache_dir) / "voice" / "piper-models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.voices_cache = None
        self.loaded_voices = {}
    
    def fetch_voice_list(self) -> dict:
        """Fetch the complete voice list from Piper."""
        if self.voices_cache:
            return self.voices_cache
        
        cache_file = self.model_dir / "voices.json"
        
        # Use cached list if less than 1 day old
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                with open(cache_file) as f:
                    self.voices_cache = json.load(f)
                return self.voices_cache
        
        print("Fetching voice list from Piper...")
        with urllib.request.urlopen(self.VOICES_JSON_URL) as response:
            self.voices_cache = json.loads(response.read())
        
        # Cache locally
        with open(cache_file, "w") as f:
            json.dump(self.voices_cache, f)
        
        return self.voices_cache
    
    def get_english_voices(self) -> list[dict]:
        """Get all English voices with metadata."""
        voices = self.fetch_voice_list()
        
        english_voices = []
        for key, data in sorted(voices.items()):
            if not key.startswith("en_"):
                continue
            
            # Parse key: en_US-lessac-medium
            parts = key.split("-")
            locale = parts[0]  # en_US or en_GB
            quality = parts[-1]  # low, medium, high
            voice_name = "-".join(parts[1:-1])  # voice name (may have hyphens)
            
            # Get model size
            files = data.get("files", {})
            model_size_bytes = 0
            model_file = None
            config_file = None
            for filename, file_info in files.items():
                if filename.endswith(".onnx") and not filename.endswith(".json"):
                    model_size_bytes = file_info.get("size_bytes", 0)
                    model_file = filename
                elif filename.endswith(".onnx.json"):
                    config_file = filename
            
            english_voices.append({
                "key": key,
                "locale": locale,
                "accent": "US" if locale == "en_US" else "GB",
                "voice": voice_name,
                "quality": quality,
                "size_mb": model_size_bytes // (1024 * 1024),
                "model_file": model_file,
                "config_file": config_file,
            })
        
        return english_voices
    
    def _get_model_url(self, voice_info: dict) -> tuple[str, str]:
        """Get download URLs for a voice."""
        locale = voice_info["locale"]
        voice = voice_info["voice"]
        quality = voice_info["quality"]
        key = voice_info["key"]
        
        # URL structure: /en/en_US/lessac/medium/en_US-lessac-medium.onnx
        model_url = f"{self.BASE_URL}/en/{locale}/{voice}/{quality}/{key}.onnx"
        config_url = f"{model_url}.json"
        return model_url, config_url
    
    def _get_model_path(self, voice_info: dict) -> tuple[Path, Path]:
        """Get local paths for a voice model."""
        key = voice_info["key"]
        model_path = self.model_dir / f"{key}.onnx"
        config_path = self.model_dir / f"{key}.onnx.json"
        return model_path, config_path
    
    def download_voice(self, voice_info: dict) -> tuple[Path, Path]:
        """Download a voice model if not cached."""
        model_path, config_path = self._get_model_path(voice_info)
        model_url, config_url = self._get_model_url(voice_info)
        
        if not model_path.exists():
            print(f"    Downloading {voice_info['key']} ({voice_info['size_mb']}MB)...")
            start = time.time()
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"    Downloaded in {time.time() - start:.1f}s")
            except Exception as e:
                print(f"    ERROR downloading: {e}")
                raise
        
        if not config_path.exists():
            urllib.request.urlretrieve(config_url, config_path)
        
        return model_path, config_path
    
    def load_voice(self, voice_info: dict):
        """Load a Piper voice model."""
        key = voice_info["key"]
        if key in self.loaded_voices:
            return self.loaded_voices[key]
        
        from piper import PiperVoice
        
        model_path, config_path = self.download_voice(voice_info)
        
        piper_voice = PiperVoice.load(
            model_path=str(model_path),
            config_path=str(config_path),
            use_cuda=False
        )
        
        self.loaded_voices[key] = piper_voice
        return piper_voice
    
    def synthesize(self, voice_info: dict, text: str) -> tuple[bytes, int]:
        """Synthesize text with a specific voice."""
        piper_voice = self.load_voice(voice_info)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with wave.open(tmp_path, "wb") as wav_file:
                piper_voice.synthesize_wav(text, wav_file)
            
            with wave.open(tmp_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                audio_bytes = wav_file.readframes(wav_file.getnframes())
            
            return audio_bytes, sample_rate
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def play_audio(self, audio_bytes: bytes, sample_rate: int) -> None:
        """Play audio bytes."""
        import sounddevice as sd
        import numpy as np
        
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        
        sd.play(audio, samplerate=sample_rate, blocking=True)


# ============================================================================
# Voice Comparison Functions
# ============================================================================

def list_voices(manager: PiperVoiceManager, accent: Optional[str] = None, 
                quality: Optional[str] = None, voice_filter: Optional[str] = None):
    """List all available voices."""
    voices = manager.get_english_voices()
    
    # Apply filters
    if accent:
        voices = [v for v in voices if v["accent"] == accent]
    if quality:
        voices = [v for v in voices if v["quality"] == quality]
    if voice_filter:
        voices = [v for v in voices if voice_filter.lower() in v["voice"].lower()]
    
    print("\n" + "="*70)
    print("AVAILABLE PIPER ENGLISH VOICES")
    print("="*70)
    
    # Group by accent
    us_voices = [v for v in voices if v["accent"] == "US"]
    gb_voices = [v for v in voices if v["accent"] == "GB"]
    
    total_size = sum(v["size_mb"] for v in voices)
    
    if us_voices:
        print(f"\n🇺🇸 American English ({len(us_voices)} variants):")
        print("-"*50)
        current_voice = None
        for v in us_voices:
            if v["voice"] != current_voice:
                current_voice = v["voice"]
                # Get all qualities for this voice
                qualities = [x["quality"] for x in us_voices if x["voice"] == current_voice]
                print(f"  {v['voice']:25} [{', '.join(qualities)}]")
    
    if gb_voices:
        print(f"\n🇬🇧 British English ({len(gb_voices)} variants):")
        print("-"*50)
        current_voice = None
        for v in gb_voices:
            if v["voice"] != current_voice:
                current_voice = v["voice"]
                qualities = [x["quality"] for x in gb_voices if x["voice"] == current_voice]
                print(f"  {v['voice']:25} [{', '.join(qualities)}]")
    
    print(f"\nTotal: {len(voices)} voice variants")
    print(f"Total download size: ~{total_size}MB ({total_size/1024:.1f}GB)")
    print("="*70)
    
    return voices


def run_comparison(manager: PiperVoiceManager, voices: list[dict], 
                   pause_between: float = 0.5):
    """Run voice comparison demo."""
    
    total_size = sum(v["size_mb"] for v in voices)
    
    print("\n" + "="*70)
    print("PIPER VOICE COMPARISON")
    print("="*70)
    print(f"Playing {len(voices)} voice variants")
    print(f"Total download size (if not cached): ~{total_size}MB")
    print("="*70 + "\n")
    
    # Group voices by accent for organized playback
    us_voices = [v for v in voices if v["accent"] == "US"]
    gb_voices = [v for v in voices if v["accent"] == "GB"]
    
    played = 0
    
    for accent_name, accent_voices in [("🇺🇸 AMERICAN ENGLISH", us_voices), 
                                        ("🇬🇧 BRITISH ENGLISH", gb_voices)]:
        if not accent_voices:
            continue
        
        print("\n" + "-"*70)
        print(f"{accent_name}")
        print("-"*70 + "\n")
        
        for voice_info in accent_voices:
            voice = voice_info["voice"]
            quality = voice_info["quality"]
            accent = "American" if voice_info["accent"] == "US" else "British"
            
            # Create announcement text
            text = f"Hello, I am {voice.replace('_', ' ')} in {quality} quality, with a {accent} accent."
            
            print(f"▶ {voice.upper()} ({quality})")
            print(f"  \"{text}\"")
            
            try:
                start = time.time()
                audio_bytes, sample_rate = manager.synthesize(voice_info, text)
                synth_time = time.time() - start
                
                audio_duration = len(audio_bytes) / (sample_rate * 2)
                print(f"  Synthesized in {synth_time:.2f}s, playing {audio_duration:.1f}s...")
                
                manager.play_audio(audio_bytes, sample_rate)
                played += 1
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            time.sleep(pause_between)
            print()
    
    print("="*70)
    print(f"COMPARISON COMPLETE - Played {played}/{len(voices)} voices")
    print("="*70)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare all Piper English TTS voices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list              List all voices without playing
  %(prog)s --us-only           Play only American voices
  %(prog)s --gb-only           Play only British voices
  %(prog)s --quality medium    Play only medium quality
  %(prog)s --voice lessac      Play only voices matching 'lessac'
  %(prog)s --yes               Skip download confirmation
        """
    )
    
    parser.add_argument("--list", action="store_true",
                        help="List available voices without playing")
    parser.add_argument("--us-only", action="store_true",
                        help="Only include US (American) voices")
    parser.add_argument("--gb-only", action="store_true",
                        help="Only include GB (British) voices")
    parser.add_argument("--quality", choices=["low", "medium", "high"],
                        help="Filter by quality level")
    parser.add_argument("--voice", type=str,
                        help="Filter by voice name (partial match)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip download confirmation")
    parser.add_argument("--pause", type=float, default=0.5,
                        help="Pause between voices in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    # Determine accent filter
    accent = None
    if args.us_only:
        accent = "US"
    elif args.gb_only:
        accent = "GB"
    
    # Initialize manager
    manager = PiperVoiceManager()
    
    # Get filtered voice list
    voices = manager.get_english_voices()
    
    if accent:
        voices = [v for v in voices if v["accent"] == accent]
    if args.quality:
        voices = [v for v in voices if v["quality"] == args.quality]
    if args.voice:
        voices = [v for v in voices if args.voice.lower() in v["voice"].lower()]
    
    if not voices:
        print("No voices match the specified filters.")
        return 1
    
    # List mode
    if args.list:
        list_voices(manager, accent, args.quality, args.voice)
        return 0
    
    # Show what will be played
    total_size = sum(v["size_mb"] for v in voices)
    print(f"\nWill play {len(voices)} voice variants")
    print(f"Estimated download size (if not cached): ~{total_size}MB")
    
    if not args.yes:
        print("\nVoices to play:")
        for v in voices:
            cached = "✓" if (manager.model_dir / f"{v['key']}.onnx").exists() else " "
            print(f"  [{cached}] {v['key']} ({v['size_mb']}MB)")
        
        response = input("\nProceed? [Y/n] ").strip().lower()
        if response and response != "y":
            print("Cancelled.")
            return 0
    
    # Run comparison
    run_comparison(manager, voices, pause_between=args.pause)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
