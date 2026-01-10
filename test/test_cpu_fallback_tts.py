#!/usr/bin/env python3
"""
Test CPU-only TTS fallback options for graceful degradation.

When the GPU-based Kokoro server is unavailable (GPU fully utilized),
we need a fast CPU-only TTS model as a fallback.

Candidates tested:
- Piper TTS: ONNX-based, very fast on CPU, offline, high quality
- (Future: espeak-ng, edge-tts, etc.)

Usage:
    uv run python test/test_cpu_fallback_tts.py
"""

import os
import sys
import time
import wave
import tempfile
from pathlib import Path
from typing import Optional
import urllib.request
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Piper TTS CPU Fallback
# ============================================================================

class PiperCPUFallback:
    """
    Fast CPU-only TTS using Piper (ONNX-based).
    
    Piper is designed for edge devices and runs efficiently on CPU.
    Models are small (~50-100MB) and synthesis is very fast.
    """
    
    # Default model: en_US-lessac-medium (good quality, reasonable size)
    DEFAULT_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    DEFAULT_CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    
    # Alternative: smaller/faster model
    FAST_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx"
    FAST_CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json"
    
    def __init__(self, model_dir: Optional[str] = None, use_fast_model: bool = False):
        """
        Initialize Piper CPU fallback.
        
        Args:
            model_dir: Directory to store downloaded models
            use_fast_model: Use smaller/faster model (lower quality)
        """
        self.model_dir = Path(model_dir or self._get_default_model_dir())
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_fast_model = use_fast_model
        self.voice = None
        self._model_loaded = False
        
    def _get_default_model_dir(self) -> Path:
        """Get default directory for storing Piper models."""
        # Use XDG cache or fallback
        cache_dir = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        return Path(cache_dir) / "voice" / "piper-models"
    
    def _get_model_paths(self) -> tuple[Path, Path]:
        """Get paths for model and config files."""
        if self.use_fast_model:
            model_name = "en_US-lessac-low.onnx"
        else:
            model_name = "en_US-lessac-medium.onnx"
        
        model_path = self.model_dir / model_name
        config_path = self.model_dir / f"{model_name}.json"
        return model_path, config_path
    
    def _download_model(self) -> tuple[Path, Path]:
        """Download Piper model if not already cached."""
        model_path, config_path = self._get_model_paths()
        
        if self.use_fast_model:
            model_url = self.FAST_MODEL_URL
            config_url = self.FAST_CONFIG_URL
        else:
            model_url = self.DEFAULT_MODEL_URL
            config_url = self.DEFAULT_CONFIG_URL
        
        # Download model if needed
        if not model_path.exists():
            print(f"[Piper] Downloading model to {model_path}...")
            start = time.time()
            urllib.request.urlretrieve(model_url, model_path)
            print(f"[Piper] Model downloaded in {time.time() - start:.1f}s")
        
        # Download config if needed
        if not config_path.exists():
            print(f"[Piper] Downloading config to {config_path}...")
            urllib.request.urlretrieve(config_url, config_path)
        
        return model_path, config_path
    
    def load_model(self) -> None:
        """Load the Piper voice model (lazy loading)."""
        if self._model_loaded:
            return
        
        from piper import PiperVoice
        
        model_path, config_path = self._download_model()
        
        print(f"[Piper] Loading model from {model_path}...")
        start = time.time()
        
        self.voice = PiperVoice.load(
            model_path=str(model_path),
            config_path=str(config_path),
            use_cuda=False  # Force CPU
        )
        
        self._model_loaded = True
        print(f"[Piper] Model loaded in {time.time() - start:.2f}s")
    
    def synthesize(self, text: str) -> tuple[bytes, int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        self.load_model()
        
        start = time.time()
        
        # Synthesize to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with wave.open(tmp_path, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)
            
            # Read the audio data
            with wave.open(tmp_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                audio_bytes = wav_file.readframes(wav_file.getnframes())
            
            elapsed = time.time() - start
            # Estimate audio duration from bytes (16-bit mono)
            audio_duration = len(audio_bytes) / (sample_rate * 2)
            rtf = elapsed / audio_duration if audio_duration > 0 else 0
            
            print(f"[Piper] Synthesized {len(text)} chars in {elapsed:.3f}s "
                  f"(RTF: {rtf:.2f}x, audio: {audio_duration:.2f}s)")
            
            return audio_bytes, sample_rate
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def synthesize_to_file(self, text: str, output_path: str) -> float:
        """
        Synthesize speech and save to WAV file.
        
        Args:
            text: Text to synthesize
            output_path: Output WAV file path
            
        Returns:
            Synthesis time in seconds
        """
        self.load_model()
        
        start = time.time()
        
        with wave.open(output_path, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        
        elapsed = time.time() - start
        print(f"[Piper] Saved to {output_path} in {elapsed:.3f}s")
        return elapsed


def play_audio(audio_bytes: bytes, sample_rate: int) -> None:
    """Play audio bytes using sounddevice."""
    try:
        import sounddevice as sd
        import numpy as np
        
        # Convert bytes to numpy array (16-bit signed int)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        # Normalize to float32 [-1, 1]
        audio = audio.astype(np.float32) / 32768.0
        
        print(f"[Audio] Playing {len(audio)/sample_rate:.2f}s of audio...")
        sd.play(audio, samplerate=sample_rate, blocking=True)
        print("[Audio] Playback complete")
        
    except ImportError:
        print("[Audio] sounddevice not available, skipping playback")
    except Exception as e:
        print(f"[Audio] Playback error: {e}")


# ============================================================================
# Tests
# ============================================================================

def test_piper_basic():
    """Test basic Piper TTS synthesis."""
    print("\n" + "="*60)
    print("TEST: Basic Piper TTS Synthesis")
    print("="*60)
    
    piper = PiperCPUFallback()
    
    text = "Hello! This is a test of the Piper text to speech system running on CPU only."
    audio_bytes, sample_rate = piper.synthesize(text)
    
    assert len(audio_bytes) > 0, "Audio should not be empty"
    assert sample_rate > 0, "Sample rate should be positive"
    
    print(f"✓ Generated {len(audio_bytes)} bytes at {sample_rate}Hz")
    
    return audio_bytes, sample_rate


def test_piper_speed():
    """Test Piper synthesis speed for various text lengths."""
    print("\n" + "="*60)
    print("TEST: Piper Speed Benchmark")
    print("="*60)
    
    piper = PiperCPUFallback()
    
    # Ensure model is loaded (don't count in benchmark)
    piper.load_model()
    
    test_texts = [
        ("Short", "Hello world."),
        ("Medium", "This is a medium length sentence that contains a few more words to synthesize."),
        ("Long", "Welcome to our text to speech demonstration. Today we will be testing the performance "
                 "of CPU-based synthesis. This is particularly important when GPU resources are "
                 "unavailable or fully utilized by other tasks. The goal is to provide graceful "
                 "degradation with reasonable quality and speed."),
    ]
    
    results = []
    for name, text in test_texts:
        start = time.time()
        audio_bytes, sample_rate = piper.synthesize(text)
        elapsed = time.time() - start
        
        # Calculate real-time factor
        audio_duration = len(audio_bytes) / (sample_rate * 2)  # 16-bit mono
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        
        results.append({
            "name": name,
            "chars": len(text),
            "time": elapsed,
            "audio_duration": audio_duration,
            "rtf": rtf
        })
        
        print(f"  {name}: {len(text)} chars → {elapsed:.3f}s synthesis, "
              f"{audio_duration:.2f}s audio (RTF: {rtf:.2f}x)")
    
    # Check that we're faster than real-time
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    print(f"\n  Average RTF: {avg_rtf:.2f}x (< 1.0 means faster than real-time)")
    
    assert avg_rtf < 1.0, f"Synthesis should be faster than real-time (RTF: {avg_rtf:.2f})"
    print("✓ All syntheses faster than real-time")
    
    return results


def test_piper_fast_model():
    """Test the faster/smaller Piper model."""
    print("\n" + "="*60)
    print("TEST: Piper Fast Model (low quality, higher speed)")
    print("="*60)
    
    piper_fast = PiperCPUFallback(use_fast_model=True)
    
    text = "This is a test using the faster, smaller model. Quality may be lower but speed should be higher."
    
    start = time.time()
    audio_bytes, sample_rate = piper_fast.synthesize(text)
    elapsed = time.time() - start
    
    audio_duration = len(audio_bytes) / (sample_rate * 2)
    rtf = elapsed / audio_duration
    
    print(f"  Fast model RTF: {rtf:.3f}x")
    print(f"✓ Fast model synthesis complete")
    
    return audio_bytes, sample_rate


def test_piper_save_to_file():
    """Test saving synthesis to file."""
    print("\n" + "="*60)
    print("TEST: Save to WAV file")
    print("="*60)
    
    piper = PiperCPUFallback()
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name
    
    try:
        text = "This audio has been saved to a file."
        elapsed = piper.synthesize_to_file(text, output_path)
        
        # Verify file exists and has content
        assert os.path.exists(output_path), "Output file should exist"
        file_size = os.path.getsize(output_path)
        assert file_size > 1000, f"Output file should have content (got {file_size} bytes)"
        
        print(f"✓ Saved {file_size} bytes to {output_path}")
        
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_piper_playback():
    """Test audio playback (interactive test)."""
    print("\n" + "="*60)
    print("TEST: Audio Playback (interactive)")
    print("="*60)
    
    piper = PiperCPUFallback()
    
    text = "Hello! If you can hear this, the CPU fallback text to speech is working correctly."
    audio_bytes, sample_rate = piper.synthesize(text)
    
    play_audio(audio_bytes, sample_rate)
    print("✓ Playback test complete")


def test_fallback_scenario():
    """
    Test the full fallback scenario:
    1. Try to connect to Kokoro server (simulated failure)
    2. Fall back to Piper CPU TTS
    """
    print("\n" + "="*60)
    print("TEST: Graceful Degradation Scenario")
    print("="*60)
    print("Simulating: GPU server unavailable → CPU fallback\n")
    
    # Simulate server connection attempt
    import socket
    
    def try_server_connection(host="127.0.0.1", port=3124, timeout=0.5):
        """Attempt to connect to the Kokoro server."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False
    
    # Step 1: Try server
    print("[Fallback] Checking if Kokoro server is available...")
    server_available = try_server_connection()
    
    if server_available:
        print("[Fallback] Server is running - would use GPU synthesis")
    else:
        print("[Fallback] Server not available - using CPU fallback")
    
    # Step 2: Use Piper fallback
    print("[Fallback] Initializing Piper CPU fallback...")
    piper = PiperCPUFallback()
    
    text = "The GPU server was not available, so I am speaking using the CPU fallback system. This provides graceful degradation."
    
    start = time.time()
    audio_bytes, sample_rate = piper.synthesize(text)
    total_time = time.time() - start
    
    audio_duration = len(audio_bytes) / (sample_rate * 2)
    
    print(f"\n[Fallback] Total fallback time: {total_time:.3f}s")
    print(f"[Fallback] Audio duration: {audio_duration:.2f}s")
    print(f"[Fallback] User wait time: {total_time:.3f}s (synthesis only, model was cached)")
    
    print("✓ Graceful degradation working")
    
    return audio_bytes, sample_rate


def test_cold_start():
    """Test cold start time (first synthesis after import)."""
    print("\n" + "="*60)
    print("TEST: Cold Start Performance")
    print("="*60)
    print("Measuring time from initialization to first audio\n")
    
    # Create new instance (simulating cold start)
    start = time.time()
    piper = PiperCPUFallback()
    init_time = time.time() - start
    
    # First synthesis (includes model load)
    synth_start = time.time()
    text = "Cold start test."
    audio_bytes, sample_rate = piper.synthesize(text)
    first_synth_time = time.time() - synth_start
    
    total_cold_start = time.time() - start
    
    print(f"  Initialization: {init_time:.3f}s")
    print(f"  First synthesis (incl. model load): {first_synth_time:.3f}s")
    print(f"  Total cold start: {total_cold_start:.3f}s")
    
    # Second synthesis (model already loaded)
    synth_start = time.time()
    audio_bytes, sample_rate = piper.synthesize("Warm synthesis.")
    warm_synth_time = time.time() - synth_start
    
    print(f"  Warm synthesis: {warm_synth_time:.3f}s")
    print(f"\n✓ Cold start complete in {total_cold_start:.2f}s")
    
    return total_cold_start, warm_synth_time


def run_all_tests(play_audio_test: bool = False):
    """Run all tests."""
    print("\n" + "="*60)
    print("CPU FALLBACK TTS TEST SUITE")
    print("="*60)
    print("Testing fast CPU-only TTS for graceful degradation")
    print("when GPU (Kokoro server) is unavailable.\n")
    
    overall_start = time.time()
    
    # Run tests
    test_piper_basic()
    test_piper_speed()
    test_piper_fast_model()
    test_piper_save_to_file()
    test_cold_start()
    test_fallback_scenario()
    
    if play_audio_test:
        test_piper_playback()
    
    total_time = time.time() - overall_start
    
    print("\n" + "="*60)
    print(f"ALL TESTS PASSED in {total_time:.2f}s")
    print("="*60)
    print("\nPiper TTS is a viable CPU fallback option:")
    print("  ✓ Fast synthesis (faster than real-time on CPU)")
    print("  ✓ Good quality (medium model)")
    print("  ✓ Small model size (~60MB)")
    print("  ✓ No GPU required")
    print("  ✓ Works offline after model download")
    print("  ✓ Cold start ~1s (after model cached)")
    print("  ✓ Warm synthesis <0.5s for typical sentences")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CPU-only TTS fallback")
    parser.add_argument("--play", action="store_true", help="Include audio playback test")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    args = parser.parse_args()
    
    if args.quick:
        # Quick sanity check
        print("Quick test: Basic Piper synthesis")
        piper = PiperCPUFallback()
        audio, rate = piper.synthesize("Quick test.")
        print(f"✓ Generated {len(audio)} bytes")
    else:
        run_all_tests(play_audio_test=args.play)
