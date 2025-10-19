"""
Example usage of the voice TTS system.

This script demonstrates how to use the voice system programmatically.
"""

from voice.voice_engine import VoiceEngine


def main():
    """Demonstrate voice TTS functionality."""
    
    # Create voice engine
    print("Initializing voice engine...")
    engine = VoiceEngine()
    
    # List available voices
    print("\n=== Available Voices ===")
    voices = engine.list_voices()
    for voice in voices:
        print(f"  - {voice}")
    
    # Show info for a specific voice
    print("\n=== GLaDOS Voice Info ===")
    glados_info = engine.get_voice_info("glados")
    print(f"Engine: {glados_info.get('tts_engine')}")
    print(f"LLM Enabled: {glados_info.get('enable_llm')}")
    
    # Synthesize some examples
    print("\n=== Synthesizing Speech ===")
    
    examples = [
        ("glados", "Hello, test subject. Your continued survival is remarkable."),
        ("pirate", "Ahoy there! Where be the treasure, matey?"),
        ("neutral", "This is a normal voice without any special personality."),
    ]
    
    for voice, text in examples:
        print(f"\n[{voice}] {text}")
        try:
            # Synthesize and play (or save with output_file parameter)
            engine.synthesize(
                text=text,
                voice_name=voice,
                output_file=None  # Set to "output.ogg" to save instead
            )
        except Exception as e:
            print(f"Error: {e}")
    
    # Save to file example
    print("\n=== Saving to File ===")
    engine.synthesize(
        text="This will be saved to a file.",
        voice_name="glados",
        output_file="example_output.ogg"
    )
    
    # Clean up
    print("\n=== Cleaning Up ===")
    engine.cleanup()
    print("Done!")


if __name__ == "__main__":
    main()
