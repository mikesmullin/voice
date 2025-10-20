"""Command-line interface for the voice TTS system."""

import sys
import argparse
from typing import Optional

from .voice_engine import VoiceEngine


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: List of arguments (for testing). If None, uses sys.argv.
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog="voice",
        description="Text-to-speech with voice presets and LLM integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  voice glados "Hello, test subject."
  voice pirate "Where be the treasure?"
  voice -o output.ogg glados "Save to file"
  voice --list
  voice --info glados
  voice --config custom.yaml neutral "Custom config"
        """
    )
    
    parser.add_argument(
        "preset",
        nargs="?",
        help="Voice preset name (e.g., glados, pirate, neutral)"
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize"
    )
    
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Save audio to file instead of playing (e.g., output.ogg)"
    )
    
    parser.add_argument(
        "-c", "--config",
        metavar="FILE",
        help="Path to custom config.yaml file"
    )
    
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available voice presets"
    )
    
    parser.add_argument(
        "-i", "--info",
        metavar="PRESET",
        help="Show information about a voice preset"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="voice 0.1.0"
    )
    
    return parser.parse_args(args)


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: List of arguments (for testing). If None, uses sys.argv.
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        parsed_args = parse_args(args)
        
        # Create voice engine with custom config if specified
        engine = VoiceEngine(config_path=parsed_args.config)
        
        # Handle list command
        if parsed_args.list:
            voices = engine.list_voices()
            print("Available voice presets:")
            for voice in voices:
                print(f"  - {voice}")
            return 0
        
        # Handle info command
        if parsed_args.info:
            try:
                info = engine.get_voice_info(parsed_args.info)
                print(f"\nVoice preset: {parsed_args.info}")
                print(f"  TTS Engine: {info.get('tts_engine')}")
                print(f"  Speed: {info.get('speed', 1.0)}")
                print(f"  LLM Enabled: {info.get('enable_llm', False)}")
                
                if info.get('voice'):
                    print(f"  Voice Variant: {info.get('voice')}")
                
                if info.get('enable_llm') and info.get('system_prompt'):
                    prompt = info['system_prompt'].strip()
                    if len(prompt) > 100:
                        prompt = prompt[:100] + "..."
                    print(f"  System Prompt: {prompt}")
                
                return 0
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        
        # Require preset and text for synthesis
        if not parsed_args.preset or not parsed_args.text:
            print("Error: Both <preset> and <text> are required for synthesis", file=sys.stderr)
            print("Use 'voice --help' for usage information", file=sys.stderr)
            print("\nQuick start:")
            print("  voice --list                    # List available voices")
            print("  voice glados \"Hello, world!\"    # Synthesize speech")
            return 1
        
        # Synthesize speech
        engine.synthesize(
            text=parsed_args.text,
            voice_name=parsed_args.preset,
            output_file=parsed_args.output
        )
        
        # Clean up
        engine.cleanup()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
