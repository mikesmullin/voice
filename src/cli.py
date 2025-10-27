"""Command-line interface for the voice TTS system."""

import sys
import argparse
import warnings
import os
from typing import Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

from .voice_engine import VoiceEngine
from .timing import start_timer


def read_stdin() -> Optional[str]:
    """
    Read text from STDIN if available.

    Returns:
        Text from STDIN or None if not available
    """
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None


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
  voice heart "Hello, test subject."
  voice adam What is in the image?
  voice bella Where be the treasure?
  voice -o output.wav heart "Save to file"
  echo "Hello from STDIN" | voice heart
  echo "partial text" | voice heart and some more text
  echo "only STDIN" | voice heart -
  voice --list
  voice --info heart
  voice --config config.local.yaml heart "Custom config"
        """,
    )

    parser.add_argument(
        "preset", nargs="?", help="Voice preset name (e.g., glados, pirate, neutral)"
    )

    parser.add_argument(
        "text",
        nargs="*",
        help="Text to synthesize (multiple arguments will be joined with spaces, use '-' to read only from STDIN)",
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Save audio to file instead of playing (e.g., output.wav)",
    )

    parser.add_argument(
        "-c", "--config", metavar="FILE", help="Path to custom config.yaml file"
    )

    parser.add_argument(
        "-l", "--list", action="store_true", help="List available voice presets"
    )

    parser.add_argument(
        "-i", "--info", metavar="PRESET", help="Show information about a voice preset"
    )

    parser.add_argument("-v", "--version", action="version", version="voice 0.1.0")

    return parser.parse_args(args)


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: List of arguments (for testing). If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Start timing
    start_timer()
    
    try:
        # Check if no arguments provided, show help
        if args is None and len(sys.argv) == 1:
            parse_args(["--help"])
            return 0

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

                if info.get("voice"):
                    print(f"  Voice Variant: {info.get('voice')}")

                if info.get("enable_llm") and info.get("system_prompt"):
                    prompt = info["system_prompt"].strip()
                    if len(prompt) > 100:
                        prompt = prompt[:100] + "..."
                    print(f"  System Prompt: {prompt}")

                return 0
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

        # Process text input from args and STDIN
        text_parts = []
        stdin_only = False

        # Check if text is provided as '-' (STDIN only mode)
        if (
            parsed_args.text
            and len(parsed_args.text) == 1
            and parsed_args.text[0] == "-"
        ):
            stdin_only = True
        elif parsed_args.text:
            # Join all text arguments with spaces
            text_parts.append(" ".join(parsed_args.text))

        # Read from STDIN if available
        stdin_text = read_stdin()
        if stdin_text:
            text_parts.append(stdin_text)
        elif stdin_only:
            print(
                "Error: STDIN input required when using '-' for text", file=sys.stderr
            )
            return 1

        # Combine all text parts
        final_text = " ".join(text_parts).strip() if text_parts else None

        # Require preset and text for synthesis
        if not parsed_args.preset or not final_text:
            print(
                "Error: Both <preset> and <text> are required for synthesis",
                file=sys.stderr,
            )
            print("Use 'voice --help' for usage information", file=sys.stderr)
            print("\nQuick start:")
            print("  voice --list                    # List available voices")
            print('  voice glados "Hello, world!"    # Synthesize speech')
            return 1

        # Synthesize speech
        engine.synthesize(
            text=final_text,
            voice_name=parsed_args.preset,
            output_file=parsed_args.output,
        )

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
