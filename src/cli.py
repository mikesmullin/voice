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
from . import timing


TOP_LEVEL_HELP = """voice

Usage:
    voice [options] <preset> <text>
    voice [options] <preset> -
    voice <command> [options]

DIRECT synthesis:
    <preset> <text>       Speak the provided text immediately
    <preset> -            Read the spoken text from STDIN

SERVER commands:
    serve                 Start a warm synthesis server
    hot                   Send a synthesis request to the running server

Options:
    -o, --output FILE     Save audio to WAV instead of playing it
    -c, --config FILE     Use a custom config.yaml
    -l, --list            List available voice presets
    -i, --info PRESET     Show preset details
    --cpu                 Force CPU usage instead of GPU
    --stinger NAME        Play a configured stinger before speech
    --gain VALUE          Apply linear gain to synthesized voice audio
    -v, --version         Show version
    -h, --help            Show this help

Examples:
    voice lessac "Hello from Piper."
    voice --gain=1.2 alan "Louder playback"
    printf '%s' "From stdin" | voice --config ./src/config.yaml cori -
    voice --list
    voice --info alan
    voice serve
    voice hot bella "Server request" --gain=0.8

Use "voice <command> --help" for command-specific help.
"""


def print_top_level_help() -> None:
    """Print custom top-level CLI help."""
    print(TOP_LEVEL_HELP.rstrip())


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
    # Get args from sys.argv if not provided
    if args is None:
        args = sys.argv[1:]

    # Check if first argument is a known subcommand
    # If not, treat as default synthesis mode
    known_subcommands = ['serve', 'hot']
    is_subcommand = len(args) > 0 and args[0] in known_subcommands
    
    if is_subcommand:
        # Parse with subcommands
        parser = argparse.ArgumentParser(
            prog="voice",
            description="Text-to-speech with voice presets",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        
        parser.add_argument("-v", "--version", action="version", version="voice 0.1.0")
        
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Serve subcommand
        serve_parser = subparsers.add_parser(
            "serve",
            help="Start voice synthesis server for low-latency hot reloading"
        )
        serve_parser.add_argument(
            "-c", "--config", metavar="FILE", help="Path to custom config.yaml file"
        )
        serve_parser.add_argument(
            "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
        )
        serve_parser.add_argument(
            "--port", type=int, default=3124, help="Port to bind to (default: 3124)"
        )
        serve_parser.add_argument(
            "--cpu", action="store_true", help="Force CPU usage instead of GPU"
        )
        
        # Hot subcommand
        hot_parser = subparsers.add_parser(
            "hot",
            help="Send synthesis request to running server (low-latency)"
        )
        hot_parser.add_argument(
            "preset", help="Voice preset name"
        )
        hot_parser.add_argument(
            "text",
            nargs="+",
            help="Text to synthesize"
        )
        hot_parser.add_argument(
            "-o", "--output", metavar="FILE", help="Save audio to file"
        )
        hot_parser.add_argument(
            "--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)"
        )
        hot_parser.add_argument(
            "--port", type=int, default=3124, help="Server port (default: 3124)"
        )
        hot_parser.add_argument(
            "--stinger", metavar="NAME", help="Stinger sound effect to play before speech (e.g., alert, error)"
        )
        hot_parser.add_argument(
            "--gain", type=float, default=1.0, metavar="VALUE", help="Adjust synthesized voice volume with a linear gain multiplier"
        )
        
        return parser.parse_args(args)
    else:
        # Parse as default synthesis command
        parser = argparse.ArgumentParser(
            prog="voice",
            description="Text-to-speech with voice presets",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False,
        )
        parser.add_argument(
            "-h", "--help", action="store_true", help="Show help"
        )
        parser.add_argument(
            "preset", nargs="?", help="Voice preset name (e.g., heart, bella, adam)"
        )
        parser.add_argument(
            "text",
            nargs="*",
            help="Text to synthesize; multiple arguments are joined with spaces, or use '-' to read from STDIN",
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
        parser.add_argument(
            "--cpu", action="store_true", help="Force CPU usage instead of GPU"
        )
        parser.add_argument(
            "--stinger", metavar="NAME", help="Stinger sound effect to play before speech (e.g., alert, error)"
        )
        parser.add_argument(
            "--gain", type=float, default=1.0, metavar="VALUE", help="Adjust synthesized voice volume with a linear gain multiplier (1.0 = unchanged)"
        )
        
        parsed = parser.parse_args(args)
        parsed.command = None  # Mark as default command
        return parsed


def validate_gain(gain: float) -> float:
    """Validate the CLI gain parameter."""
    if gain < 0:
        raise ValueError("--gain must be greater than or equal to 0")
    return gain


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: List of arguments (for testing). If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Show custom top-level help for the default entrypoint.
        if args is None and len(sys.argv) == 1:
            print_top_level_help()
            return 0

        parsed_args = parse_args(args)

        if getattr(parsed_args, 'command', None) is None and getattr(parsed_args, 'help', False):
            print_top_level_help()
            return 0

        gain = validate_gain(getattr(parsed_args, 'gain', 1.0))
        
        # Handle serve subcommand
        if parsed_args.command == "serve":
            from .server import start_server
            start_server(
                config_path=parsed_args.config,
                host=parsed_args.host,
                port=parsed_args.port,
                force_cpu=parsed_args.cpu
            )
            return 0
        
        # Handle hot subcommand
        if parsed_args.command == "hot":
            from .client import send_synthesis_request
            
            text = " ".join(parsed_args.text) if parsed_args.text else ""
            
            response = send_synthesis_request(
                voice_name=parsed_args.preset,
                text=text,
                output_file=getattr(parsed_args, 'output', None),
                host=parsed_args.host,
                port=parsed_args.port,
                connection_timeout=0.5,
                stinger=getattr(parsed_args, 'stinger', None),
                gain=gain,
            )
            
            # If server connection failed (None), fall back to Piper CPU synthesis
            if response is None:
                timing.start_timer()
                timing.log("[Voice] Server not available, falling back to Piper CPU synthesis...")
                
                engine = VoiceEngine(config_path=None, force_cpu=True)
                
                if not parsed_args.preset or not text:
                    print("Error: Missing preset or text", file=sys.stderr)
                    return 1
                
                # Get the fallback voice from config
                fallback_voice = engine.get_fallback_voice()
                timing.log(f"[Voice] Using fallback voice: {fallback_voice}")
                
                try:
                    engine.synthesize(
                        text=text,
                        voice_name=fallback_voice,
                        output_file=getattr(parsed_args, 'output', None),
                        stinger=getattr(parsed_args, 'stinger', None),
                        gain=gain,
                    )
                    return 0
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    return 1
            
            # Server responded with error
            if "error" in response:
                # Check if it was an interrupt
                if response['error'] == "Interrupted by user":
                    raise KeyboardInterrupt()
                print(f"Error: {response['error']}", file=sys.stderr)
                return 1
            
            # Success
            return 0

        # Start timing for normal synthesis
        timing.start_timer()
        
        # Create voice engine with custom config if specified
        engine = VoiceEngine(config_path=parsed_args.config, force_cpu=getattr(parsed_args, 'cpu', False))

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

        # Process text input from args and optional STDIN.
        text_parts = []
        stdin_only = False

        # A lone '-' means the text should be read from STDIN.
        if (
            parsed_args.text
            and len(parsed_args.text) == 1
            and parsed_args.text[0] == "-"
        ):
            stdin_only = True
        elif parsed_args.text:
            # Join all text arguments with spaces
            text_parts.append(" ".join(parsed_args.text))

        # Read from STDIN only for the explicit '-' convention.
        if stdin_only:
            stdin_text = read_stdin()
            if stdin_text:
                text_parts.append(stdin_text)
            else:
                print(
                    "Error: STDIN input required when using '-' for text",
                    file=sys.stderr,
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
            print('  voice lessac "Hello, world!"    # Synthesize speech')
            print('  printf "%s" "Hello" | voice cori -  # Read spoken text from STDIN')
            return 1

        # Synthesize speech
        engine.synthesize(
            text=final_text,
            voice_name=parsed_args.preset,
            output_file=parsed_args.output,
            stinger=getattr(parsed_args, 'stinger', None),
            gain=gain,
        )

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
