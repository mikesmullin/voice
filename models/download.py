"""Download all required TTS model files for the voice project.

This script will:
1. Clean the models/TTS/ directory
2. Download all required ONNX models and supporting files
3. Verify checksums to ensure file integrity
"""

import asyncio
import shutil
from hashlib import sha256
from pathlib import Path

import httpx
from rich import print as rprint
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn


# Get the models directory (parent of this script)
MODELS_DIR = Path(__file__).parent
TTS_DIR = MODELS_DIR / "TTS"

# Model file details with download URLs and checksums
# Source: https://github.com/dnhkng/GLaDOS
# Note: Only the main ONNX models are available in GLaDOS releases.
# The pkl and json files must be obtained from a backup or Piper TTS project.
MODEL_DETAILS = {
    "TTS/glados.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/glados.onnx",
        "checksum": "17ea16dd18e1bac343090b8589042b4052f1e5456d42cad8842a4f110de25095",
    },
    "TTS/kokoro-v1.0.fp16.onnx": {
        "url": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-v1.0.fp16.onnx",
        "checksum": "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204",
    },
    "TTS/kokoro-voices-v1.0.bin": {
        "url": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-voices-v1.0.bin",
        "checksum": "c5adf5cc911e03b76fa5025c1c225b141310d0c4a721d6ed6e96e73309d0fd88",
    },
    "TTS/phomenizer_en.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/phomenizer_en.onnx",
        "checksum": "b64dbbeca8b350927a0b6ca5c4642e0230173034abd0b5bb72c07680d700c5a0",
    },
}

# Additional required files (not available for download, must be from backup):
# - TTS/glados.json (GLaDOS model config)
# - TTS/phoneme_to_id.pkl (GLaDOS phoneme mapping)
# - TTS/lang_phoneme_dict.pkl (Phonemizer dictionary)
# - TTS/token_to_idx.pkl (Phonemizer token mapping)
# - TTS/idx_to_token.pkl (Phonemizer reverse token mapping)


async def download_with_progress(
    client: httpx.AsyncClient,
    url: str,
    file_path: Path,
    expected_checksum: str,
    progress: Progress,
) -> bool:
    """Download a single file with progress tracking and SHA-256 checksum verification.

    Args:
        client: Async HTTP client
        url: URL to download from
        file_path: Path where file should be saved
        expected_checksum: Expected SHA-256 checksum for verification
        progress: Rich progress bar instance

    Returns:
        bool: True if download and verification succeeded, False otherwise
    """
    task_id = progress.add_task(f"Downloading {file_path.name}", status="")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    hash_sha256 = sha256()

    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            # Set total size for progress bar
            total_size = int(response.headers.get("Content-Length", 0))
            if total_size:
                progress.update(task_id, total=total_size)

            with file_path.open(mode="wb") as f:
                async for chunk in response.aiter_bytes(32768):  # 32KB chunks
                    f.write(chunk)
                    # Update the hash as we go along, for speed
                    hash_sha256.update(chunk)
                    progress.update(task_id, advance=len(chunk))

        # Verify checksum
        actual_checksum = hash_sha256.hexdigest()
        
        # Skip checksum verification for files with no checksum
        if expected_checksum is None:
            progress.update(task_id, status="[yellow]✓ Downloaded (no checksum)[/yellow]")
            rprint(f"[dim]Info: {file_path.name} checksum: {actual_checksum}[/dim]")
            return True
        
        if actual_checksum == expected_checksum:
            progress.update(task_id, status="[green]✓ Verified[/green]")
            return True
        else:
            progress.update(task_id, status=f"[red]✗ Checksum failed[/red]")
            rprint(f"[red]Expected: {expected_checksum}[/red]")
            rprint(f"[red]Got:      {actual_checksum}[/red]")
            file_path.unlink()  # Delete corrupted file
            return False

    except httpx.HTTPStatusError as e:
        progress.update(task_id, status=f"[red]✗ HTTP {e.response.status_code}[/red]")
        return False
    except Exception as e:
        progress.update(task_id, status=f"[red]✗ Error: {str(e)}[/red]")
        return False


async def download_models() -> int:
    """Download all required model files.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    rprint("\n[bold cyan]TTS Model Downloader[/bold cyan]")
    rprint(f"Target directory: {TTS_DIR.absolute()}\n")

    with Progress(
        TextColumn("[grey50][progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TextColumn("  {task.fields[status]}"),
    ) as progress:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            # Create a download task for each file
            tasks = [
                asyncio.create_task(
                    download_with_progress(
                        client,
                        model_info["url"],
                        MODELS_DIR / path,
                        model_info["checksum"],
                        progress,
                    )
                )
                for path, model_info in MODEL_DETAILS.items()
            ]
            results: list[bool] = await asyncio.gather(*tasks)

    successful_downloads = sum(results)
    total_downloads = len(results)
    
    if not all(results):
        rprint(f"\n[bold red]⚠ {successful_downloads}/{total_downloads} files downloaded successfully[/bold red]")
        return 1
    
    rprint("\n[bold green]✓ All downloadable files retrieved successfully![/bold green]")
    return 0


def clean_tts_directory():
    """Clean the TTS directory before downloading fresh models.
    
    Note: This only removes files, not directories, to preserve any backup files.
    """
    if TTS_DIR.exists():
        rprint(f"\n[yellow]Cleaning {TTS_DIR}...[/yellow]")
        # Remove only the ONNX and BIN files that we'll be downloading
        files_to_remove = [
            "glados.onnx",
            "kokoro-v1.0.fp16.onnx",
            "kokoro-voices-v1.0.bin",
            "phomenizer_en.onnx",
        ]
        for filename in files_to_remove:
            file_path = TTS_DIR / filename
            if file_path.exists():
                file_path.unlink()
                rprint(f"  Removed: {filename}")
        rprint("[green]Directory cleaned (pkl/json files preserved).[/green]")
    else:
        rprint(f"\n[cyan]Creating {TTS_DIR}...[/cyan]")
        TTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point."""
    rprint("[bold]Voice TTS Models Download Script[/bold]")
    rprint("=" * 60)
    
    rprint("\n[dim]This script downloads the main ONNX model files.[/dim]")
    rprint("[dim]Additional pkl/json files must be restored from backup.[/dim]\n")
    
    # Clean the directory first
    clean_tts_directory()
    
    # Download all models
    exit_code = asyncio.run(download_models())
    
    if exit_code == 0:
        rprint("\n[bold green]✓ Download complete![/bold green]")
        rprint("\n[yellow]⚠ Additional required files (restore from backup):[/yellow]")
        rprint("  • glados.json")
        rprint("  • phoneme_to_id.pkl")
        rprint("  • lang_phoneme_dict.pkl")
        rprint("  • token_to_idx.pkl")
        rprint("  • idx_to_token.pkl")
        rprint(f"\n[dim]Location: {TTS_DIR.absolute()}[/dim]")
        rprint("\n[cyan]After restoring backup files, test with:[/cyan]")
        rprint("  [cyan]voice glados \"Hello, test subject.\"[/cyan]")
    else:
        rprint("\n[bold red]Download failed. Please check the errors above.[/bold red]")
    
    return exit_code


if __name__ == "__main__":
    exit(main())
