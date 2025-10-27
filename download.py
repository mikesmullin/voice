#!/usr/bin/env python3
"""
Download required model files for StyleTTS2.

This script downloads the LibriTTS pretrained model from Hugging Face.
Run this after installing the package with: uv tool install --editable .[styletts2]
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve


class DownloadProgressBar:
    """Simple progress indicator for file downloads."""
    
    def __init__(self, filename):
        self.filename = filename
        self.last_percent = -1
    
    def __call__(self, block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(int(downloaded * 100 / total_size), 100)
            
            # Print every 10%
            if percent >= self.last_percent + 10:
                self.last_percent = percent
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"  {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")


def download_file(url, dest_path):
    """Download a file with progress indicator."""
    print(f"Downloading {dest_path.name}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urlretrieve(url, dest_path, DownloadProgressBar(dest_path.name))
        print(f"✓ Downloaded {dest_path.name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {dest_path.name}: {e}")
        return False


def main():
    # Determine the base directory (where this script is located)
    script_dir = Path(__file__).parent.resolve()
    models_dir = script_dir / "tmp" / "StyleTTS2" / "Models" / "LibriTTS"
    plbert_dir = script_dir / "tmp" / "StyleTTS2" / "Utils" / "PLBERT"
    
    print(f"StyleTTS2 model directory: {models_dir}")
    print()
    
    # Check if already downloaded
    config_path = models_dir / "config.yml"
    checkpoint_path = models_dir / "epochs_2nd_00020.pth"
    plbert_config_path = plbert_dir / "config.yml"
    plbert_checkpoint_path = plbert_dir / "step_1000000.t7"
    
    if (config_path.exists() and checkpoint_path.exists() and 
        plbert_config_path.exists() and plbert_checkpoint_path.exists()):
        print("✓ Model files already exist!")
        print(f"  - {config_path}")
        print(f"  - {checkpoint_path}")
        print(f"  - {plbert_config_path}")
        print(f"  - {plbert_checkpoint_path}")
        print()
        print("If you want to re-download, delete these files and run this script again.")
        return 0
    
    # Hugging Face URLs
    base_url = "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main"
    files = {
        "config.yml": f"{base_url}/Models/LibriTTS/config.yml",
        "epochs_2nd_00020.pth": f"{base_url}/Models/LibriTTS/epochs_2nd_00020.pth",
    }
    
    # PLBERT files
    plbert_files = {
        "config.yml": f"{base_url}/Utils/PLBERT/config.yml",
        "step_1000000.t7": f"{base_url}/Utils/PLBERT/step_1000000.t7",
    }
    
    print("Downloading StyleTTS2 LibriTTS model files...")
    print("This may take a few minutes (checkpoint is ~400MB, PLBERT is ~50MB)")
    print()
    
    success = True
    for filename, url in files.items():
        dest_path = models_dir / filename
        if not download_file(url, dest_path):
            success = False
    
    # Download PLBERT files
    for filename, url in plbert_files.items():
        dest_path = plbert_dir / filename
        if not download_file(url, dest_path):
            success = False
    
    print()
    if success:
        print("✓ All model files downloaded successfully!")
        print()
        print("You can now use StyleTTS2 voices like 'adjutant':")
        print('  voice adjutant "Affirmative, Commander. All systems operational."')
    else:
        print("✗ Some downloads failed. Please check your internet connection and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
