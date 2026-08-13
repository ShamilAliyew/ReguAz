from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

MODEL_REPO = "unsloth/gemma-4-E4B-it-GGUF"
MODEL_FILE = "gemma-4-E4B-it-Q4_K_M.gguf"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "backend" / "reguaz" / "models"


def main() -> None:
    """Download a GGUF model from Hugging Face."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = MODELS_DIR / MODEL_FILE

    # -----------------------------------------------------------------
    # Skip download if model already exists
    # -----------------------------------------------------------------

    if output_file.exists():
        print(f"Model already exists:\n{output_file}")
        return

    # -----------------------------------------------------------------
    # Check HF CLI
    # -----------------------------------------------------------------

    if shutil.which("hf") is None:
        print(
            "\nERROR: Hugging Face CLI ('hf') is not installed.\n\n"
            "Install it using:\n"
            "    pip install -U huggingface_hub\n\n"
            "Then login (only if the repository requires authentication):\n"
            "    hf auth login\n"
        )
        sys.exit(1)

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------

    command = [
        "hf",
        "download",
        MODEL_REPO,
        MODEL_FILE,
        "--local-dir",
        str(MODELS_DIR),
    ]

    print("\nDownloading model...\n")
    print("Command:")
    print(" ".join(command))
    print()

    try:
        subprocess.run(
            command,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print("\nDownload failed.")
        print(
            "\nPossible reasons:\n"
            "  • Repository name is incorrect.\n"
            "  • File name is incorrect.\n"
            "  • Repository is private.\n"
            "  • Internet connection failed.\n"
        )
        raise exc

    print("\nDownload completed successfully.")
    print(f"Model saved to:\n{output_file}")


if __name__ == "__main__":
    main()