#!/usr/bin/env python3
"""Download audio clips from direct URLs into data/sample_audio/."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_audio"

# Add a filename and its direct download URL for each clip.
# For example:
# AUDIO_URLS = {
#     "en_sample.mp3": "https://example.com/audio/en_sample.mp3",
#     "hi_sample.mp3": "https://example.com/audio/hi_sample.mp3",
# }
AUDIO_URLS: dict[str, str] = {}


def download_audio() -> None:
    """Download each configured clip, reporting network and HTTP errors."""
    if not AUDIO_URLS:
        raise ValueError(
            "Add one or more filename-to-URL entries to AUDIO_URLS before running."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in AUDIO_URLS.items():
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ValueError(f"Invalid output filename: {filename!r}")

        parsed_url = urlparse(url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError(f"Use a valid HTTPS direct-download URL for {filename!r}.")

        destination = OUTPUT_DIR / filename
        try:
            with urlopen(url, timeout=60) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {filename!r}: {exc}") from exc

        print(f"Downloaded {filename} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    download_audio()
