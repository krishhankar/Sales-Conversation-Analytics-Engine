#!/usr/bin/env python3
"""Instructions for downloading Mozilla Common Voice audio clips.

As of 2026, Common Voice clips are exclusively hosted on the Mozilla Data
Collective (MDC) portal. They are CC0-licensed and free, but require creating
access credentials on the portal.

This script provides instructions — it does NOT auto-download.
For an out-of-the-box demo without MDC signup, use generate_tts_samples.py instead.
"""


def main():
    print("=" * 70)
    print("Mozilla Common Voice — Manual Download Instructions")
    print("=" * 70)
    print()
    print("As of 2026, Common Voice clips are hosted exclusively on the")
    print("Mozilla Data Collective portal.")
    print()
    print("Steps:")
    print("  1. Visit: https://datacollective.mozillafoundation.org/")
    print("  2. Create free access credentials (no payment required)")
    print("  3. Search for your target languages (e.g., Tamil, Hindi, English)")
    print("  4. Download a small subset of validated clips (5-10 per language)")
    print("  5. Place the audio files in: data/sample_audio/")
    print()
    print("Recommended languages for multilingual demo:")
    print("  - English (en)")
    print("  - Hindi (hi)")
    print("  - Tamil (ta)")
    print()
    print("File format: Common Voice clips are typically MP3.")
    print("faster-whisper supports MP3, WAV, FLAC, and other formats.")
    print()
    print("License: CC0 (Public Domain) — free for any use.")
    print()
    print("-" * 70)
    print("ALTERNATIVE: For a quick demo without MDC signup, run:")
    print("  python scripts/generate_tts_samples.py")
    print("This generates synthetic audio using gTTS (free, no API key).")
    print("=" * 70)


if __name__ == "__main__":
    main()
