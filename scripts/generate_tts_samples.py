#!/usr/bin/env python3
"""Generate multilingual TTS audio samples for demonstrating faster-whisper transcription.

Uses gTTS (Google Text-to-Speech) — free, no API key, no signup required.
Generates short sales conversation phrases in English, Hindi, and Tamil.

Usage:
    python scripts/generate_tts_samples.py

Output:
    data/sample_audio/en_sales_sample.mp3
    data/sample_audio/hi_sales_sample.mp3
    data/sample_audio/ta_sales_sample.mp3
"""

import os
from pathlib import Path

from gtts import gTTS

# Output directory
OUTPUT_DIR = Path("data/sample_audio")

# Sample sales phrases in 3 languages
SAMPLES = [
    {
        "lang": "en",
        "filename": "en_sales_sample.mp3",
        "text": (
            "Hello, thank you for taking the time to speak with me today. "
            "I'd like to discuss our enterprise analytics platform and how it can help "
            "your team reduce manual reporting by up to 60 percent. "
            "We currently offer three pricing tiers, and I think the professional plan "
            "would be the best fit for your organization. "
            "Would you be interested in scheduling a product demo next week?"
        ),
    },
    {
        "lang": "hi",
        "filename": "hi_sales_sample.mp3",
        "text": (
            "नमस्ते, आज मुझसे बात करने के लिए धन्यवाद। "
            "मैं आपके साथ हमारे एंटरप्राइज एनालिटिक्स प्लेटफॉर्म के बारे में चर्चा करना चाहता हूँ। "
            "यह आपकी टीम की मैनुअल रिपोर्टिंग को 60 प्रतिशत तक कम कर सकता है। "
            "हम तीन मूल्य निर्धारण योजनाएं प्रदान करते हैं। "
            "क्या आप अगले सप्ताह एक उत्पाद डेमो शेड्यूल करना चाहेंगे?"
        ),
    },
    {
        "lang": "ta",
        "filename": "ta_sales_sample.mp3",
        "text": (
            "வணக்கம், இன்று என்னுடன் பேச நேரம் ஒதுக்கியதற்கு நன்றி। "
            "எங்கள் நிறுவன பகுப்பாய்வு தளம் பற்றி விவாதிக்க விரும்புகிறேன்। "
            "இது உங்கள் குழுவின் கைமுறை அறிக்கையிடலை 60 சதவீதம் வரை குறைக்க உதவும்। "
            "நாங்கள் மூன்று விலை நிர்ணய திட்டங்களை வழங்குகிறோம்। "
            "அடுத்த வாரம் ஒரு தயாரிப்பு டெமோவை திட்டமிட விரும்புவீர்களா?"
        ),
    },
]


def main():
    """Generate all TTS samples."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        output_path = OUTPUT_DIR / sample["filename"]
        print(f"Generating {sample['lang']} sample: {output_path}")

        tts = gTTS(text=sample["text"], lang=sample["lang"])
        tts.save(str(output_path))

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  ✓ Saved ({size_kb:.1f} KB)")

    print(f"\nAll samples saved to {OUTPUT_DIR}/")
    print("Use these with: curl -X POST -F 'audio=@data/sample_audio/hi_sales_sample.mp3' http://localhost:8000/calls")


if __name__ == "__main__":
    main()
