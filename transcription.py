"""
Transcription layer for ISL / domestic football footage
========================================================
Turns match audio/video (Hindi, Bengali, Tamil, Malayalam, English, etc.)
into timestamped commentary text the scouting engine can read.

Uses OpenAI Whisper if installed. Whisper is multilingual and can either
transcribe in the original language or translate to English.

Install (one of):
    pip install openai-whisper        # local, free, needs ffmpeg
    # or use faster-whisper for speed:  pip install faster-whisper

Usage:
    from transcription import transcribe
    text = transcribe("match.mp4", language="hi", translate_to_english=True)

CLI:
    python transcription.py match.mp4 --language hi --translate
"""

import os
import argparse

SUPPORTED_HINT = "hi (Hindi), bn (Bengali), ta (Tamil), ml (Malayalam), en (English) ..."


def transcribe(path, language=None, translate_to_english=True,
               model_size="small", save=True):
    """
    Transcribe an audio/video file to text.

    path                 : audio or video file
    language             : ISO code (e.g. 'hi'); None = auto-detect
    translate_to_english : if True, output English (Whisper 'translate' task)
                           so the scouting lexicons (English) work directly
    model_size           : tiny / base / small / medium / large
    Returns: transcript string. Also writes <path>.transcript.txt if save=True.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "Whisper not installed. Run:  pip install openai-whisper\n"
            "(also needs ffmpeg on your system). "
            "Until then, feed the scouting engine plain-text commentary directly."
        )

    model = whisper.load_model(model_size)
    task = "translate" if translate_to_english else "transcribe"
    result = model.transcribe(path, language=language, task=task, verbose=False)

    # Build a timestamped transcript (mirrors the 'Min X:' style the engine likes)
    lines = []
    for seg in result.get("segments", []):
        mm = int(seg["start"] // 60)
        ss = int(seg["start"] % 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {seg['text'].strip()}")
    transcript = "\n".join(lines) if lines else result.get("text", "")

    if save:
        out = os.path.splitext(path)[0] + ".transcript.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Transcript saved: {out}")

    return transcript


def transcribe_with_faster_whisper(path, language=None, translate_to_english=True,
                                   model_size="small", save=True):
    """Optional faster backend. Same signature; used if faster-whisper installed."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, compute_type="int8")
    task = "translate" if translate_to_english else "transcribe"
    segments, _ = model.transcribe(path, language=language, task=task)
    lines = []
    for seg in segments:
        mm, ss = int(seg.start // 60), int(seg.start % 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {seg.text.strip()}")
    transcript = "\n".join(lines)
    if save:
        out = os.path.splitext(path)[0] + ".transcript.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Transcript saved: {out}")
    return transcript


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Transcribe match footage to commentary text")
    ap.add_argument("file", help="audio/video file")
    ap.add_argument("--language", default=None, help=f"language code: {SUPPORTED_HINT}")
    ap.add_argument("--translate", action="store_true",
                    help="translate to English (recommended for the scouting engine)")
    ap.add_argument("--model", default="small", help="tiny/base/small/medium/large")
    args = ap.parse_args()
    txt = transcribe(args.file, language=args.language,
                     translate_to_english=args.translate, model_size=args.model)
    print("\n----- transcript preview -----")
    print(txt[:1200])
