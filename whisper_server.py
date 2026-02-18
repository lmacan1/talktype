#!/usr/bin/env python3
"""
Whisper API Server - Local transcription server for TalkType.

Run this once, keep it running, and TalkType connects to it.
Faster startup since the model stays loaded in memory.

Usage:
    python whisper_server.py                    # Default: base model, auto device
    python whisper_server.py --model small      # Use small model
    python whisper_server.py --port 8080        # Different port
    CUDA_VISIBLE_DEVICES=0 python whisper_server.py  # Specific GPU

Then run TalkType with:
    python talktype.py --api http://localhost:8002/transcribe
"""

import argparse
import os
import tempfile
from typing import Dict, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel
import uvicorn

# === Configuration ===
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "base")
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # "auto", "cuda", or "cpu"
DEFAULT_COMPUTE = os.getenv("WHISPER_COMPUTE", "auto")  # "auto", "float16", "int8"

# === Model Cache ===
_models: Dict[Tuple[str, str, str], WhisperModel] = {}


def get_model(name: str, device: str, compute: str) -> WhisperModel:
    """Get or load a Whisper model (cached)."""
    key = (name, device, compute)
    if key not in _models:
        print(f"Loading model: {name} (device={device}, compute={compute})...")
        _models[key] = WhisperModel(name, device=device, compute_type=compute)
        print("Model loaded.")
    return _models[key]


# === FastAPI App ===
app = FastAPI(
    title="Whisper API",
    description="Local Whisper transcription server for TalkType",
    version="1.0.0"
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "device": DEFAULT_DEVICE,
        "compute": DEFAULT_COMPUTE
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None),
    model: str = Form(None),
):
    """
    Transcribe an audio file.

    - **file**: Audio file (WAV, MP3, etc.)
    - **language**: Language code (e.g., "en", "es"). Auto-detect if not specified.
    - **model**: Model to use (tiny, base, small, medium, large-v3). Uses default if not specified.
    """
    m = model or DEFAULT_MODEL
    whisper = get_model(m, DEFAULT_DEVICE, DEFAULT_COMPUTE)

    # Save uploaded file to temp
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        # Transcribe
        segments, info = whisper.transcribe(tmp.name, language=language)
        segments_list = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        text = "".join(s["text"] for s in segments_list)

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "model": m,
            "segments": segments_list
        }
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass


def main():
    global DEFAULT_MODEL, DEFAULT_DEVICE, DEFAULT_COMPUTE

    parser = argparse.ArgumentParser(description="Whisper API Server")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Whisper model (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", "-d", default=DEFAULT_DEVICE,
                        help=f"Device: auto, cuda, cpu (default: {DEFAULT_DEVICE})")
    parser.add_argument("--compute", "-c", default=DEFAULT_COMPUTE,
                        help=f"Compute type: auto, float16, int8 (default: {DEFAULT_COMPUTE})")
    parser.add_argument("--port", "-p", type=int, default=8002,
                        help="Port to run on (default: 8002)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    # Update defaults from args
    DEFAULT_MODEL = args.model
    DEFAULT_DEVICE = args.device
    DEFAULT_COMPUTE = args.compute

    # Pre-load model
    print(f"Whisper API Server starting on http://{args.host}:{args.port}")
    get_model(DEFAULT_MODEL, DEFAULT_DEVICE, DEFAULT_COMPUTE)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
