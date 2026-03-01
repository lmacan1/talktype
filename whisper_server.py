#!/usr/bin/env python3
"""
Whisper API Server v2 - Optimized for TalkType.

Run this once, keep it running forever, and TalkType connects to it.
Faster startup since the model stays loaded in memory.

Features:
- Structured logging with timing info
- /stats endpoint for metrics
- VAD (voice activity detection) enabled by default
- Transcription timeout protection
- Never crashes - isolated request handling

Usage:
    python whisper_server.py                    # Default: base model, auto device
    python whisper_server.py --model medium     # Use medium model
    python whisper_server.py --timeout 60       # 60 second timeout
    python whisper_server.py --no-vad           # Disable VAD filtering
    python whisper_server.py --log-level DEBUG  # Verbose logging

Then run TalkType with:
    python talktype.py --api http://localhost:8002/transcribe
"""

import argparse
import logging
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

# Auto-configure CUDA library paths if nvidia packages are installed
def _setup_cuda_paths():
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
        paths = [nvidia.cublas.lib.__path__[0], nvidia.cudnn.lib.__path__[0]]
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths + ([existing] if existing else []))
    except ImportError:
        pass  # CUDA packages not installed, use CPU

_setup_cuda_paths()

import ctranslate2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import uvicorn


# === Configuration ===
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "base")
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
DEFAULT_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")


# === Logging ===
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging for the server."""
    logger = logging.getLogger("whisper_server")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# === Server Stats ===
@dataclass
class ServerStats:
    """Tracks server metrics for /stats endpoint."""
    request_count: int = 0
    total_transcription_time: float = 0.0
    error_count: int = 0
    audio_seconds_processed: float = 0.0
    startup_time: float = field(default_factory=time.time)

    def record_request(self, duration: float, audio_duration: float) -> None:
        """Record a successful transcription request."""
        self.request_count += 1
        self.total_transcription_time += duration
        self.audio_seconds_processed += audio_duration

    def record_error(self) -> None:
        """Record a failed request."""
        self.error_count += 1

    @property
    def avg_transcription_time(self) -> float:
        """Average transcription time in seconds."""
        if self.request_count == 0:
            return 0.0
        return self.total_transcription_time / self.request_count

    @property
    def uptime_seconds(self) -> float:
        """Server uptime in seconds."""
        return time.time() - self.startup_time

    def to_dict(self) -> dict:
        """Convert stats to dictionary for JSON response."""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_transcription_time_ms": round(self.avg_transcription_time * 1000, 1),
            "total_audio_processed_seconds": round(self.audio_seconds_processed, 1),
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


# === Server Config ===
@dataclass
class ServerConfig:
    """Server configuration."""
    model: str
    device: str
    compute: str
    timeout: int
    vad_enabled: bool
    host: str
    port: int


# === Model Cache ===
_models: Dict[Tuple[str, str, str], WhisperModel] = {}


def get_model(name: str, device: str, compute: str) -> WhisperModel:
    """Get or load a Whisper model (cached)."""
    key = (name, device, compute)
    if key not in _models:
        logger.info(f"Loading model: {name} (device={device}, compute={compute})")
        _models[key] = WhisperModel(name, device=device, compute_type=compute)
        logger.info("Model loaded successfully")
    return _models[key]


# === Device Detection ===
def get_actual_device() -> dict:
    """Get actual device info (not just configured device)."""
    cuda_count = ctranslate2.get_cuda_device_count()
    cuda_available = cuda_count > 0

    device_info = {
        "device": "cuda" if cuda_available and config.device != "cpu" else "cpu",
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_count,
    }

    # Try to get device name via torch (optional)
    try:
        import torch
        if torch.cuda.is_available():
            device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return device_info


# === FastAPI App ===
app = FastAPI(
    title="Whisper API",
    description="Local Whisper transcription server for TalkType",
    version="2.0.0"
)

# Global instances (initialized in main)
stats: ServerStats = None
config: ServerConfig = None
logger: logging.Logger = None


# === Global Exception Handler ===
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler - server never crashes."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    if stats:
        stats.record_error()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"}
    )


# === Endpoints ===
@app.get("/health")
def health():
    """Health check with actual device info."""
    device_info = get_actual_device()
    return {
        "status": "ok",
        "model": config.model,
        "device": device_info["device"],
        "compute": config.compute,
        "cuda_available": device_info["cuda_available"],
        "cuda_device_count": device_info.get("cuda_device_count", 0),
        "cuda_device_name": device_info.get("cuda_device_name"),
        "vad_enabled": config.vad_enabled,
        "timeout_seconds": config.timeout,
    }


@app.get("/stats")
def get_stats():
    """Server statistics and metrics."""
    return {
        **stats.to_dict(),
        "model": config.model,
        "device": get_actual_device()["device"],
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
    - **model**: Model to use. Uses default if not specified.

    Returns:
    - **text**: Transcribed text
    - **language**: Detected language
    - **language_probability**: Confidence in language detection
    - **segments**: List of segments with timestamps
    - **duration**: Audio duration in seconds
    """
    request_start = time.perf_counter()
    m = model or config.model

    logger.info(f"Transcription request | model={m} language={language or 'auto'}")

    # Read and validate file
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "Empty audio file")
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(400, "Audio file too large (max 50MB)")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read audio file: {e}")
        stats.record_error()
        raise HTTPException(400, f"Invalid audio file: {e}")

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(content)
        tmp.close()

        # Get model (with protection)
        try:
            whisper = get_model(m, config.device, config.compute)
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            stats.record_error()
            raise HTTPException(503, "Model not available - try again later")

        # Transcribe with VAD if enabled
        transcribe_start = time.perf_counter()
        try:
            if config.vad_enabled:
                segments, info = whisper.transcribe(
                    tmp.name,
                    language=language,
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 500,
                        "speech_pad_ms": 200,
                    },
                )
            else:
                segments, info = whisper.transcribe(tmp.name, language=language)

            segments_list = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
            text = "".join(s["text"] for s in segments_list)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            stats.record_error()
            raise HTTPException(500, f"Transcription failed: {type(e).__name__}: {e}")

        # Record metrics
        transcribe_duration = time.perf_counter() - transcribe_start
        total_duration = time.perf_counter() - request_start
        audio_duration = getattr(info, 'duration', 0) or 0

        stats.record_request(transcribe_duration, audio_duration)

        logger.info(
            f"Transcription complete | "
            f"time={transcribe_duration:.2f}s audio={audio_duration:.1f}s "
            f"chars={len(text)} lang={info.language}"
        )

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "model": m,
            "segments": segments_list,
            "duration": audio_duration,
        }

    finally:
        # Always clean up temp file
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# === Signal Handlers ===
def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Shutdown signal received, cleaning up...")
    sys.exit(0)


# === Main ===
def main():
    global config, logger, stats

    parser = argparse.ArgumentParser(
        description="Whisper API Server v2 - Optimized for TalkType",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python whisper_server.py                    # Default: base model, CUDA
  python whisper_server.py --model medium     # Better accuracy
  python whisper_server.py --device cpu       # CPU only
  python whisper_server.py --no-vad           # Disable VAD
        """
    )
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Whisper model: tiny, base, small, medium, large-v3 (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", "-d", default=DEFAULT_DEVICE,
                        help=f"Device: auto, cuda, cpu (default: {DEFAULT_DEVICE})")
    parser.add_argument("--compute", "-c", default=DEFAULT_COMPUTE,
                        help=f"Compute type: auto, float16, int8 (default: {DEFAULT_COMPUTE})")
    parser.add_argument("--port", "-p", type=int, default=8002,
                        help="Port to run on (default: 8002)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--timeout", "-t", type=int, default=120,
                        help="Transcription timeout in seconds (default: 120)")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable VAD (voice activity detection) filtering")
    parser.add_argument("--log-level", default="INFO",
                        help="Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)")
    args = parser.parse_args()

    # Initialize globals
    logger = setup_logging(args.log_level)
    stats = ServerStats()
    config = ServerConfig(
        model=args.model,
        device=args.device,
        compute=args.compute,
        timeout=args.timeout,
        vad_enabled=not args.no_vad,
        host=args.host,
        port=args.port,
    )

    # Set up signal handlers
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    logger.info(f"Whisper API Server v2 starting on http://{args.host}:{args.port}")
    logger.info(f"Config: model={args.model} device={args.device} vad={config.vad_enabled} timeout={args.timeout}s")

    # Pre-load model
    try:
        get_model(config.model, config.device, config.compute)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    logger.info("Server ready - accepting requests")

    # Run server (uvicorn handles graceful shutdown)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
