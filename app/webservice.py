import importlib.metadata
import os
import random
import signal
from os import path
from threading import Lock
from typing import Annotated, Optional, Union
from urllib.parse import quote

import click
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

# Lazy loading: model will be loaded per worker on first request
asr_model = None
asr_model_lock = Lock()

# Scheduled worker restart, counted in processed files.
#
# Why: over ten days a worker grows from 4.7 to 8.4 GB, and at some point the
# kernel picks the Bitrix mysqld as its victim rather than the worker. Here the
# worker leaves on its own, between files: it finishes sending the current
# response, closes the connection and exits, while the uvicorn supervisor
# starts a fresh one within half a second (keep_subprocess_alive in
# uvicorn/supervisors/multiprocess.py). No request is ever cut short.
#
# The jitter keeps the four workers from recycling in lockstep: each picks its
# own threshold when the process starts.
#
# gunicorn --max-requests is no use here: UvicornWorker never increments
# gunicorn's counter, so the flag silently does nothing.
MAX_REQUESTS_PER_WORKER = int(os.getenv("MAX_REQUESTS_PER_WORKER", "0"))
MAX_REQUESTS_JITTER = int(os.getenv("MAX_REQUESTS_JITTER", "0"))
RECYCLE_AFTER = (
    MAX_REQUESTS_PER_WORKER + random.randint(0, MAX_REQUESTS_JITTER)
    if MAX_REQUESTS_PER_WORKER > 0
    else 0
)

requests_served = 0
recycle_signalled = False
requests_served_lock = Lock()


def note_request_finished():
    """
    Count a finished file and, when due, send the worker off to restart.

    Called from a Starlette background task, that is, only once the response
    body has gone out to the client in full.
    """
    global requests_served, recycle_signalled
    if not RECYCLE_AFTER:
        return
    with requests_served_lock:
        requests_served += 1
        served = requests_served
        # While uvicorn drains its remaining connections more files still
        # arrive here. Signal exactly once: repeated SIGTERMs change nothing
        # but clutter the log.
        if served < RECYCLE_AFTER or recycle_signalled:
            return
        recycle_signalled = True
    print(f"Worker {os.getpid()}: {served} requests served, restarting on schedule")
    os.kill(os.getpid(), signal.SIGTERM)


def load_asr_model():
    """
    Load the model on first use, one per worker, in a thread-safe way.
    """
    global asr_model
    if asr_model is None:
        with asr_model_lock:
            # Double-check pattern: another thread might have loaded it while we waited
            if asr_model is None:
                model = ASRModelFactory.create_asr_model()
                model.load_model()
                asr_model = model
    return asr_model

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectMetadata["Home-page"]},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": "https://github.com/ahmetoner/whisper-asr-webservice/blob/main/LICENCE"},
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
async def asr(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    initial_offset: Union[float, None] = Query(
        default=None,
        description="Initial silence offset in seconds to add to all timestamps",
    ),
    auto_calculate_offset: bool = Query(
        default=False,
        description="Automatically calculate initial silence offset from audio",
    ),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    result = load_asr_model().transcribe(
        load_audio(audio_file.file, encode),
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        {
            "diarize": diarize,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "initial_offset": initial_offset,
            "auto_calculate_offset": auto_calculate_offset,
        },
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
        background=BackgroundTask(note_request_finished),
    )


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detected_lang_code, confidence = load_asr_model().language_detection(load_audio(audio_file.file, encode))
    return JSONResponse(
        content={
            "detected_language": tokenizer.LANGUAGES[detected_lang_code],
            "language_code": detected_lang_code,
            "confidence": confidence,
        },
        background=BackgroundTask(note_request_finished),
    )


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=9000,
    help="Port for the webservice (default: 9000)",
)
@click.option(
    "-w",
    "--workers",
    metavar="WORKERS",
    default=None,
    type=int,
    help="Number of worker processes (default: from UVICORN_WORKERS env var or 1)",
)
@click.version_option(version=projectMetadata["Version"])
def start(host: str, port: Optional[int] = None, workers: Optional[int] = None):
    # Get workers from environment variable or CLI argument
    num_workers = workers if workers is not None else int(os.getenv("UVICORN_WORKERS", "1"))
    # Pass app as import string to enable workers support
    uvicorn.run("app.webservice:app", host=host, port=port, workers=num_workers)


if __name__ == "__main__":
    start()
