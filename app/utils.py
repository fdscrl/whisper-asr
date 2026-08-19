import ctypes
import ctypes.util
import json
import os
from dataclasses import asdict
from typing import BinaryIO, TextIO

import ffmpeg
import numpy as np
from faster_whisper.utils import format_timestamp

from app.config import CONFIG


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, result: dict, audio_path: str):
        audio_basename = os.path.basename(audio_path)
        output_path = os.path.join(self.output_dir, audio_basename + "." + self.extension)

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f)

    def write_result(self, result: dict, file: TextIO):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: TextIO):
        for segment in result["segments"]:
            print(segment.text.strip(), file=file, flush=True)


class WriteVTT(ResultWriter):
    extension: str = "vtt"

    def write_result(self, result: dict, file: TextIO):
        print("WEBVTT\n", file=file)
        for segment in result["segments"]:
            print(
                f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
                f"{segment.text.strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class WriteSRT(ResultWriter):
    extension: str = "srt"

    def write_result(self, result: dict, file: TextIO):
        for i, segment in enumerate(result["segments"], start=1):
            # write srt lines
            print(
                f"{i}\n"
                f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
                f"{segment.text.strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(self, result: dict, file: TextIO):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment.start), file=file, end="\t")
            print(round(1000 * segment.end), file=file, end="\t")
            print(segment.text.strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO):
        if "segments" in result:
            result["segments"] = [asdict(segment) for segment in result["segments"]]
        json.dump(result, file)


def load_audio(file: BinaryIO, encode=True, sr: int = CONFIG.SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if encode:
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def calculate_initial_silence(
    audio_array, sample_rate=16000, silence_threshold=0.01, min_speech_duration=0.2
):
    """
    Calculate initial silence duration in audio.

    This function analyzes the audio waveform to detect the start of speech,
    accounting for initial silence periods. It uses RMS (Root Mean Square)
    energy calculation in sliding windows to identify when speech begins.

    Parameters
    ----------
    audio_array : np.ndarray
        NumPy array containing the audio waveform, in float32 dtype (normalized).
    sample_rate : int, optional
        Sample rate of the audio in Hz. Default is 16000.
    silence_threshold : float, optional
        RMS threshold for silence detection as a fraction of maximum RMS.
        Default is 0.01 (1% of maximum RMS).
    min_speech_duration : float, optional
        Minimum duration of speech in seconds to confirm speech start.
        Default is 0.2 seconds.

    Returns
    -------
    float
        Time offset in seconds until speech starts. Returns 0.0 if:
        - Audio array is empty
        - No speech detected
        - Window size exceeds audio length
    """
    if len(audio_array) == 0:
        return 0.0

    # Calculate window size (100ms windows)
    window_size = int(sample_rate * 0.1)
    if window_size > len(audio_array):
        return 0.0

    # Calculate RMS for each window
    rms_values = []
    for i in range(0, len(audio_array) - window_size, window_size):
        window = audio_array[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)

    if not rms_values:
        return 0.0

    # Find maximum RMS to determine threshold
    max_rms = max(rms_values)
    threshold = max_rms * silence_threshold

    # Find first window above threshold with minimum speech duration
    speech_start_window = None
    consecutive_speech_windows = 0
    required_windows = int(min_speech_duration * 10)  # 100ms windows

    for i, rms in enumerate(rms_values):
        if rms > threshold:
            consecutive_speech_windows += 1
            if consecutive_speech_windows >= required_windows:
                speech_start_window = i - required_windows + 1
                break
        else:
            consecutive_speech_windows = 0

    if speech_start_window is None:
        return 0.0

    # Return time offset in seconds
    offset = (speech_start_window * window_size) / sample_rate
    return max(0.0, offset)


def _load_malloc_trim():
    """
    Look up malloc_trim in libc, returning None when it is not there.

    The symbol is glibc-only; on musl and friends it is missing, and in that
    case trim_heap() turns into a no-op instead of failing at import time.
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        fn = libc.malloc_trim
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return fn


_MALLOC_TRIM = _load_malloc_trim()


def trim_heap():
    """
    Hand memory freed inside the process back to the kernel.

    gc.collect() returns objects to the Python allocator, which returns them
    to glibc, and there the chain stops: the memory never reaches the kernel
    and keeps counting towards RSS. That is expensive for WhisperX. Every
    align-model reload frees ~1.2 GB that settles into malloc arenas as holes,
    so over ten days a worker grows to 8.4 GB against a 4.7 GB working set.
    Four such workers push the Bitrix MySQL into swap, and the OOM-killer
    follows.

    malloc_trim(0) walks the arenas and releases whatever pages it can. The
    call costs milliseconds and runs between files, so it does not show up in
    transcription time.
    """
    if _MALLOC_TRIM is None:
        return
    _MALLOC_TRIM(0)
