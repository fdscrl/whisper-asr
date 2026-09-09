import gc
import time
from io import StringIO
from threading import Thread
from typing import BinaryIO, Union

import whisperx
from whisperx.audio import N_SAMPLES
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import ResultWriter, SubtitlesWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG
from app.utils import calculate_initial_silence, trim_heap


class WhisperXASR(ASRModel):
    @staticmethod
    def _empty_model():
        return {'whisperx': None, 'diarize_model': None, 'align_model': {}}

    def __init__(self):
        super().__init__()
        self.model = self._empty_model()

    def release_model(self):
        """
        Release the models after an idle period.

        The base implementation sets self.model to None, which broke WhisperX:
        load_model() writes to self.model['whisperx'], so the first request
        after an idle period raised TypeError. Empty the dict instead — the
        structure stays, only the contents go.
        """
        self.model = self._empty_model()
        gc.collect()
        trim_heap()
        print("Model unloaded due to timeout")

    def load_model(self):
        asr_options = {"without_timestamps": False}
        self.model['whisperx'] = whisperx.load_model(
            CONFIG.MODEL_NAME,
            device=CONFIG.DEVICE,
            compute_type=CONFIG.MODEL_QUANTIZATION,
            asr_options=asr_options
        )

        if CONFIG.HF_TOKEN != "":
            self.model['diarize_model'] = DiarizationPipeline(
                use_auth_token=CONFIG.HF_TOKEN,
                device=CONFIG.DEVICE
            )

        # Pin the align models the call queue actually uses, so a mixed
        # uk/ru/pl stream never pays the reload when the language changes.
        for lang in CONFIG.ALIGN_MODEL_LANGUAGES:
            self._align_model_for(lang)

        Thread(target=self.monitor_idleness, daemon=True).start()

    def _align_model_for(self, lang):
        """
        Return (model, metadata) for `lang`, loading it if it is not cached.

        Callers must already hold self.model_lock; this method never takes it.

        Languages in ALIGN_MODEL_LANGUAGES are pinned and stay for the life of
        the worker. That is affordable because the weights come from mmap'd
        safetensors: measured on this host, three pinned languages add ~3.6 GiB
        of page cache shared by every worker, and nothing to private memory.

        Any other language gets a single slot and is evicted as soon as a
        different unpinned language shows up, so one stray request for an
        unexpected language cannot grow the cache without bound.
        """
        cache = self.model['align_model']
        if lang in cache:
            return cache[lang]

        if lang not in CONFIG.ALIGN_MODEL_LANGUAGES:
            evicted = [c for c in cache if c not in CONFIG.ALIGN_MODEL_LANGUAGES]
            for old_lang in evicted:
                del cache[old_lang]
            if evicted:
                gc.collect()
                trim_heap()
                print(f"Align model cache: evicted unpinned {evicted}")

        print(f"Align model cache: loading '{lang}'")
        cache[lang] = whisperx.load_align_model(language_code=lang, device=CONFIG.DEVICE)
        return cache[lang]

    def transcribe(
        self,
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
        output,
    ):
        self.last_activity_time = time.time()
        with self.model_lock:
            if self.model['whisperx'] is None:
                self.load_model()

        # Язык не задан — определяем сами по нескольким окнам и передаём дальше
        # явно. whisperX тогда пропускает собственную детекцию по первым 30 с,
        # а вызывающая сторона получает confidence прямо в ответе /asr.
        language_confidence = None
        if not language:
            language, language_confidence = self.language_detection(audio)
            print(f"Auto-detected language: {language} (confidence {language_confidence})")
        else:
            print(f"Using specified language: {language}")

        options_dict = {"task": task, "language": language}
        if initial_prompt:
            options_dict["initial_prompt"] = initial_prompt
        with self.model_lock:
            result = self.model['whisperx'].transcribe(audio, **options_dict)
            language = result["language"]

        with self.model_lock:
            model_x, metadata = self._align_model_for(result["language"])

        # Align whisper output
        result = whisperx.align(
            result["segments"], model_x, metadata, audio, CONFIG.DEVICE, return_char_alignments=False
        )

        if options.get("diarize", False) and CONFIG.HF_TOKEN != "":
            min_speakers = options.get("min_speakers", None)
            max_speakers = options.get("max_speakers", None)
            # add min/max number of speakers if known
            diarize_segments = self.model['diarize_model'](audio, min_speakers, max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        result["language"] = language
        # None, если язык был задан явно; число — если определяли сами.
        result["language_confidence"] = language_confidence

        # Apply initial silence offset if specified
        offset = 0.0
        if options and options.get("initial_offset") is not None:
            offset = float(options["initial_offset"])
        elif options and options.get("auto_calculate_offset", False):
            offset = calculate_initial_silence(audio)

        if offset > 0:
            for segment in result["segments"]:
                segment["start"] += offset
                segment["end"] += offset
                # Apply offset to word timestamps if present
                if "words" in segment and segment["words"]:
                    for word in segment["words"]:
                        word["start"] += offset
                        word["end"] += offset

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)

        # The intermediate tensors for this file are freed, but glibc holds
        # on to them. Give them back to the kernel here, between files,
        # rather than accumulating until the OOM-killer steps in.
        gc.collect()
        trim_heap()
        # Stamp activity on completion as well, otherwise a long transcription
        # looks like idleness to monitor_idleness.
        self.last_activity_time = time.time()

        return output_file

    # Сколько 30-секундных окон опрашивать и где их брать (доля от полезной длины).
    LANGUAGE_DETECTION_WINDOWS = (0.0, 0.5, 0.85)

    def language_detection(self, audio):
        """
        Определяет язык по нескольким окнам, разнесённым по всей записи.

        Штатный language_detection_segments у faster-whisper берёт окна подряд
        от начала и обрывается на первом же уверенном — для звонка, который
        начинается с приветствия, гудков или музыки, это бесполезно. Здесь окна
        разнесены по началу, середине и концу разговора.

        Возвращает (код языка, confidence), где confidence — доля победителя в
        сумме вероятностей по окнам. Она падает и когда модель не уверена, и
        когда окна расходятся между собой: для смешанных звонков это ровно то,
        что нужно отлавливать порогом.
        """
        with self.model_lock:
            if self.model['whisperx'] is None:
                self.load_model()

            total = audio.shape[0]
            if total < N_SAMPLES:
                print("Warning: audio is shorter than 30s, language detection may be inaccurate.")

            if total <= N_SAMPLES:
                starts = [0]
            else:
                usable = total - N_SAMPLES
                starts = sorted({int(usable * f) for f in self.LANGUAGE_DETECTION_WINDOWS})

            scores = {}
            probed = []
            for start in starts:
                window = audio[start:start + N_SAMPLES]
                # Огрызок короче 5 с — шума больше, чем пользы.
                if window.shape[0] < N_SAMPLES // 6:
                    continue
                lang, prob, _ = self.model['whisperx'].model.detect_language(
                    window, language_detection_segments=1
                )
                prob = float(prob)
                scores[lang] = scores.get(lang, 0.0) + prob
                probed.append(f"{start // 16000}s:{lang}={prob:.2f}")

            # Запись короче 5 с: все окна отсеялись — спрашиваем по целому файлу.
            if not scores:
                lang, prob, _ = self.model['whisperx'].model.detect_language(
                    audio, language_detection_segments=1
                )
                scores[lang] = float(prob)
                probed.append(f"full:{lang}={float(prob):.2f}")

            language = max(scores, key=scores.get)
            # Делим на ЧИСЛО ОКОН, а не на сумму вероятностей. Иначе три окна,
            # согласно указавшие один язык с жалкой вероятностью 0.39 каждое
            # (шум, музыка, тишина), дали бы confidence 1.0. При таком делении
            # метрика падает и от неуверенности модели, и от расхождения окон.
            language_probability = round(scores[language] / len(probed), 2)
            print(
                f"Detected language: {language} ({language_probability}) "
                f"from {len(probed)} window(s): {', '.join(probed)}"
            )
        return language, language_probability


    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        default_options = {
            "max_line_width": CONFIG.SUBTITLE_MAX_LINE_WIDTH,
            "max_line_count": CONFIG.SUBTITLE_MAX_LINE_COUNT,
            "highlight_words": CONFIG.SUBTITLE_HIGHLIGHT_WORDS
        }

        if output == "srt":
            WriteSRT(SubtitlesWriter).write_result(result, file=file, options=default_options)
        elif output == "vtt":
            WriteVTT(SubtitlesWriter).write_result(result, file=file, options=default_options)
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file, options=default_options)
        elif output == "json":
            WriteJSON(ResultWriter).write_result(result, file=file, options=default_options)
        else:
            WriteTXT(ResultWriter).write_result(result, file=file, options=default_options)
