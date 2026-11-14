# Transcription evaluation: criteria and measurements

Status as of 2026-11-14. This page is the starting point for a study to choose
the best speech-to-text setup for 24print. It has two parts: the criteria
every candidate is judged by, and every measurement taken so far. The quality
results so far are side-by-side reads, not scores. No reference
transcripts exist yet, so no WER has been computed.

## Consumers and constraints

| Consumer | Audio | Speakers come from | Languages | Service today |
|---|---|---|---|---|
| Odoo Discuss call recording (test SFU, 24print-hub D005) | one Opus track per participant, 48 kHz, cut into 5-min chunks | one track per participant | ru, uk mixed | `parakeet-asr`, gateway `:9444` |
| Sales Evaluator Kyiv | Bitrix telephony, stereo (client left, employee right) | channel split | uk (fixed) | whisper-asr `:9000`, `ASR_LANGUAGE=uk` |
| Sales Evaluator Warsaw | Odoo telephony, mono, 8 kHz-class phone audio | diarization (pyannote in whisperx) | pl, ru, uk, one per call or mixed | whisper-asr `:9443`, `ASR_LANGUAGE=auto`, `diarize=true` |

Constraints set by the owner:

- Audio stays on our servers. Cloud APIs are allowed only as a quality reference
  (D005 R10).
- The service runs on CPU on `bitrix-prod-01`, next to the Bitrix MySQL
  (Hetzner SX65: 16 cores, 125 GB RAM). There is no GPU.
- A 4-hour Discuss recording must be transcribed in 30 minutes or less.
- Speed may be given up for quality: twice as slow is acceptable if the result
  is twice as good.

## Criteria

### 1. Text quality

| Criterion | Definition | How to measure |
|---|---|---|
| WER / CER | word / character error rate against a reference transcript | `jiwer` after normalization (lower case, no punctuation, numbers spelled one way), reported per language: ru, uk, pl, mixed |
| Language fidelity | share of segments written in the language actually spoken (no translation, no transliteration) | language ID of each hypothesis segment compared with the reference segment |
| Code-switching loss | speech lost or mangled where the language changes inside a call | seconds of reference speech with no matching hypothesis text around each switch |
| Hallucinations / junk | hypothesis segments with no counterpart in the reference (fillers in the wrong language, invented phrases, repetition loops) | count and share of inserted segments; separate count for segments of 3 words or fewer |
| Deletions | reference speech missing from the hypothesis | seconds and share of reference speech with no hypothesis text |
| Entity accuracy | names of people and companies, sizes, prices, quantities, phone numbers, print terms (ламінація, тиснення, шильда) | tagged entities in the reference, exact-match rate in the hypothesis |
| Punctuation and casing | sentence boundaries and capitals present | punctuation F1 against the reference |

### 2. Speaker attribution (mono audio and multi-party calls)

| Criterion | Definition | How to measure |
|---|---|---|
| DER | diarization error rate: missed speech, false alarm, speaker confusion | `pyannote.metrics` against hand-labelled turns |
| WDER | share of words attributed to the wrong speaker in the final transcript | word-level alignment of hypothesis to reference with speaker labels |
| Role accuracy | employee vs client label correct (Sales Evaluator) | share of calls and of words with the right role |
| More than two speakers | transfers and third parties handled | DER on calls with a transfer (for example Warsaw call 25079) |

### 3. Timestamps

| Criterion | Definition | How to measure |
|---|---|---|
| Segment boundary error | start/end offset against the reference | median and p95 absolute offset, seconds |

Discuss recordings align screen frames to the transcript. Sales Evaluator
stores the time of each line.

### 4. Speed and resources

| Criterion | Definition | How to measure |
|---|---|---|
| RTF | processing time / audio duration (lower is faster) | wall clock per request, single request on an idle service |
| Throughput under load | audio hours processed per hour with the real concurrency (Sales Evaluator Warsaw sends 2 files at once, the SFU 1) | parallel requests, wall clock |
| 4-hour job time | time to transcribe a 4-hour Discuss recording | end to end through the gateway |
| Model load time | container start to `/health` OK | logs |
| Memory | peak RSS per request and at rest | `docker stats` every 5 s |
| CPU | cores used, effect on MySQL on the same host | `docker stats`, `mysqld` CPU, load average |
| Image and model size | disk footprint | `docker image ls`, model files |
| Cost per audio hour | own server amortized, or API price | EUR or USD per hour of audio |

### 5. Operations

- Failure rate: timeouts, OOM kills, 5xx responses per 1000 requests.
- Long input: behaviour on 5-min chunks and on whole calls over 25 s per segment.
- Locality: runs on our servers or sends audio out.
- Licence of model and code.
- Dependencies: PyTorch, Hugging Face token for gated models, ONNX only.
- API compatibility with the whisper-asr `POST /asr` contract (`audio_file`,
  `output=json`, `segments` with `start`, `end`, `text`, optional `speaker`).

### 6. Effect on the evaluation

The final criterion is whether a seller's score stays the same. Evaluate the
same calls from each candidate's transcript and from the reference transcript,
then compare the criterion scores. The criteria above explain the differences.
This one decides.

## Proposed test set

None of it exists yet. Someone has to be named to correct the transcripts by
hand.

- 20 to 30 recordings: Kyiv stereo calls, Warsaw mono calls, Discuss
  recordings with separate tracks.
- Every language: ru, uk, pl, plus calls where the language changes mid-call.
- Hand-corrected reference text with speaker turns and timestamps.
- Optional public baseline: FLEURS ru, uk, pl, 50 to 100 utterances each, for
  WER on clean read speech.

Customer calls contain personal data. Keep the test set out of public
repositories.

## Measurements so far

Hardware labels: "laptop" is the owner's Apple Silicon MacBook, CPU only
(CoreML disabled), 3 threads. "server" is `bitrix-prod-01`.

### Speed

| Implementation | Where | Input | Result | RTF |
|---|---|---|---|---|
| Whisper large-v3-turbo (whisperx, int8, 4 workers × 3 threads, `cpus: 10`) | server | Discuss audio | ≈ 0.9 min per minute of audio; a 4-hour recording ≈ 3.6 h | ≈ 0.9 |
| same | server | 1-min clip, 2 requests in parallel | 102 s | 1.7 |
| same | server | 5-min Discuss chunk | ≈ 4.5 min, longer under load | ≈ 0.9 |
| same, `diarize=true`, auto language | server | 3 Warsaw calls (151 s, 117 s, 266 s) sent in parallel | 117 s call: 253 s; the other two: 767 s each | 2.2 to 5.1 under load |
| Whisper large-v3 (whisperx, same setup) | server | Discuss audio | ≈ 2.6 min per minute of audio | ≈ 2.6 |
| Parakeet TDT 0.6B v3 (onnx-asr 0.12.0, int8, Silero VAD, 3 threads) | laptop | 647 s Discuss mix, VAD `min_silence` 100 ms | 22.5 s | 0.035 |
| same, VAD 800 ms silence / 500 ms speech / 200 ms pad / 25 s max | laptop | same mix | 24.9 s | 0.038 |
| same | laptop | 20 s clip | 0.44 s | 0.022 |
| same, VAD batch 1 (`parakeet-asr` image) | server | 1 h of speech in 5-min chunks | 243 s; a 4-hour recording ≈ 16 min; a 5-min chunk ≤ 22.5 s | 0.068 |
| same, VAD batch 8 | server | 5-min chunks | 25 to 41 s per chunk, OOM kill on the 5th chunk (4 GB limit) | 0.08 to 0.14 |
| Canary 1B v2 (int8, `language=ru`) | laptop | 647 s Discuss mix | 82.9 s | 0.13 |
| GigaAM v3 e2e RNNT | laptop | 647 s Discuss mix | 19.2 s | 0.030 |
| OpenAI gpt-transcribe | cloud | 647 s Discuss mix (mixed track) | 12 s | 0.019 |
| OpenAI gpt-4o-transcribe-diarize | cloud | same | ≈ 5 min | ≈ 0.46 |
| pyannote speaker-diarization-3.1 (PyTorch 2.7.1 CPU, 3 threads) | laptop | Warsaw calls 151 s / 117 s / 266 s | 91.8 s / 72.8 s / 173.3 s | 0.61 / 0.62 / 0.65 |
| sherpa-onnx 2.0.1 diarization: pyannote segmentation 3.0 + WeSpeaker ResNet34-LM, 2 clusters | laptop | same calls | 11.5 s / 7.9 s / 19.7 s | 0.07 to 0.08 |
| sherpa-onnx: pyannote segmentation 3.0 + 3D-Speaker ERes2Net base | laptop | same calls | 11.4 s / 9.3 s / 23.4 s | 0.08 to 0.09 |
| Parakeet per pyannote turn, VAD inside each turn | laptop | same calls | 4.4 s / 4.0 s / 7.8 s | 0.03 |

### Resources

| Implementation | Memory | CPU | Size |
|---|---|---|---|
| whisper-asr (4 workers) | compose limit 28 GB; observed 6.7 to 9.1 GB while idle and under light load; ≈ 19 GB for 4 workers by the compose sizing (not measured) | `cpus: 10` | models: large-v3, large-v3-turbo, wav2vec2 aligners ru/uk/pl, pyannote 3.1 |
| parakeet-asr | peak 1.5 GB of a 4 GB limit with VAD batch 1 (batch 8: 3.7 GB locally, 4.18 GB at OOM on the server); laptop peak RSS 2.6 GB | up to 3 cores (311 %) | model 622 MB (int8) |
| MySQL next to parakeet-asr during the 1-hour run | — | `mysqld` average 53 % of one core, peak 103 % (32 % / 122 % before the run); load average up to 3.1 of 16 | — |

### Cost of a cloud reference

- OpenAI gpt-transcribe: $0.0045 per minute, ≈ $0.27 per audio hour.
- OpenAI gpt-4o-transcribe: $0.006 per minute, retiring February 2027.

### Quality: Discuss test call (owner's call, 10.8 min, mostly Russian)

Notes from reading the outputs side by side. No reference transcript.

- **Whisper large-v3-turbo, per track.** Speaker names are correct (they come from
  the tracks). Long passages have no punctuation or capitals. Errors include
  «подписку плода до расширить» for «подписку Клода расширить», «шоу есть» and
  «покладывать». It invented «Продолжение следует...» at 04:55, a known Whisper
  hallucination on silence.
- **OpenAI gpt-transcribe, mixed track.** The best text: clean punctuation and
  almost no errors («кода» for «Клода»).
- **OpenAI gpt-4o-transcribe-diarize.** The most accurate meaning, but speakers
  are only A/B and phrases are cut into fragments.
- **OpenAI gpt-4o-transcribe.** Unusable: it wrote Russian speech in Ukrainian.
- **Parakeet TDT 0.6B v3.** The main text is on the level of gpt-transcribe,
  with punctuation. Errors include «буквы учет», «уникать», «плода». Junk with the
  default VAD: 44 of 141 lines (11 of them empty), short Latin-script
  fragments on fillers and noise («Yeah», «No», «What»). With the larger VAD
  segments: 18 of 69 lines without Cyrillic, all 1 to 3 words. The SFU now
  drops phrases of 3 words or fewer with no Cyrillic letter (D005 R12). That
  filter cannot be used for Polish.
- **Canary 1B v2.** No junk, but only because the language was fixed to `ru`.
  It does not detect the language. The text is not better («Рубрикуила»).
- **GigaAM v3.** The best Russian text («поклацала», «вникать», «Клода»), but it
  is Russian only.

### Quality: Sales Evaluator Warsaw calls (2026-11-14)

Three mono calls from the evaluator's audio cache: 25079 (151 s, Russian,
transferred to a second employee), 50874 (117 s, Polish) and 50892 (266 s,
Ukrainian). No reference transcript.

| Setup | Language | Speakers | Text |
|---|---|---|---|
| Whisper turbo, `language=pl`, pyannote (production until 2026-11-14) | Russian and Ukrainian translated into Polish, with meaning errors («макеты» → «małżeństwo») | clean turns | complete but translated |
| Whisper turbo, auto language, pyannote (production since 2026-11-14) | detected correctly: ru, pl, uk | clean turns | the most complete and cleanest of all setups, no junk; one language per request, so a 9 s Polish greeting at the start of the Russian call 25079 was lost |
| Parakeet + pyannote 3.1 turns, VAD inside each turn | original language per segment | as good as Whisper's | Polish phone speech clearly worse than Whisper («faza rozwania Pani Zwani» for «Maria, oddzwaniam»); English junk on short turns («Um trailer my kids eh uh what»); 24 s of the client's speech in 50892 came out as English junk |
| Parakeet + sherpa-onnx (WeSpeaker) | original | employee and client often swapped inside one turn | as above |
| Parakeet + sherpa-onnx (3D-Speaker) | original | mostly right, worse than pyannote | as above |
| Parakeet on whole turns over 25 s, no VAD | original | — | text truncated: parts of long turns missing |

Separate defect found: the evaluator's role rule (company name in the greeting,
otherwise who spoke first) labelled the employee as the client in call 50892
and in part of 25079. It does not depend on the ASR engine.

### Known behaviour of the whisper-asr deployment

- The whisperx engine returns 500 on `initial_prompt` per request. It reads the
  prompt only when the model is loaded.
- Language is detected once per request. For the SFU that means once per 5-min
  chunk, and for the evaluator once per call.
- The gateway allows 4 connections in total on `:9443` and has a 3600 s timeout.
  `parakeet-asr` sits on `:9444` with 2 connections and a 600 s timeout.

## Candidates not yet measured

- Speech recognition: Whisper large-v3 on Warsaw and Kyiv calls; Whisper
  large-v3 fine-tuned for Ukrainian or Polish; Canary 1B v2 with a language-ID
  step per segment; Voxtral.
- Diarization: pyannote community-1; NeMo Sortformer.
- Whisper settings: `condition_on_previous_text=false` against hallucinations;
  word alignment switched off for a 10 to 20 % speed estimate (not measured).
- A GPU server or serverless GPU for large-v3 (prices not checked).

## Sources

- 24print-hub #227 («Факты», «Почему Parakeet»), #267 (server measurement),
  #266 (Warsaw comparison, closed 2026-11-14), decision card D005.
- `kyiv-hetzner-infra`: `compose/parakeet-asr/app.py`,
  `ansible/files/bitrix/whisper-asr/`.
- `sales-evaluator`: `app/integrations/asr.py`,
  `app/services/transcription_service.py`; commit `8a4b5dd` (auto language for
  Warsaw).
- The audio and transcripts used here were working copies on the owner's
  laptop. They are not stored in any repository.
