# Внешний доступ к расшифровке

## Схема

```
Внешний клиент ──HTTPS :9443 + X-API-Key──> asr-gateway (nginx) ──┐
                                                                  ├─> whisper :9000
Sales Evaluator ──HTTP 172.17.0.1:9000, без ключа ───────────────┘

Интернет ──> :9000   порт больше не слушает публичный IP
```

Порт `9000` привязан только к `172.17.0.1` (шлюз `docker0`) и `127.0.0.1`.
Sales Evaluator обращается по `ASR_API_URL=http://host.docker.internal:9000`,
что внутри его контейнеров резолвится ровно в `172.17.0.1` — поэтому он
продолжает работать без ключа и без каких-либо изменений на своей стороне.

## Применение

Ничего не применяется автоматически. Порядок:

```bash
cd /opt/whisper-asr

# 1. Проверить конфиг (ничего не запускает)
docker compose config >/dev/null && echo OK

# 2. Поднять. whisper перезапустится — ~минута простоя расшифровки.
#    Первый запрос после старта дольше обычного: модель грузится лениво.
docker compose up -d

# 3. Убедиться, что оба контейнера живы
docker compose ps
```

### Обязательная проверка после применения

```bash
# а) Sales Evaluator по-прежнему достаёт whisper без ключа
docker exec transcription sh -c \
  'curl -s -o /dev/null -w "internal: %{http_code}\n" http://host.docker.internal:9000/openapi.json'
# ожидается 200

# б) снаружи порт 9000 закрыт
curl -s --max-time 5 -o /dev/null -w "public 9000: %{http_code}\n" \
  http://195.248.225.99:9000/openapi.json
# ожидается пусто/таймаут (000)

# в) шлюз без ключа не пускает
curl -s -o /dev/null -w "no key: %{http_code}\n" \
  https://app.24print.ua:9443/openapi.json
# ожидается 401

# г) шлюз с ключом пускает
curl -s -o /dev/null -w "with key: %{http_code}\n" \
  -H "X-API-Key: $(grep ASR_API_KEY /opt/whisper-asr/.env | cut -d= -f2)" \
  https://app.24print.ua:9443/openapi.json
# ожидается 200
```

Если (а) вернуло не 200 — расшифровка звонков встала. Откат: вернуть в
`docker-compose.yml` строку `- "9000:9000"` и выполнить `docker compose up -d`.

### Обновление сертификата

certbot обновляет сертификат `app.24print.ua` примерно раз в 60 дней
(текущий действует до 18 октября 2026). nginx держит открытым старый файл,
поэтому нужен reload. Поставить в cron:

```bash
crontab -e
# добавить:
17 4 * * * /opt/whisper-asr/scripts/reload-asr-gateway.sh >/dev/null 2>&1
```

### Смена ключа

Вписать новое значение `ASR_API_KEY` в `.env`, затем:

```bash
docker compose up -d asr-gateway   # whisper не трогается, расшифровка не прерывается
```

Сгенерировать новый ключ: `openssl rand -hex 32`

---

## Инструкция для агента

**Задача:** отправить mp3-файл на расшифровку с диаризацией на удалённый Whisper ASR.

**Эндпоинт:** `https://app.24print.ua:9443/asr` — POST, `multipart/form-data`,
TLS с валидным сертификатом (`-k` не нужен).

**Авторизация:** заголовок `X-API-Key: <ключ>`. Альтернативно принимается
`Authorization: Bearer <ключ>`. Без ключа сервис отвечает `401`.
Ключ выдаётся владельцем сервера; в репозитории его нет.

**Команда:**

```bash
curl -X POST \
  "https://app.24print.ua:9443/asr?task=transcribe&language=uk&output=json&encode=true&diarize=true&min_speakers=2&max_speakers=2&auto_calculate_offset=true" \
  -H "X-API-Key: ВАШ_КЛЮЧ" \
  -H "accept: application/json" \
  -F "audio_file=@/путь/к/файлу.mp3;type=audio/mpeg" \
  --max-time 3600 \
  -o result.json \
  -w "\nHTTP=%{http_code} time=%{time_total}s\n"
```

**Параметры — в query string, не в form-data:**

| Параметр | Значение | Зачем |
|---|---|---|
| `diarize` | `true` | **Ключевой.** По умолчанию `false`, без него разделения по спикерам не будет |
| `min_speakers` / `max_speakers` | `2` / `2` | Для звонка «менеджер ↔ клиент». Если число спикеров неизвестно — убрать оба |
| `output` | `json` | **Обязательно.** Метки спикеров есть только в JSON; в `txt`/`srt`/`vtt` теряются |
| `language` | `uk` | Код языка. Убрать — будет автоопределение, но медленнее |
| `task` | `transcribe` | `translate` переведёт на английский |
| `encode` | `true` | Прогон через ffmpeg на сервере, для mp3 обязательно |
| `auto_calculate_offset` | `true` | Компенсация начальной тишины в таймкодах |

Имя поля файла строго `audio_file`.

**Ожидаемый ответ:**

```json
{
  "segments": [
    {
      "start": 1.23, "end": 4.56,
      "text": "Добрий день, компанія...",
      "speaker": "SPEAKER_00",
      "words": [{"word": "Добрий", "start": 1.23, "end": 1.51, "speaker": "SPEAKER_00"}]
    }
  ],
  "language": "uk"
}
```

**Проверка результата:**

```bash
python3 -c "
import json; d=json.load(open('result.json'))
print('Спикеры:', set(s.get('speaker') for s in d['segments']))
for s in d['segments'][:5]: print(f\"[{s['start']:.1f}] {s.get('speaker','?')}: {s['text']}\")
"
```

Если `speaker` везде `null` или ключа нет — `diarize=true` не долетел.
Частая ошибка: параметр положили в form-data вместо query string.

**Время выполнения.** Модель `large-v3` работает на CPU с квантизацией int8,
GPU нет. Расшифровка идёт медленнее реального времени. Первый запрос с
`diarize=true` дополнительно грузит модель pyannote — это несколько лишних минут.
Для 10-минутного звонка закладывайте 15–40 минут. Таймаут от 3600 секунд,
ретраи по таймауту не делать — они удвоят нагрузку на сервер, который
параллельно расшифровывает боевой поток звонков.

**Ограничения шлюза:** максимум 3 одновременных внешних соединения
(чтобы не занять все воркеры), максимальный размер файла 1 ГБ.
Оба лимита правятся в `nginx/asr.conf.template`.

**Коды ответов:**

| Код | Значение |
|---|---|
| `200` | Успех |
| `401` | Ключ отсутствует или неверный |
| `413` | Файл больше 1 ГБ |
| `503` | Превышен лимит одновременных соединений — повторить позже |
| `504` | Расшифровка не уложилась в час |
