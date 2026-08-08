#!/usr/bin/env bash
# Перечитать TLS-сертификаты в asr-gateway.
#
# certbot обновляет сертификат примерно раз в 60 дней и переставляет симлинки
# в /etc/letsencrypt/live/. nginx держит открытым старый файл, поэтому после
# обновления нужен reload — иначе он продолжит отдавать просроченный сертификат.
#
# Ставится в cron:
#   17 4 * * * /opt/whisper-asr/scripts/reload-asr-gateway.sh >/dev/null 2>&1
set -euo pipefail

CONTAINER=$(docker ps -q \
    --filter "label=com.docker.compose.project=whisper-asr" \
    --filter "label=com.docker.compose.service=asr-gateway")

if [ -z "$CONTAINER" ]; then
    echo "asr-gateway не запущен — reload пропущен" >&2
    exit 0
fi

# Сначала проверяем конфиг: битый конфиг не должен уронить работающий nginx.
docker exec "$CONTAINER" nginx -t
docker exec "$CONTAINER" nginx -s reload
echo "asr-gateway: сертификаты перечитаны"
