#!/usr/bin/env bash
# Rebuild the whisper workers on a schedule, without downtime.
#
# Workers drift upwards in memory: a 4.7 GB working set becomes 8.4 GB in the
# process after ten days. The main treatment is MALLOC_ARENA_MAX, malloc_trim
# and the restart-by-file-count (MAX_REQUESTS_PER_WORKER), all of it in
# docker-compose.yml. This script is the backstop in case any of that falls
# short: once a day every worker is guaranteed to be fresh.
#
# Why SIGHUP rather than `docker compose restart`:
# the uvicorn supervisor handles HUP as restart_all, replacing workers ONE AT
# A TIME. Each finishes the file it holds while the others keep serving, so
# the service never disappears for a moment and the listening socket stays
# open. `docker compose restart` instead stops everything at once and cuts
# transcriptions short.
#
# Installed in cron:
#   40 4 * * * /opt/whisper-asr/scripts/recycle-asr-workers.sh >/dev/null 2>&1
set -euo pipefail

CONTAINER=$(docker ps -q \
    --filter "label=com.docker.compose.project=whisper-asr" \
    --filter "label=com.docker.compose.service=whisper-asr-webservice")

if [ -z "$CONTAINER" ]; then
    echo "whisper-asr-webservice is not running - worker rebuild skipped" >&2
    exit 0
fi

docker kill -s HUP "$CONTAINER" >/dev/null
echo "whisper-asr-webservice: workers restarted one at a time"
