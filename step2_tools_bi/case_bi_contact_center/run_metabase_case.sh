#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Данные уже лежат в репозитории (data/contact_center_speech_analytics.csv) —
# пересборка не нужна. Нужен только запущенный Docker.
chmod +x postgres/init/02_load.sh

if ! docker ps >/dev/null 2>&1; then
  echo "Docker не запущен. Запустите Docker (Desktop) и повторите." >&2
  exit 1
fi

docker compose up -d
"$PYTHON_BIN" setup_metabase_case.py

RUNTIME_VALUES="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

runtime = json.loads(Path("artifacts/metabase_runtime.json").read_text(encoding="utf-8"))
print("\t".join([
    runtime["admin_url"],
    runtime["public_url"],
    runtime["admin_email"],
    runtime["admin_password"],
]))
PY
)"
IFS=$'\t' read -r ADMIN_URL PUBLIC_URL ADMIN_EMAIL ADMIN_PASSWORD <<<"$RUNTIME_VALUES"

echo "Metabase готов."
echo "Конструктор дашборда: $ADMIN_URL"
echo "Публичный просмотр: $PUBLIC_URL"
echo "Логин: $ADMIN_EMAIL"
echo "Пароль: $ADMIN_PASSWORD  (учебный пароль локального стенда — не секрет)"
if command -v open >/dev/null 2>&1; then open "$ADMIN_URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$ADMIN_URL"; fi
