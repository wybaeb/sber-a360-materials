#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Данные и готовый отчёт уже лежат в репозитории (data/, artifacts/) —
# пересборка не нужна, просто открываем отчёт.
REPORT_PATH="$SCRIPT_DIR/artifacts/cluster_report.html"
REPORT_URL="file://$REPORT_PATH"
echo "Открываю отчёт по кластеризации: $REPORT_URL"
if command -v open >/dev/null 2>&1; then
  open "$REPORT_URL"          # macOS
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$REPORT_URL"      # Linux
else
  echo "Откройте файл вручную в браузере: $REPORT_PATH"
fi
