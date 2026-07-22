#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

./run_cluster_case.sh >/dev/null
sleep 3

"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new \
  --disable-gpu \
  --hide-scrollbars \
  --window-size=1440,1200 \
  --virtual-time-budget=8000 \
  --screenshot="$SCRIPT_DIR/artifacts/cluster_report_screenshot.png" \
  "file://$SCRIPT_DIR/artifacts/cluster_report.html"

echo "Saved screenshot to $SCRIPT_DIR/artifacts/cluster_report_screenshot.png"
