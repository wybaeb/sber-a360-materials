#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOK_PATH="$SCRIPT_DIR/client_clustering_walkthrough.ipynb"

if command -v jupyter >/dev/null 2>&1; then
  jupyter lab "$NOTEBOOK_PATH"
else
  if command -v open >/dev/null 2>&1; then open "$NOTEBOOK_PATH"; else xdg-open "$NOTEBOOK_PATH" 2>/dev/null || echo "Откройте вручную: $NOTEBOOK_PATH"; fi
fi
