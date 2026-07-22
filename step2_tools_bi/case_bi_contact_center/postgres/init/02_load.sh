#!/usr/bin/env bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<'SQL'
\copy contact_center_training FROM '/seed/contact_center_speech_analytics.csv' WITH (FORMAT csv, HEADER true)
SQL
