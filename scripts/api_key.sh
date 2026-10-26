#!/usr/bin/env bash
# Affiche la clé d'API (en-tête X-API-Key) pour appeler l'API de qualification.
set -euo pipefail
grep -E '^API_KEY=' "$(cd "$(dirname "$0")/.." && pwd)/.env" | cut -d= -f2-
