#!/usr/bin/env bash
# Génère le fichier .env (non versionné) du service IA à partir des sorties Terraform de la
# plateforme (bloc 2), et génère une fois pour toutes la clé d'API.
#   PLATFORM_DIR=<chemin d'artelio-data-platform> ./scripts/configure_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLATFORM_DIR="${PLATFORM_DIR:-${ROOT}/../../Bloc2_Architecture/artelio-data-platform}"
ENV_FILE="${ROOT}/.env"
command -v terraform >/dev/null || { echo "Terraform est requis (plateforme du bloc 2)." >&2; exit 1; }

secrets=$(terraform -chdir="${PLATFORM_DIR}/terraform" output -json secrets)
lire() { python3 -c "import sys, json; print(json.load(sys.stdin)['$1'])" <<<"${secrets}"; }
existant() { [[ -f "${ENV_FILE}" ]] && grep -E "^$1=" "${ENV_FILE}" | cut -d= -f2- || true; }

# Clé d'API : seule son empreinte SHA-256 est donnée à l'API (la clé en clair reste chez l'appelant)
API_KEY=$(existant API_KEY); API_KEY=${API_KEY:-$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")}
API_KEY_SHA256=$(printf '%s' "${API_KEY}" | shasum -a 256 | cut -d' ' -f1)

umask 077
cat > "${ENV_FILE}" <<ENV
# Généré par scripts/configure_env.sh — NE PAS VERSIONNER
PG_ML_PASSWORD=$(lire postgres_ml_service)
PG_ETL_PASSWORD=$(lire postgres_pipeline)
S3_ML_ACCESS=$(lire s3_ml_access)
S3_ML_SECRET=$(lire s3_ml_secret)
API_KEY=${API_KEY}
API_KEY_SHA256=${API_KEY_SHA256}
ENV
echo "Fichier .env écrit (droits 600). Clé d'API : ./scripts/api_key.sh"
