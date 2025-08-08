#!/usr/bin/env bash
set -euo pipefail

# Script: manage_env.sh
# Purpose: Generate/rotate secrets, lock down Langfuse signup, and set retention.
# Usage:
#   ./scripts/manage_env.sh generate-secrets [--rotate]
#   ./scripts/manage_env.sh lockdown-signup
#   ./scripts/manage_env.sh set-retention [DAYS]
#
# Notes:
# - Idempotent: generates values only if missing unless --rotate is provided.
# - Works on the project .env file in repo root.

ENV_FILE=".env"

color() { local c=$1; shift; printf "\033[%sm%s\033[0m\n" "$c" "$*"; }
info() { color 36 "[INFO] $*"; }
ok()   { color 32 "[OK]   $*"; }
warn() { color 33 "[WARN] $*"; }
err()  { color 31 "[ERR]  $*"; }

require_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    err "Missing $ENV_FILE. Copy .env.example to .env first."
    exit 1
  fi
}

ensure_hex_64() {
  local key=$1
  local rotate=${2:-false}
  local current
  current=$(grep -E "^${key}=" "$ENV_FILE" | head -n1 | cut -d= -f2- || true)
  if [[ -z "$current" || ${#current} -ne 64 || "$rotate" == "true" ]]; then
    local value
    value=$(openssl rand -hex 32)
    if grep -qE "^${key}=" "$ENV_FILE"; then
      sed -i '' -E "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
      printf "\n%s=%s\n" "$key" "$value" >> "$ENV_FILE"
    fi
    ok "Set ${key} to new 64-hex value"
  else
    info "${key} already set (64-hex), skipping"
  fi
}

set_kv() {
  local key=$1
  local val=$2
  if grep -qE "^${key}=" "$ENV_FILE"; then
    sed -i '' -E "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    printf "\n%s=%s\n" "$key" "$val" >> "$ENV_FILE"
  fi
  ok "Updated ${key}=${val}"
}

generate_secrets() {
  require_env_file
  local rotate=false
  if [[ "${1:-}" == "--rotate" ]]; then rotate=true; fi
  info "Generating required secrets (rotate=${rotate})"
  ensure_hex_64 NEXTAUTH_SECRET "$rotate"
  ensure_hex_64 SALT "$rotate"
  ensure_hex_64 ENCRYPTION_KEY "$rotate"
  ensure_hex_64 CLICKHOUSE_PASSWORD "$rotate"
}

lockdown_signup() {
  require_env_file
  set_kv AUTH_DISABLE_SIGNUP true
}

set_retention() {
  require_env_file
  local days=${1:-90}
  if ! [[ "$days" =~ ^[0-9]+$ ]]; then
    err "Retention days must be an integer"
    exit 1
  fi
  set_kv LANGFUSE_RETENTION_DAYS "$days"
}

usage() {
  cat <<EOF
Usage:
  $0 generate-secrets [--rotate]
  $0 lockdown-signup
  $0 set-retention [DAYS]
EOF
}

cmd=${1:-}
case "$cmd" in
  generate-secrets)
    shift || true
    generate_secrets "$@"
    ;;
  lockdown-signup)
    lockdown_signup
    ;;
  set-retention)
    shift || true
    set_retention "${1:-90}"
    ;;
  *)
    usage
    exit 1
    ;;
esac


