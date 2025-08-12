#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# manage_env.sh — Manage .env configuration: generate/rotate secrets, lock down signup, set retention.
# Backward-compatible CLI:
#   ./scripts/manage_env.sh generate-secrets [--rotate]
#   ./scripts/manage_env.sh lockdown-signup
#   ./scripts/manage_env.sh set-retention [DAYS]
#
# Global options (before the subcommand):
#   -f, --env-file PATH   Use custom env file (default: .env)
#       --create          Auto-create env file (from .env.example if present)
#   -q, --quiet           Suppress non-error logs
#   -h, --help            Show help

ENV_FILE=".env"
AUTO_CREATE=false
QUIET=false

color() { local c=$1; shift; [[ "$QUIET" == "true" && "$c" != "31" ]] && return 0; printf "\033[%sm%s\033[0m\n" "$c" "$*"; }
info() { color 36 "[INFO] $*"; }
ok()   { color 32 "[OK]   $*"; }
warn() { color 33 "[WARN] $*"; }
err()  { color 31 "[ERR]  $*"; }

on_error() {
  local exit_code=$?
  local line_no=${1:-}
  err "Failed at line ${line_no}. Exit code: ${exit_code}"
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

check_dependencies() {
  local missing=()
  for dep in openssl sed grep; do
    if ! command -v "$dep" >/dev/null 2>&1; then
      missing+=("$dep")
    fi
  done
  if ((${#missing[@]} > 0)); then
    err "Missing dependencies: ${missing[*]}"
    err "Please install required tools and re-run."
    exit 1
  fi
}

# Cross-platform in-place sed (BSD/macOS vs GNU/Linux)
sed_inplace() {
  local expr=$1
  local file=$2
  if sed --version >/dev/null 2>&1; then
    sed -i -E "$expr" "$file"
  else
    sed -i '' -E "$expr" "$file"
  fi
}

require_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    return 0
  fi
  if [[ "$AUTO_CREATE" == "true" ]]; then
    if [[ -f ".env.example" ]]; then
      cp ".env.example" "$ENV_FILE"
      ok "Created $ENV_FILE from .env.example"
    else
      : >"$ENV_FILE"
      ok "Created empty $ENV_FILE"
    fi
  else
    err "Missing $ENV_FILE. Provide it or pass --create (optionally add -f PATH)."
    exit 1
  fi
}

is_hex_64() {
  [[ "$1" =~ ^[0-9a-fA-F]{64}$ ]]
}

ensure_hex_64() {
  local key=$1
  local rotate=${2:-false}
  local current
  current=$(grep -E "^${key}=" "$ENV_FILE" | head -n1 | cut -d= -f2- | tr -d '\r' || true)
  if [[ "$rotate" == "true" || -z "$current" ]]; then
    :
  elif ! is_hex_64 "$current"; then
    :
  else
    info "${key} already set (64-hex), skipping"
    return 0
  fi

  local value
  value=$(openssl rand -hex 32)
  if grep -qE "^${key}=" "$ENV_FILE"; then
    sed_inplace "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
  else
    printf "\n%s=%s\n" "$key" "$value" >> "$ENV_FILE"
  fi
  ok "Set ${key} to new 64-hex value"
}

set_kv() {
  local key=$1
  local val=$2
  if grep -qE "^${key}=" "$ENV_FILE"; then
    sed_inplace "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
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
Usage: $0 [GLOBAL_OPTIONS] <command> [ARGS]

Commands:
  generate-secrets [--rotate]   Generate required 64-hex secrets (rotate forces regeneration)
  lockdown-signup               Set AUTH_DISABLE_SIGNUP=true
  set-retention [DAYS]          Set LANGFUSE_RETENTION_DAYS (default: 90)

Global options (place before command):
  -f, --env-file PATH           Path to .env file (default: ./.env)
      --create                  Auto-create env file (from .env.example if present)
  -q, --quiet                   Suppress non-error logs
  -h, --help                    Show this help

Examples:
  $0 generate-secrets
  $0 generate-secrets --rotate
  $0 --env-file .env.local --create set-retention 120
  $0 -q lockdown-signup
EOF
}

main() {
  check_dependencies

  # Parse global options until the subcommand
  local cmd=""
  while (($# > 0)); do
    case "$1" in
      -f|--env-file)
        [[ $# -lt 2 ]] && { err "--env-file requires a path"; exit 1; }
        ENV_FILE="$2"; shift 2 ;;
      --create)
        AUTO_CREATE=true; shift ;;
      -q|--quiet)
        QUIET=true; shift ;;
      -h|--help)
        usage; exit 0 ;;
      --)
        shift; break ;;
      -*)
        err "Unknown option: $1"; usage; exit 1 ;;
      *)
        cmd="$1"; shift; break ;;
    esac
  done

  case "${cmd:-}" in
    generate-secrets)
      generate_secrets "${1:-}" ;;
    lockdown-signup)
      lockdown_signup ;;
    set-retention)
      set_retention "${1:-90}" ;;
    ""|-h|--help)
      usage ;;
    *)
      err "Unknown command: ${cmd}"; usage; exit 1 ;;
  esac
}

main "$@"


