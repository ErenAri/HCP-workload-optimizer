#!/usr/bin/env bash
# =============================================================================
# bootstrap-secrets.sh — Create local secrets directory with a random API key
# =============================================================================
# Usage:
#   chmod +x scripts/bootstrap-secrets.sh
#   ./scripts/bootstrap-secrets.sh
#
# This script creates the secrets/ directory and generates a random API key
# for local development. For production, replace with your own keys.
# =============================================================================

set -euo pipefail

SECRETS_DIR="$(cd "$(dirname "$0")/.." && pwd)/secrets"
KEYS_FILE="${SECRETS_DIR}/api_keys.txt"

if [ -f "$KEYS_FILE" ]; then
    echo "[INFO] $KEYS_FILE already exists. Skipping."
    echo "[INFO] To regenerate, delete it first: rm $KEYS_FILE"
    exit 0
fi

mkdir -p "$SECRETS_DIR"

# Generate a 32-character hex API key
API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null \
    || openssl rand -hex 32 2>/dev/null \
    || head -c 64 /dev/urandom | xxd -p | head -c 64)

echo "$API_KEY" > "$KEYS_FILE"
chmod 600 "$KEYS_FILE"

echo "[OK] Created $KEYS_FILE"
echo "[OK] Your development API key: $API_KEY"
echo ""
echo "Set this header in your requests:"
echo "  X-API-Key: $API_KEY"
echo ""
echo "Or disable auth entirely by not setting HPCOPT_API_KEYS_FILE."
