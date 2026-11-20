#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js 20+ is required. Install Node.js and retry." >&2
  exit 2
fi

NODE_MAJOR="$(node -p 'Number(process.versions.node.split(".")[0])')"
if [ "$NODE_MAJOR" -lt 20 ]; then
  echo "Node.js 20+ is required; found $(node --version)." >&2
  exit 2
fi

cd "$ROOT_DIR"
exec npx --yes @playwright/mcp@0.0.86 --config "$SCRIPT_DIR/mcp.config.json" "$@"
