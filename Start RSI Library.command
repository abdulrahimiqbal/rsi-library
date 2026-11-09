#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec python3 -m rsi_library serve --port 0 --open
