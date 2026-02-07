#!/bin/bash
# Run arena troop detection test (activates venv automatically)
cd "$(dirname "$0")"
source .venv/bin/activate
exec python test_arena_tracking.py "$@"
