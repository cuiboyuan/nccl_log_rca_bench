#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

rm -rf "$ROOT/labels" "$ROOT/output" "$ROOT/phase1_runs" "$ROOT/phase2_runs"
