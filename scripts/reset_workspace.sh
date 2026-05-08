#!/usr/bin/env bash
# Reset workspace/initial_program.cpp to the canonical Leviathan baseline.
# Use between agent iterations or after a botched edit.
set -e
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$PKG/scripts/extras/canonical_baseline.cpp"
DST="$PKG/workspace/initial_program.cpp"
if [ ! -f "$SRC" ]; then
    echo "ERROR: missing $SRC" >&2
    exit 1
fi
cp "$SRC" "$DST"
echo "Reset $DST → canonical Leviathan baseline ($(wc -l < "$DST") lines)."
