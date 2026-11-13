#!/bin/sh
# Terrain validation: both reference creatures on every terrain suite, spawn seeds 0-2.
#
# Usage (from the repo root): sh scripts/validate_terrains.sh [OUT_DIR]
#
# Writes <ref>_<suite>.csv/.json into OUT_DIR (default docs/validation/terrains),
# the files docs/TERRAINS.md quotes. The robustness command never overwrites, so
# re-running into a non-empty folder adds "-2" copies: use a fresh folder to compare.
# Uses 4 worker processes (the house cap is 6).
set -e
OUT="${1:-docs/validation/terrains}"
for ref in quadruped hexapod; do
  for suite in flat slope friction uneven uneven-seeds train test; do
    .venv/bin/python -m evomorph robustness --reference "$ref" --suite "$suite" \
      --seeds 3 --workers 4 --out "$OUT"
  done
done
