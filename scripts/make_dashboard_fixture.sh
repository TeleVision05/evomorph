#!/bin/sh
# Regenerate the dashboard test fixture with the real evolve command.
#
#   sh scripts/make_dashboard_fixture.sh
#
# Writes tests/fixtures/dashboard/runs/{ga_tiny,me_tiny}. Runs are never
# overwritten by evolve, so delete the old folders first if you regenerate.
# Uses 2 workers (the engine caps every pool at 6).
set -eu
PY=${PY:-.venv/bin/python}
CFG=tests/fixtures/dashboard/tiny.yaml
OUT=tests/fixtures/dashboard/runs
"$PY" -m evomorph evolve --config "$CFG" --results "$OUT" --name ga_tiny --workers 2
"$PY" -m evomorph evolve --config "$CFG" --results "$OUT" --name me_tiny --workers 2 \
    --algorithm map_elites
