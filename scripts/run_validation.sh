#!/usr/bin/env bash
# Runs all validation scripts, saves one log per script, and prints a summary. This includes the non standard plot for all three parts (Sequential/Parallel FedAvg and Differential Privacy)
# Usage:  
# scripts/run_validation.sh (full run)
# QUICK=1 scripts/run_validation.sh (shortened experiments)
# scripts/run_validation.sh fedavg dp (only the named parts: fedavg parallel dp plot)
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAL_DIR="$ROOT/src/validation" # The location of validation scripts
OUT_DIR="$ROOT/results/validation" # Storing log files
PY="${PYTHON:-python3}"
mkdir -p "$OUT_DIR"

declare -A SCRIPT=( [fedavg]=val_fed_avg.py [parallel]=val_parallel.py [dp]=val_differential_privacy.py [plot]=non_standard_plot.py )
ORDER=(fedavg parallel dp plot)
PARTS=("$@")
[ ${#PARTS[@]} -eq 0 ] && PARTS=("${ORDER[@]}")

cd "$VAL_DIR" || exit 2
export PYTHONUNBUFFERED=1
declare -A STATUS

for part in "${PARTS[@]}"; do
  file="${SCRIPT[$part]:-}"
  if [ -z "$file" ]; then echo "unknown part: $part (use: ${ORDER[*]})"; exit 2; fi
  log="$OUT_DIR/${file%.py}.log"
  echo "=== $part: $file  (log: ${log#$ROOT/}) ==="
  start=$(date +%s)
  "$PY" "$file" 2>&1 | tee "$log"
  STATUS[$part]=${PIPESTATUS[0]}
  echo "--- $part finished in $(( $(date +%s) - start ))s, exit code ${STATUS[$part]}"
  echo
done

# Summary: exit codes plus the per tag counts each script prints
SUMMARY="$OUT_DIR/summary.txt"
{
  echo "Validation summary ($(date))  QUICK=${QUICK:-0}"
  for part in "${PARTS[@]}"; do
    file="${SCRIPT[$part]}"; log="$OUT_DIR/${file%.py}.log"
    verdict=$([ "${STATUS[$part]}" -eq 0 ] && echo OK || echo "FAILED (exit ${STATUS[$part]})")
    echo "$file: $verdict"
    grep -E "^(EDGE|VALID) *:|checks passed" "$log" | sed 's/^/    /'
    grep -E "^\[FAIL\]" "$log" | sed 's/^/    /'
  done
} | tee "$SUMMARY"

fail=0
for part in "${PARTS[@]}"; do [ "${STATUS[$part]}" -ne 0 ] && fail=1; done
exit $fail