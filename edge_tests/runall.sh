#!/bin/bash
# Run scheduler + validator on every edge test. Print one PASS/FAIL line per
# test plus a summary at the end.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RED='\033[0;31m'
GREEN='\033[0;32m'
RESET='\033[0m'

cd "$ROOT"

# Make sure scheduler is built (no-op for python).
./build.sh >/dev/null 2>&1 || true

pass=0
fail=0
failed_tests=()

for tdir in edge_tests/*/; do
    [ -f "$tdir/input.json" ] || continue
    name=$(basename "$tdir")

    # 1. run scheduler
    ./run.sh "$tdir/input.json" "$tdir/simple.json" "$tdir/pip.json" >/dev/null 2>"/tmp/edge_${name}.err"
    rc=$?
    if [ $rc -ne 0 ]; then
        printf "${RED}FAIL${RESET} %-32s scheduler crashed (see /tmp/edge_${name}.err)\n" "$name"
        fail=$((fail+1))
        failed_tests+=("$name")
        continue
    fi

    # 2. run validator
    out=$(python edge_tests/validate.py "$tdir/input.json" "$tdir/simple.json" "$tdir/pip.json" 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        printf "${GREEN}PASS${RESET} %-32s %s\n" "$name" "$(head -n1 < "$tdir/desc.txt")"
        pass=$((pass+1))
    else
        printf "${RED}FAIL${RESET} %-32s\n" "$name"
        printf "%s\n" "$out" | sed 's/^/       /'
        fail=$((fail+1))
        failed_tests+=("$name")
    fi
done

echo
echo "Summary: $pass passed, $fail failed"
if [ $fail -gt 0 ]; then
    echo "Failed: ${failed_tests[*]}"
    exit 1
fi
