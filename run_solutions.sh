#!/usr/bin/env bash
# Run all reference solutions and report checks that fail.
# Usage: ./run_solutions.sh          (from the repository root)
#        PYTHON=/path/to/python ./run_solutions.sh
set -u
export MPLBACKEND=Agg
export PWLIB_THREADS=${PWLIB_THREADS:-4}
PYTHON=${PYTHON:-python3}
cd "$(dirname "$0")/solutions"
status=0
for f in ex*.py chal*.py; do
    start=$(date +%s)
    out=$("$PYTHON" "$f" 2>&1)
    rc=$?
    npass=$(grep -c '^\[PASS\]' <<< "$out")
    nfail=$(grep -c '^\[FAIL\]' <<< "$out")
    printf '%-28s exit %d   PASS %3d   FAIL %d   %4ds\n' "$f" "$rc" "$npass" "$nfail" $(( $(date +%s) - start ))
    if [ "$rc" -ne 0 ] || [ "$nfail" -ne 0 ]; then
        status=1
        grep -E '^\[FAIL\]|Error' <<< "$out"
    fi
done
exit $status
