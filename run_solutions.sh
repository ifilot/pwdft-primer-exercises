#!/usr/bin/env bash
# Run the reference solutions and report checks that fail.
#
# Usage (from the repository root):
#   bash run_solutions.sh            all solutions (about 15 minutes; run locally)
#   bash run_solutions.sh --quick    skip the long calculations (about 1 minute;
#                                    this is what the GitHub workflow runs)
#   PYTHON=/path/to/python bash run_solutions.sh
set -u
export MPLBACKEND=Agg
export PWLIB_THREADS=${PWLIB_THREADS:-4}
PYTHON=${PYTHON:-python3}

# solutions that take more than about half a minute (complete SCF runs)
LONG="ex6_1_first_calculation.py ex6_2_cell_size.py chal5_4_scf.py
      chal6_3_allelectron_co.py chal7_3_co_pseudo.py"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

cd "$(dirname "$0")/solutions"
status=0
for f in ex*.py chal*.py; do
    if [ "$QUICK" -eq 1 ] && grep -qw "$f" <<< "$LONG"; then
        printf '%-28s skipped (long; run locally)\n' "$f"
        continue
    fi
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
