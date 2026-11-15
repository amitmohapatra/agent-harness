#!/usr/bin/env bash
# ANTHROPIC_KEY_FILE=<file> sweep_final.sh: the sweep on the final code at the windows 8,000 did not rerun.
CASES=carry-03,carry-04,planning-02,planning-03,long-multi-01,long-multi-05,long-chain-01
for VAR in todos todos-unthreaded; do
  for WIN in 16000 32000 128000 -; do
    /var/tmp/cons/decide2/scripts/sweep.sh "fix-$VAR-w${WIN/-/none}" "$VAR" "$WIN" "$CASES" 0.15
  done
done
echo done > /var/tmp/cons/decide2/out/FINAL_DONE
