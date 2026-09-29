#!/bin/bash
# Single-instance wrapper around run_benchmark.sh.
#
# The nightly launchd trigger and a hand-started run are different processes, so
# launchd's own "don't start a job that is already running" does not protect
# against the pair of them. Two benchmark processes appending to one predictions
# file would produce duplicate rows -- which the reader now refuses, so the
# damage would be a wasted night rather than a wrong table, but a wasted night
# is worth avoiding.
#
# mkdir is the lock: it is atomic on every filesystem, including the exFAT
# volume the repository currently lives on, which is not true of flock.
set -u
LOCK=/tmp/sleepstatelab-benchmark.lock

if ! mkdir "$LOCK" 2>/dev/null; then
  owner=$(cat "$LOCK/pid" 2>/dev/null || echo unknown)
  if [ "$owner" != unknown ] && kill -0 "$owner" 2>/dev/null; then
    echo "$(date): benchmark already running as pid $owner; leaving it alone"
    exit 0
  fi
  echo "$(date): found a stale lock from pid $owner; taking it over"
  rm -rf "$LOCK" && mkdir "$LOCK" || exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

exec /Volumes/maxone/sleep-state-lab/scripts/run_benchmark.sh
