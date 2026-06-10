#!/usr/bin/env bash
# Runs the full benchmark suite and collects results under benchmarks/results/.
#
#   1. Pipeline benchmark (macOS, deterministic, no binary): cost profile of
#      the StreamingSession state machine (calls / transcribed seconds).
#   2. Engine benchmarks (iOS simulator, real inference, CPU path): WER/RTF
#      for single-shot, long-audio and the E2E streaming pipeline.
#
# Usage:
#   bash scripts/benchmark.sh
#
# Env overrides:
#   PARAKEET_BENCH_MODEL  GGUF path (default: parakeet-ios vendor model)
#   PARAKEET_BENCH_SIM    simulator name (default: first available iPhone)
#   PARAKEET_BENCH_LABEL  result-directory label (default: git short SHA)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${PARAKEET_BENCH_MODEL:-$ROOT/../parakeet-ios/vendor/models/parakeet-tdt-0.6b-v3-q4_k.gguf}"
LABEL="${PARAKEET_BENCH_LABEL:-$(git -C "$ROOT" rev-parse --short HEAD)}"
STAMP="$(date +%Y%m%d-%H%M%S)-$LABEL"
OUT="$ROOT/benchmarks/results/$STAMP"
mkdir -p "$OUT"

[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL (set PARAKEET_BENCH_MODEL)" >&2; exit 1; }

echo "=== 1/2 Pipeline benchmark (macOS, swift test) ==="
( cd "$ROOT" && PARAKEET_BENCH_OUT="$OUT" swift test --filter StreamingPipelineBenchmarkTests )

SIM="${PARAKEET_BENCH_SIM:-$(xcrun simctl list devices available -j | python3 -c '
import json,sys
for devs in json.load(sys.stdin)["devices"].values():
    for d in devs:
        if d.get("isAvailable") and d["name"].startswith("iPhone"):
            print(d["name"]); raise SystemExit
')}"
[ -n "$SIM" ] || { echo "ERROR: no available iPhone simulator found" >&2; exit 1; }

echo "=== 2/2 Engine benchmarks (simulator: $SIM, model: $(basename "$MODEL")) ==="
# TEST_RUNNER_-prefixed vars are stripped by xcodebuild and handed to the
# test process — plain env vars do NOT reach simulator tests.
( cd "$ROOT" && \
  TEST_RUNNER_PARAKEET_BENCH_MODEL="$MODEL" \
  TEST_RUNNER_PARAKEET_BENCH_OUT="$OUT" \
  xcodebuild test \
    -scheme ParakeetKit-Package \
    -destination "platform=iOS Simulator,name=$SIM" \
    -only-testing:ParakeetKitBenchmarks \
    -derivedDataPath "$ROOT/.build/benchmark-dd" \
    2>&1 | grep -E "Test Case|Test Suite|\[bench\]|error:|failed" || true )

echo "=== Results: $OUT ==="
ls -la "$OUT"
