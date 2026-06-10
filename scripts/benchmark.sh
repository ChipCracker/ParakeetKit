#!/usr/bin/env bash
# Runs the full benchmark suite and collects results under benchmarks/results/.
#
#   1. Pipeline benchmark (macOS, deterministic, no binary): cost profile of
#      the StreamingSession state machine (calls / transcribed seconds).
#   2. Engine benchmarks (real inference): WER/RTF for single-shot, long-audio
#      and the E2E streaming pipeline.
#      - simulator (default): CPU path, model read from the host filesystem.
#      - device: Metal path (real flash-attention numbers). The test runner
#        downloads the q4_K model (~466 MB) into its app container on first
#        run and caches it; results are recovered from the xcodebuild log
#        ([bench-json] markers) because devices can't write to host paths.
#
# Usage:
#   bash scripts/benchmark.sh                          # iOS simulator
#   PARAKEET_BENCH_DEST=device bash scripts/benchmark.sh   # connected device
#
# Env overrides:
#   PARAKEET_BENCH_DEST       simulator (default) | device
#   PARAKEET_BENCH_MODEL      GGUF path for simulator runs (default: parakeet-ios vendor model)
#   PARAKEET_BENCH_SIM        simulator name (default: first available iPhone)
#   PARAKEET_BENCH_DEVICE_ID  device UDID (default: first connected non-Mac device via xctrace)
#   PARAKEET_BENCH_TEAM       DEVELOPMENT_TEAM for device code signing
#                             (default: first team from Xcode's IDEProvisioningTeams)
#   PARAKEET_BENCH_LABEL      result-directory label (default: git short SHA)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${PARAKEET_BENCH_DEST:-simulator}"
LABEL="${PARAKEET_BENCH_LABEL:-$(git -C "$ROOT" rev-parse --short HEAD)}"
STAMP="$(date +%Y%m%d-%H%M%S)-$LABEL"
OUT="$ROOT/benchmarks/results/$STAMP"
mkdir -p "$OUT"

echo "=== 1/2 Pipeline benchmark (macOS, swift test) ==="
( cd "$ROOT" && PARAKEET_BENCH_OUT="$OUT" swift test --filter StreamingPipelineBenchmarkTests )

LOG="$OUT/xcodebuild.log"

if [ "$DEST" = "device" ]; then
    # First connected non-Mac device from xctrace's "== Devices ==" block;
    # the UDID is the last parenthesized token of the line.
    DEVICE_ID="${PARAKEET_BENCH_DEVICE_ID:-$(xcrun xctrace list devices 2>/dev/null \
        | sed -n '/^== Devices ==$/,/^== Devices Offline ==$/p' \
        | grep -v 'MacBook\|Mac mini\|Mac Studio\|iMac\|Mac Pro\|^==\|^$' \
        | head -1 | sed -E 's/.*\(([0-9A-Fa-f-]+)\)$/\1/')}"
    [ -n "$DEVICE_ID" ] || { echo "ERROR: no connected device found (xcrun xctrace list devices)" >&2; exit 1; }

    TEAM="${PARAKEET_BENCH_TEAM:-$(defaults read com.apple.dt.Xcode IDEProvisioningTeams 2>/dev/null \
        | grep -o 'teamID = [A-Z0-9]*' | head -1 | awk '{print $3}')}"
    [ -n "$TEAM" ] || { echo "ERROR: no signing team — set PARAKEET_BENCH_TEAM=<TEAMID>" >&2; exit 1; }

    echo "=== 2/2 Engine benchmarks (device: $DEVICE_ID, team: $TEAM, Metal) ==="
    echo "    (first run downloads the q4_K model ~466 MB onto the device)"
    # Devices can't run SPM test bundles tool-hosted → generate the host-app
    # project (project.yml) and run the hosted DeviceBenchmarks target.
    ( cd "$ROOT" && xcodegen generate --quiet )
    ( cd "$ROOT" && \
      TEST_RUNNER_PARAKEET_BENCH_DOWNLOAD=1 \
      xcodebuild test \
        -project ParakeetBench.xcodeproj \
        -scheme ParakeetBench \
        -destination "platform=iOS,id=$DEVICE_ID" \
        -derivedDataPath "$ROOT/.build/benchmark-dd-device" \
        -allowProvisioningUpdates \
        DEVELOPMENT_TEAM="$TEAM" \
        CODE_SIGN_STYLE=Automatic \
        2>&1 | tee "$LOG" | grep -E "Test Case|Test Suite|\[bench\]|error:|failed" || true )
else
    MODEL="${PARAKEET_BENCH_MODEL:-$ROOT/../parakeet-ios/vendor/models/parakeet-tdt-0.6b-v3-q4_k.gguf}"
    [ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL (set PARAKEET_BENCH_MODEL)" >&2; exit 1; }

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
        2>&1 | tee "$LOG" | grep -E "Test Case|Test Suite|\[bench\]|error:|failed" || true )
fi

# Recover result JSONs from the log's [bench-json] markers (the only channel
# for device runs; a no-op when the test already wrote the file directly).
grep -o '\[bench-json\] .*' "$LOG" 2>/dev/null | while read -r _ name json; do
    [ -f "$OUT/$name.json" ] || printf '%s\n' "$json" > "$OUT/$name.json"
done

echo "=== Results: $OUT ==="
ls -la "$OUT"
