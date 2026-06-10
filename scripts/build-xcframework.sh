#!/bin/bash
# Produces Frameworks/Parakeet.xcframework for the local SPM binary target by
# copying the artifact built in the parakeet-ios repo and injecting a
# module.modulemap (so SPM can expose it as the Clang module `CParakeet`).
#
# The static-library xcframework is built (from the CrispStrobe/CrispASR sources)
# by parakeet-ios/scripts/build-parakeet-xcframework.sh. Pass --rebuild to run
# that first.
#
# Usage:
#   bash scripts/build-xcframework.sh            # copy + inject modulemap
#   bash scripts/build-xcframework.sh --rebuild  # rebuild upstream first, then copy
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARAKEET_IOS="$ROOT/../parakeet-ios"
SRC="$PARAKEET_IOS/vendor/Parakeet.xcframework"
OUT="$ROOT/Frameworks/Parakeet.xcframework"

if [ "${1:-}" = "--rebuild" ]; then
  echo "=== Rebuilding upstream Parakeet.xcframework ==="
  ( cd "$PARAKEET_IOS" && bash scripts/build-parakeet-xcframework.sh )
fi

[ -d "$SRC" ] || { echo "ERROR: $SRC not found (build it in parakeet-ios first)." >&2; exit 1; }

echo "=== Copying $SRC → $OUT ==="
mkdir -p "$ROOT/Frameworks"
rm -rf "$OUT"
cp -R "$SRC" "$OUT"

# Inject the Clang module map into every slice's Headers/ (idempotent).
echo "=== Injecting module.modulemap ==="
for headers in "$OUT"/*/Headers; do
  [ -d "$headers" ] || continue
  cat > "$headers/module.modulemap" <<'EOF'
module CParakeet {
    header "parakeet.h"
    header "firered_vad.h"
    header "titanet.h"
    header "pyannote_seg.h"
    export *
}
EOF
  echo "  $headers/module.modulemap"
done

echo "=== Done ==="
echo "Build for iOS:  xcodebuild -scheme ParakeetKit -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build"
echo "Pure tests:     swift test"
