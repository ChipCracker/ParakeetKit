#!/bin/bash
# Prepares the consumer-facing release artifact for the REMOTE SPM binary target:
# zips Frameworks/Parakeet.xcframework and computes the SPM checksum. The static
# xcframework is tiny (~6 MB) and has no dSYMs, so there is nothing to strip.
#
# Usage:
#   bash scripts/package-xcframework.sh <version-tag>   # e.g. parakeet-1
set -euo pipefail

VERSION="${1:?usage: package-xcframework.sh <version-tag>  (e.g. parakeet-1)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/Frameworks/Parakeet.xcframework"
DIST="$ROOT/dist"

[ -d "$SRC" ] || { echo "ERROR: $SRC not found — run scripts/build-xcframework.sh first." >&2; exit 1; }

rm -rf "$DIST/Parakeet.xcframework.zip"
mkdir -p "$DIST"

echo "=== Zipping xcframework ==="
ditto -c -k --keepParent "$SRC" "$DIST/Parakeet.xcframework.zip"
CHECKSUM="$(swift package compute-checksum "$DIST/Parakeet.xcframework.zip")"
SIZE="$(du -h "$DIST/Parakeet.xcframework.zip" | awk '{print $1}')"

cat <<EOF

=== Done ===
Artifact : $DIST/Parakeet.xcframework.zip ($SIZE)

1) Publish (do NOT overwrite an existing tag — cut a new one):
     git tag $VERSION && git push --tags
     gh release create $VERSION "$DIST/Parakeet.xcframework.zip" \\
        --title "parakeet.cpp xcframework ($VERSION)" --notes "static-lib xcframework for ParakeetKit"

2) Paste into Package.swift:
     let remoteURL = "https://github.com/ChipCracker/ParakeetKit/releases/download/$VERSION/Parakeet.xcframework.zip"
     let remoteChecksum = "$CHECKSUM"
EOF
