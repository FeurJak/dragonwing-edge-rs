#!/usr/bin/env bash
#
# deploy.sh — build dragonwing-probe and run it on the Arduino UNO Q.
#
# Transport: adb (the Arduino UNO Q exposes itself as an adb device over USB,
# at /Users/tarasworonjanski/Library/Arduino15/packages/arduino/tools/adb).
#
# Usage:
#   scripts/deploy.sh                   # build, push, run, print JSON
#   scripts/deploy.sh --save NAME       # also save report to artifacts/probes/NAME.json
#   scripts/deploy.sh --target gnu      # use aarch64-unknown-linux-gnu (requires cross linker)
#   scripts/deploy.sh --target musl     # default; fully static, no external linker
#   scripts/deploy.sh --no-build        # skip cargo build, push existing binary
#
# This script intentionally avoids external deps (no ssh/scp config); adb is
# the supported channel for the Arduino UNO Q hardware.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-/Users/tarasworonjanski/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb}"
ADB_SERIAL="${ADB_SERIAL:-539985843}"
TARGET="musl"
SAVE_NAME=""
DO_BUILD=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --save)   SAVE_NAME="$2"; shift 2 ;;
        --no-build) DO_BUILD=0; shift ;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$TARGET" in
    musl) RUST_TARGET="aarch64-unknown-linux-musl" ;;
    gnu)  RUST_TARGET="aarch64-unknown-linux-gnu"  ;;
    *) echo "unknown --target: $TARGET (musl|gnu)" >&2; exit 2 ;;
esac

BIN_PATH="${REPO_ROOT}/target/${RUST_TARGET}/release/dragonwing-probe"
DEVICE_PATH="/tmp/dragonwing-probe"

if [[ "$DO_BUILD" -eq 1 ]]; then
    echo "==> cargo build -p dragonwing-probe --target ${RUST_TARGET} --release"
    (cd "$REPO_ROOT" && cargo build -p dragonwing-probe --target "$RUST_TARGET" --release)
fi

if [[ ! -x "$BIN_PATH" ]]; then
    echo "error: binary not found at $BIN_PATH" >&2
    exit 1
fi

if ! "$ADB" -s "$ADB_SERIAL" get-state >/dev/null 2>&1; then
    echo "error: adb device $ADB_SERIAL not connected" >&2
    "$ADB" devices >&2
    exit 1
fi

echo "==> pushing $(basename "$BIN_PATH") -> ${DEVICE_PATH}"
"$ADB" -s "$ADB_SERIAL" push "$BIN_PATH" "$DEVICE_PATH" >/dev/null
"$ADB" -s "$ADB_SERIAL" shell chmod +x "$DEVICE_PATH"

echo "==> running probe on device"
TMP_OUT="$(mktemp)"
trap 'rm -f "$TMP_OUT"' EXIT

"$ADB" -s "$ADB_SERIAL" shell "$DEVICE_PATH" > "$TMP_OUT"
RC=$?
if [[ $RC -ne 0 ]]; then
    echo "error: probe exited with code $RC" >&2
    cat "$TMP_OUT" >&2
    exit $RC
fi

# adb shell on some platforms inserts \r\n; strip CRs for clean JSON.
tr -d '\r' < "$TMP_OUT" > "${TMP_OUT}.clean"
mv "${TMP_OUT}.clean" "$TMP_OUT"

if [[ -n "$SAVE_NAME" ]]; then
    DEST_DIR="${REPO_ROOT}/artifacts/probes"
    mkdir -p "$DEST_DIR"
    DEST="${DEST_DIR}/${SAVE_NAME}.json"
    cp "$TMP_OUT" "$DEST"
    echo "==> saved to ${DEST}"
fi

cat "$TMP_OUT"
