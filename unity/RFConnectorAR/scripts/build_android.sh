#!/usr/bin/env bash
set -euo pipefail

# CLI driver for the Android build. Invokes Unity headlessly via the
# BuildScript.BuildAndroid entry point. Output lands at
# <project>/Builds/Android/RFConnectorAR.apk.
#
# Requirements on this machine:
#   - Unity 6.0 LTS installed with Android Build Support, Android SDK + NDK,
#     OpenJDK modules (install via Unity Hub once).
#   - UNITY_EDITOR env var pointing at Unity.exe, or edit the default below.

UNITY="${UNITY_EDITOR:-/e/unity/6000.0.73f1/Editor/Unity.exe}"
PROJECT="$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null || cd "$(dirname "$0")/.." && pwd)"
LOG="${PROJECT}/Builds/android_build.log"

if [[ ! -x "$UNITY" ]]; then
    echo "error: Unity Editor not found at $UNITY (set UNITY_EDITOR env var)" >&2
    exit 2
fi

mkdir -p "$(dirname "$LOG")"
echo "Building Android APK from $PROJECT"
echo "Log: $LOG"

"$UNITY" \
    -batchmode \
    -projectPath "$PROJECT" \
    -buildTarget Android \
    -executeMethod RFConnectorAR.EditorTools.BuildScript.BuildAndroid \
    -quit -nographics \
    -logFile "$LOG"

APK="$PROJECT/Builds/Android/RFConnectorAR.apk"
if [[ -f "$APK" ]]; then
    SIZE=$(stat -c%s "$APK" 2>/dev/null || stat -f%z "$APK")
    echo "success: $APK ($SIZE bytes)"
else
    echo "error: APK not produced; see $LOG" >&2
    exit 1
fi
