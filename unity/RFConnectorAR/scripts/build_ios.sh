#!/usr/bin/env bash
set -euo pipefail

# CLI driver for the iOS build. Invokes Unity headlessly via the
# BuildScript.BuildIOS entry point. Output is an Xcode project at
# <project>/Builds/iOS/ — open in Xcode, set signing team, select a real
# device, press Run.
#
# Requirements:
#   - Unity 6.0 LTS installed with iOS Build Support module.
#   - macOS required for the subsequent Xcode → device step; the Unity side
#     of this script runs on any OS.

UNITY="${UNITY_EDITOR:-/e/unity/6000.0.73f1/Editor/Unity.exe}"
PROJECT="$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null || cd "$(dirname "$0")/.." && pwd)"
LOG="${PROJECT}/Builds/ios_build.log"

if [[ ! -x "$UNITY" ]]; then
    echo "error: Unity Editor not found at $UNITY (set UNITY_EDITOR env var)" >&2
    exit 2
fi

mkdir -p "$(dirname "$LOG")"
echo "Building iOS Xcode project from $PROJECT"
echo "Log: $LOG"

"$UNITY" \
    -batchmode \
    -projectPath "$PROJECT" \
    -buildTarget iOS \
    -executeMethod RFConnectorAR.EditorTools.BuildScript.BuildIOS \
    -quit -nographics \
    -logFile "$LOG"

XCODE_DIR="$PROJECT/Builds/iOS"
if [[ -f "$XCODE_DIR/Unity-iPhone.xcodeproj/project.pbxproj" ]]; then
    echo "success: Xcode project at $XCODE_DIR/Unity-iPhone.xcodeproj"
    echo "Next: on macOS, open Unity-iPhone.xcodeproj, set your Team, select a real device, Run."
else
    echo "error: Xcode project not produced; see $LOG" >&2
    exit 1
fi
