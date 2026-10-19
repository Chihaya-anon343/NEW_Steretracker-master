#!/usr/bin/env bash
set -euo pipefail
out="${1:-output/mono_normalized_20261016}"
g++ -std=c++17 -O2 -Iinclude -I/usr/include/eigen3 -I/usr/include/opencv4 \
 scripts/evaluate_mono_normalized.cpp src/feature/BinaryCornerExtractor.cpp \
 src/matching/MonoCornerIdentity.cpp src/utils/PoseUtils.cpp \
 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d \
 -o "$out/evaluate_mono_normalized"
if [ "${3:-}" = --capture ]; then
 "$out/evaluate_mono_normalized" "$out" "$out/corners/current/jobs.txt" "${2:-0}" --capture > "$out/corner_extraction.log" 2>&1
else
 "$out/evaluate_mono_normalized" "$out" "$out/corners/current/jobs.txt" "${2:-0}" > "$out/corner_extraction.log" 2>&1
fi
