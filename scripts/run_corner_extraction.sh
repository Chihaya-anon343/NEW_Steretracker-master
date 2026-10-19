#!/usr/bin/env bash
set -euo pipefail
out="${1:-output/corner_quality_20261016}"
g++ -std=c++17 -O2 -Iinclude -I/usr/include/eigen3 -I/usr/include/opencv4 \
 scripts/evaluate_corner_extraction.cpp src/feature/BinaryCornerExtractor.cpp \
 src/feature/TinyTargetExtractor.cpp src/utils/PoseUtils.cpp \
 src/matching/MonoCornerIdentity.cpp \
 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d \
 -o "$out/evaluate_corner_extraction"
if [ "${2:-}" != --build-only ]; then
 "$out/evaluate_corner_extraction" "$out/config_snapshot.json" "$out/jobs.txt" "$out/detections.csv" > "$out/extraction.log" 2>&1
fi
