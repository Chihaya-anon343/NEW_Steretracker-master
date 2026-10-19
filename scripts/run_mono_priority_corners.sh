#!/usr/bin/env bash
set -euo pipefail
out="${1:-output/mono_priority_20261019}"
g++ -std=c++17 -O2 -Iinclude -I/usr/include/eigen3 -I/usr/include/opencv4 \
 scripts/evaluate_mono_priority.cpp src/feature/BinaryCornerExtractor.cpp \
 src/matching/MonoCornerIdentity.cpp src/utils/PoseUtils.cpp \
 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d \
 -o "$out/evaluate_corners"
"$out/evaluate_corners" "$out/before/config/tracker_config.json" "$out/corner_jobs.txt" "$out/corners.csv" 1 > "$out/corners.log" 2>&1
