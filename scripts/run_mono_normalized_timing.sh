#!/usr/bin/env bash
set -euo pipefail
out="${1:-output/mono_normalized_20261016}"
g++ -std=c++17 -O2 -Iinclude -I/usr/include/eigen3 -I/usr/include/opencv4 \
 scripts/benchmark_mono_normalized.cpp src/feature/BinaryCornerExtractor.cpp \
 src/matching/MonoCornerIdentity.cpp src/utils/PoseUtils.cpp \
 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d \
 -o "$out/benchmark_mono_normalized"
"$out/benchmark_mono_normalized" "$out/config_before.json" "$out/timing_jobs.txt" "$out/paired_timing.csv" > "$out/timing.log" 2>&1
