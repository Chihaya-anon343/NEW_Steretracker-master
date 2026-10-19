#!/usr/bin/env bash
set -euo pipefail
out="output/mono_normalized_20261016/representatives"
g++ -std=c++17 -O2 -Iinclude -I/usr/include/eigen3 -I/usr/include/opencv4 \
 scripts/benchmark_mono_resources.cpp src/feature/BinaryCornerExtractor.cpp \
 src/matching/MonoCornerIdentity.cpp src/utils/PoseUtils.cpp \
 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d \
 -o "$out/benchmark_mono_resources"
for mode in current normalized refined; do
 "$out/benchmark_mono_resources" output/mono_normalized_20261016/config_before.json "$out/resource_jobs.txt" "$mode" > "$out/resources_$mode.log" 2>&1
done
