#!/usr/bin/env bash
set -euo pipefail
out=output/tt_priority_20261019
g++ -std=c++17 -O3 -DNDEBUG -DTT_LEGACY_BASELINE \
 -I"$out/before/include" -Iinclude -I/usr/include/eigen3 -I/usr/include/opencv4 \
 scripts/evaluate_tt_priority.cpp "$out/before/src/feature/TinyTargetExtractor.cpp" \
 src/utils/PoseUtils.cpp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
 -lopencv_calib3d -lopencv_features2d -o "$out/evaluate_tt_priority_before"
cp -p build-codex-bc-bench/tests/evaluate_tt_priority "$out/evaluate_tt_priority"
chmod u+x "$out/before/build-codex-bc-bench/GPNP"
