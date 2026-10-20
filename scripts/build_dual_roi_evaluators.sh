#!/usr/bin/env bash
set -euo pipefail
out=output/dual_roi_priority_20261020
target=build-codex-bc-bench/tests/CMakeFiles/evaluate_mono_dual_roi.dir
g++ -std=c++17 -O3 -DNDEBUG -Iinclude -I/usr/include/eigen3 -I/usr/include/opencv4 \
 -c "$out/before/src/tracker/MonoTracker.cpp" -o "$out/MonoTracker_before.o"
objects=()
while IFS= read -r -d '' object; do
    [[ "$object" == */src/tracker/MonoTracker.cpp.o ]] && continue
    objects+=("$object")
done < <(find "$target" -name '*.o' -print0)
g++ "${objects[@]}" "$out/MonoTracker_before.o" -lopencv_core -lopencv_imgproc \
 -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_video -pthread \
 -o "$out/evaluate_before"
cp -p build-codex-bc-bench/tests/evaluate_mono_dual_roi "$out/evaluate_after"
