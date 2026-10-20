#!/usr/bin/env bash
set -euo pipefail
out=output/contour_scan_20261019
version="${1:?before or after}"
if [ "$version" = before ]; then
    source_root="$out/before"
elif [ "$version" = after ]; then
    source_root=.
else
    exit 2
fi
g++ -std=c++17 -O2 -I"$source_root/include" -I/usr/include/eigen3 -I/usr/include/opencv4 \
 scripts/benchmark_contour_scan.cpp "$source_root/src/feature/BinaryCornerExtractor.cpp" \
 "$source_root/src/feature/TinyTargetExtractor.cpp" "$source_root/src/matching/MonoCornerIdentity.cpp" \
 "$source_root/src/utils/PoseUtils.cpp" \
 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d \
 -o "$out/benchmark_$version"
