#!/usr/bin/env bash
set -euo pipefail
out=output/mono_priority_20261019
cmake --build build-codex-bc-bench --target GPNP test_roi_generator test_yolo_decode \
 test_mono_corner_identity test_mono_pose_prior test_dual_roi_flow test_disparity_chain \
 test_pose_solvers test_extractors -j 3 > "$out/build.log" 2>&1
ctest --test-dir build-codex-bc-bench \
 -R '^(test_roi_generator|test_yolo_decode|test_mono_corner_identity|test_mono_pose_prior|test_dual_roi_flow|test_disparity_chain|test_pose_solvers|test_extractors)$' \
 --output-on-failure > "$out/ctest.log" 2>&1
cat "$out/ctest.log"
