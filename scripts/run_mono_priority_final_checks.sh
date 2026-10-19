#!/usr/bin/env bash
set -euo pipefail
out=output/mono_priority_20261019
ctest --test-dir build-codex-bc-bench \
 -R '^(test_mono_pose_prior|test_mono_corner_identity|test_dual_roi_flow)$' \
 --output-on-failure > "$out/final_ctest.log" 2>&1
"$out/test_roi_final" > "$out/final_roi.log" 2>&1
cat "$out/final_ctest.log"
