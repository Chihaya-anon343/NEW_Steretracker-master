#!/usr/bin/env bash
set -euo pipefail
cp build-codex-bc-bench/GPNP output/stereo_priority_20261020/GPNP_after
bash scripts/build_stereo_priority.sh test_dual_roi_flow test_disparity_chain
python3 scripts/experiment_stereo_priority.py physical
python3 scripts/experiment_stereo_priority.py offset
python3 scripts/experiment_stereo_priority.py timing
python3 scripts/verify_stereo_priority.py runtime
python3 scripts/experiment_stereo_priority.py crop_after
