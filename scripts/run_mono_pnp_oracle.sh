#!/usr/bin/env bash
# Run inside the project container via scripts/docker-toolchain.sh bash ...
set -euo pipefail
out="${1:-output/mono_robustness_20261015}"
g++ -std=c++17 -O2 -Iinclude -I/usr/include/eigen3 -I/usr/include/opencv4 \
    scripts/mono_pnp_oracle.cpp src/pose/MonoPnPSolver.cpp \
    -lopencv_core -lopencv_calib3d -o "$out/mono_pnp_oracle"
"$out/mono_pnp_oracle" "$out/oracle_input.txt" "$out/analysis/oracle_poses.csv" > "$out/oracle_stdout.log" 2>&1
