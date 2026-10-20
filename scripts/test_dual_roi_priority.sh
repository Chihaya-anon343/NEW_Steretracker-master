#!/usr/bin/env bash
# Reuse the freshly built production objects; avoid compiling all tracker
# implementations once per regression executable.
set -euo pipefail
out=output/dual_roi_priority_20261020
obj=build-codex-bc-bench/CMakeFiles/GPNP.dir
objects=()
while IFS= read -r -d '' object; do
    [[ "$object" == */main.cpp.o ]] && continue
    objects+=("$object")
done < <(find "$obj" -name '*.o' -print0)
mkdir -p "$out/regression"
names=(test_dual_roi_flow test_disparity_chain test_extractors test_mono_pose_prior)
if [ "$#" -gt 0 ];then names=("$@");fi
for name in "${names[@]}"; do
    g++ -std=c++17 -O3 -march=native -DNDEBUG -Iinclude -I. -Itests/framework -I/usr/include/eigen3 -I/usr/include/opencv4 \
      -I/opt/onnxruntime-linux-x64-gpu-1.20.1/include "tests/unit/$name.cpp" "${objects[@]}" \
      $(pkg-config --libs opencv4) -L/opt/onnxruntime-linux-x64-gpu-1.20.1/lib \
      -Wl,-rpath,/opt/onnxruntime-linux-x64-gpu-1.20.1/lib -lonnxruntime -pthread -o "$out/regression/$name"
    "$out/regression/$name" > "$out/regression/$name.log" 2>&1
    tail -n 4 "$out/regression/$name.log"
done
