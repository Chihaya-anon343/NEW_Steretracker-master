#!/usr/bin/env bash
set -euo pipefail
out=output/stereo_priority_20261020
flags=(-std=c++17 -O3 -march=native -DNDEBUG -I. -I/usr/include/eigen3 -I/usr/include/opencv4)
libs=($(pkg-config --libs opencv4) -L/opt/onnxruntime-linux-x64-gpu-1.20.1/lib -Wl,-rpath,/opt/onnxruntime-linux-x64-gpu-1.20.1/lib -lonnxruntime -pthread)
if [[ "${1:-}" == before ]];then
    g++ "${flags[@]}" -I"$out/before/include" scripts/evaluate_stereo_priority.cpp "$out"/before_objects/*.o "${libs[@]}" -o "$out/evaluate_before"
else
    objects=();while IFS= read -r -d '' p;do [[ "$p" == */main.cpp.o ]]||objects+=("$p");done < <(find build-codex-bc-bench/CMakeFiles/GPNP.dir -name '*.o' -print0)
    g++ "${flags[@]}" -DSTEREO_ROBUST_EVAL -Iinclude scripts/evaluate_stereo_priority.cpp "${objects[@]}" "${libs[@]}" -o "$out/evaluate_after"
    mkdir -p "$out/regression"
    for name in "${@}";do
        g++ "${flags[@]}" -Iinclude -Itests/framework "tests/unit/$name.cpp" "${objects[@]}" "${libs[@]}" -o "$out/regression/$name"
        "$out/regression/$name" > "$out/regression/$name.log" 2>&1
        tail -n 4 "$out/regression/$name.log"
    done
fi
