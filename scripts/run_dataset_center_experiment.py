"""Run in the project's Docker toolchain. No image-processing Python needed."""
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import shlex
import subprocess

root = Path(__file__).resolve().parents[1]
out = root / 'output/dataset_center_alignment_20261020'
out.mkdir(parents=True, exist_ok=True)
pool = []
for folder in ('big', 'mid', 'small', 'rotated'):
    pool.extend(p for p in (root / 'data' / folder).rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
pool.extend((root / 'data/stereo_crop').glob('left_*.png'))
seen, unique = set(), []
for p in sorted(pool):
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest not in seen:
        seen.add(digest)
        unique.append(p)
random.Random(20261020).shuffle(unique)
(out / 'randomized_pool.txt').write_text('\n'.join(p.relative_to(root).as_posix() for p in unique)+'\n', encoding='utf-8')
cfg_path = root / 'config/tracker_config.json'
(out / 'config_snapshot.json').write_bytes(cfg_path.read_bytes())
subprocess.run(['cmake', '--build', 'build-codex-bc-bench', '--target', 'test_dual_roi_flow', '-j', '4'], cwd=root, check=True)
build = root / 'build-codex-bc-bench/tests'
options = []
for line in (build / 'CMakeFiles/test_dual_roi_flow.dir/flags.make').read_text().splitlines():
    if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
        options.extend(shlex.split(line.split('=', 1)[1]))
obj = out / 'experiment.o'
subprocess.run(['/usr/bin/c++', *options, '-c', str(root / 'scripts/experiment_dataset_centers.cpp'), '-o', str(obj)], check=True)
link = shlex.split((build / 'CMakeFiles/test_dual_roi_flow.dir/link.txt').read_text())
link = [str(obj) if x.endswith('/unit/test_dual_roi_flow.cpp.o') else x for x in link]
link[link.index('-o') + 1] = str(out / 'experiment')
subprocess.run(link, cwd=build, check=True)
with (out / 'console.log').open('w') as log:
    subprocess.run([str(out / 'experiment'), str(out / 'randomized_pool.txt'), str(out)], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
data = json.loads((out / 'results.json').read_text())
data.update(seed=20261020, deduplicated_pool_size=len(unique))
files = [root / s['path'] for s in data['samples']]
files += [cfg_path, root / 'scripts/experiment_dataset_centers.cpp', root / 'scripts/experiment_class_centers.cpp', root / 'scripts/run_dataset_center_experiment.py', root / 'yolo_onnx/yolov8n_p2.onnx', root / 'scripts/render_dataset_center_results.cpp', root / 'scripts/report_dataset_center_experiment.py']
data['sha256'] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
data['protocol'] = 'Random shuffle seed 20261020, first 6 YOLO-contained complete pairs; no pose-success filtering; independent cold-start MonoTracker; configured K and scales unchanged'
(out / 'summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
subprocess.run(['/usr/bin/c++', '-std=c++17', '-O2', '-I/usr/include/opencv4',
                str(root / 'scripts/render_dataset_center_results.cpp'), '-o', str(out / 'render'),
                '-lopencv_core', '-lopencv_imgproc', '-lopencv_imgcodecs'], check=True)
subprocess.run([str(out / 'render'), str(out)], check=True)
subprocess.run(['python3', str(root / 'scripts/report_dataset_center_experiment.py')], check=True)
