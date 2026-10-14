"""Run the saved before binary and current binary on identical stereo inputs.

Each arm gets one complete warmup and three formal runs, alternated by arm.
Uses the repository Docker wrapper; no tracker configuration or input is overwritten.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--output', default='output/solver_geometry_optimization')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project / args.output
    root.mkdir(parents=True, exist_ok=True)
    after_binary = root / 'GPNP_after'
    after_binary.write_bytes((project / 'build-codex-bc-bench/GPNP').read_bytes())
    binaries = {'before': root / 'GPNP_before', 'after': after_binary}
    configs = {'before': root / 'config_before.json', 'after': project / 'config/tracker_config.json'}
    inputs = sorted((project / 'data/stereo_crop').glob('*.png'))
    manifest = project / 'data/stereo_crop/manifest.json'
    current_config = json.loads(configs['after'].read_text(encoding='utf-8-sig'))
    strategies = current_config['strategies']
    template_files = {project / strategies['board_template']['template_path']}
    for name in ('binary_corner', 'tiny_target'):
        template_files.update(p for p in (project / strategies[name]['template_dir']).iterdir() if p.is_file())
    for key in ('class0_points_path', 'class1_points_path'):
        template_files.add(project / strategies['dual_roi'][key])
    source_files = [p for folder in ('include', 'src') for p in (project / folder).rglob('*')
                    if p.suffix in ('.cpp', '.hpp')]
    tracked = [*inputs, manifest, *sorted(template_files), *source_files,
               project / 'yolo_onnx/yolov8n_p2.onnx', *configs.values(), *binaries.values()]
    hashes = {str(p.relative_to(project)): sha(p) for p in tracked}
    for arm in binaries:
        folder = root / arm
        folder.mkdir(exist_ok=True)
        (folder / 'config_snapshot.json').write_bytes(configs[arm].read_bytes())
        (folder / 'manifest_snapshot.json').write_bytes(manifest.read_bytes())
        cfg = json.loads(configs[arm].read_text(encoding='utf-8-sig'))
        cfg['output']['visualize'] = False
        cfg['output']['log_file'] = True
        cfg['input_system']['max_frames'] = 0
        for name in ['warmup', *[f'run{i}' for i in range(1, args.runs + 1)]]:
            run_cfg = copy.deepcopy(cfg)
            run_cfg['output']['directory'] = str((folder / name).relative_to(project)).replace('\\', '/')
            save(folder / f'{name}.json', run_cfg)
    save(root / 'provenance.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
         'hashes_before': hashes, 'formal_runs_per_arm': args.runs,
         'protocol': 'one full warmup per arm, alternating before/after each formal repeat',
         'scope': 'disparity only; optimizer accuracy covered separately by physical synthetic tests'})
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    for name in ['warmup', *[f'run{i}' for i in range(1, args.runs + 1)]]:
        for arm in binaries:
            folder = root / arm
            print(f'{arm}/{name}: starting', flush=True)
            command = [str(bash), 'scripts/docker-toolchain.sh',
                       './' + str(binaries[arm].relative_to(project)).replace('\\', '/'),
                       str((folder / f'{name}.json').relative_to(project)).replace('\\', '/')]
            with (folder / f'{name}_console.log').open('wb') as log:
                subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'{arm}/{name}: complete', flush=True)
    for arm in binaries:
        command = [sys.executable, str(project / 'scripts/analyze_stereo_mixed_strategies.py'),
                   str(root / arm), '--runs', *[f'run{i}' for i in range(1, args.runs + 1)]]
        with (root / arm / 'analysis_console.log').open('wb') as log:
            subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
    hashes_after = {str(p.relative_to(project)): sha(p) for p in tracked}
    if hashes_after != hashes:
        raise RuntimeError('Inputs/config/program changed during measurement; results are not comparable')
    provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(), inputs_and_programs_unchanged=True)
    save(root / 'provenance.json', provenance)
    summary = {arm: json.loads((root / arm / 'summary.json').read_text(encoding='utf-8')) for arm in binaries}
    save(root / 'comparison.json', summary)
    for arm, data in summary.items():
        print(arm, json.dumps(data['overall'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
