"""Final frozen-artifact and output compatibility verification."""
import argparse,csv,hashlib,json,re,subprocess
from pathlib import Path
from experiment_dual_roi_priority import OUT,read,dump,sha
def unit():
    dest=OUT/'regression'
    for name in ('test_mono_corner_identity','test_tiny_target_robustness','test_mono_dual_roi'):
        with (dest/(name+'.log')).open('w') as f:
            subprocess.run(['build-codex-bc-bench/tests/'+name],stdout=f,stderr=subprocess.STDOUT,check=True)
def verify():
    def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
    fixed={m:rows(OUT/('fixed_'+m+'.csv')) for m in ('before','after')}
    assert all(len(a)==845 for a in fixed.values())
    actual={int(r['job']):r for r in fixed['after']}
    native=rows(OUT/'visual_native.csv');differences=[]
    for r in native:
        for k in r:
            if k not in ('extract_ms','pnp_ms','total_ms') and r[k]!=actual[int(r['job'])][k]:differences.append((r['job'],k))
    assert not differences,differences
    logs={}
    for name in ('test_dual_roi_flow','test_disparity_chain','test_extractors','test_mono_pose_prior',
                 'test_mono_corner_identity','test_tiny_target_robustness','test_mono_dual_roi'):
        text=(OUT/'regression'/(name+'.log')).read_text(encoding='utf-8',errors='replace')
        assert '[FAIL]' not in text,name
        logs[name]=len(re.findall(r'\[PASS\]',text))
    sources=['main.cpp','include/common/Types.hpp','include/common/MonoDualRoiPolicy.hpp',
             'include/tracker/MonoTracker.hpp','include/feature/BinaryCornerExtractor.hpp',
             'src/tracker/MonoTracker.cpp','src/feature/BinaryCornerExtractor.cpp']
    for j in read(OUT/'experiment.json')['jobs']:assert sha(j['config'])==j['config_sha256']
    dump(OUT/'verification.json',dict(fixed_cases_per_version=845,pipeline_frames_per_version=480,
         tests=logs,total_test_cases=sum(logs.values()),native_visual_cases=len(native),capture_numerical_differences=differences,
         binaries={m:sha(OUT/('GPNP_'+m)) for m in ('before','after')},
         evaluators={m:sha(OUT/('evaluate_'+m)) for m in ('before','after')},
         production_matches_archive=sha('build-codex-bc-bench/GPNP')==sha(OUT/'GPNP_after'),
         source_sha256={p:sha(p) for p in sources},compatibility=read(OUT/'compatibility.json')))
    print('Tests',sum(logs.values()),'native capture unchanged',len(native),'cases')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['unit','verify']);a=p.parse_args()
    unit() if a.action=='unit' else verify()
