"""Runtime capture and host-side final checks, with no production input changes."""
import csv,json,re,shutil,subprocess
from pathlib import Path
from experiment_stereo_priority import OUT,read,dump,sha
def runtime():
    for mode in ('before','after'):
        print('VISUAL',mode,flush=True)
        with (OUT/('visual_'+mode+'.log')).open('w') as f:
            subprocess.run([str(OUT/('evaluate_'+mode)),str(OUT/'configs'/('physical_'+mode+'.json')),
                str(OUT/'visual_jobs.txt'),str(OUT/('visual_'+mode+'.csv')),'1',str(OUT/'native_visuals'/mode)],stdout=f,stderr=subprocess.STDOUT,check=True)
        print('MONO SMOKE',mode,flush=True)
        with (OUT/('smoke_'+mode+'.log')).open('w') as f:
            subprocess.run([str(OUT/('GPNP_'+mode)),str(OUT/'configs'/('smoke_'+mode+'.json'))],stdout=f,stderr=subprocess.STDOUT,check=True)
def records(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def verify():
    current=read('config/tracker_config.json');before=read(OUT/'config_before.json')
    new=current['stereo_frontend'].pop('robustness');assert current==before
    assert new['enabled'] is False
    old_hashes=read(OUT/'provenance.json')['hashes'];changed={p:sha(p) for p,h in old_hashes.items() if sha(p)!=h}
    assert set(changed)<= {'config/tracker_config.json'},changed
    assert sha(OUT/'GPNP_after')==sha('build-codex-bc-bench/GPNP')
    tests={}
    for p in (OUT/'regression').glob('*.log'):
        content=p.read_text(errors='replace');m=re.search(r'总计:\s*(\d+)\s*通过:\s*(\d+)\s*失败:\s*(\d+)',content)
        assert m and m[1]==m[2] and int(m[3])==0,p
        tests[p.stem]=int(m[1])
    visual={}
    ignore={'extract_ms','flow_ms','pnp_ms','total_ms'}
    for mode in ('before','after'):
        fixed={r['job']:r for r in records(OUT/('physical_'+mode+'.csv'))}
        diffs=[]
        for r in records(OUT/('visual_'+mode+'.csv')):
            diff=[k for k in r if k not in ignore and r[k]!=fixed[r['job']][k]]
            if diff:diffs.append(dict(job=r['job'],keys=diff))
        assert not diffs,diffs
        visual[mode]=dict(cases=5,non_timing_differences=diffs)
    a,b=(records(OUT/('smoke_'+m)/'poses.csv') for m in ('before','after'))
    ignore_smoke={'extract_ms','pnp_ms','roi_verify_ms'}
    diffs=[dict(frame=x['frame'],keys=[k for k in x if k not in ignore_smoke and x[k]!=y[k]]) for x,y in zip(a,b)]
    assert len(a)==len(b)==2 and not any(d['keys'] for d in diffs),diffs
    off=read(OUT/'off_compatibility.json');assert off['cases']==96 and not off['differences']
    for mode in ('before','after'):
        rows=records(OUT/('physical_'+mode+'.csv'));assert len(rows)==96
    manifest=dict(root_only_new_disabled_block=True,original_inputs_unchanged=True,regression=tests,
        total_tests=sum(tests.values()),visual=visual,mono_smoke=dict(frames=2,numerical_differences=[]),off_switch=off,
        binary_hashes={name:sha(OUT/name) for name in ('GPNP_before','GPNP_after','evaluate_before','evaluate_after')},
        timing=dict(rounds=3,order=['before/after','after/before','before/after'],cases=20,repeats=5,
                    warmup_per_case=1,samples_per_arm=300,opencv_threads=1))
    dump(OUT/'verification.json',manifest)
    files=[Path('main.cpp'),Path('CMakeLists.txt'),Path('tests/CMakeLists.txt'),Path('config/tracker_config.json'),
           Path('README.md'),Path('docs/stereo_priority_20261020.md'),Path('tests/unit/test_dual_roi_flow.cpp'),
           Path('include/common/Types.hpp'),Path('include/common/StereoRobustness.hpp'),Path('include/tracker/StereoTracker.hpp'),
           Path('include/pose/GPnPSolver.hpp'),Path('src/pose/GPnPSolver.cpp'),
           *Path('src/tracker').glob('StereoTracker*.cpp'),*Path('scripts').glob('*stereo_priority*')]
    archived={}
    for p in files:
        dest=OUT/'final_sources'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);archived[p.as_posix()]=sha(p)
    dump(OUT/'final_source_manifest.json',archived)
    print('verified',sum(tests.values()),'tests; original inputs intact',flush=True)
if __name__=='__main__':
    import sys
    runtime() if len(sys.argv)>1 and sys.argv[1]=='runtime' else verify()
