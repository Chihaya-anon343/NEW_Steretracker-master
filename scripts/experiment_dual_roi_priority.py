"""Frozen before/after DualROI replay. No ground truth enters runtime configs."""
import argparse, copy, csv, hashlib, json, shutil, subprocess
from pathlib import Path
OUT=Path('output/dual_roi_priority_20261020')
DATA=Path('tests/data/mono_robustness_20261015')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
    base=read(OUT/'config_before.json'); jobs=[]
    for name in ('reference','normal_ambiguity','oblique_degraded'):
        for mode in ('before','after'):
            # Existing cache contains pixel-exact BMPs, verified by the TT study.
            source=Path('output/tt_priority_20261019/configs')/(name+'_G_cached.json')
            cfg=read(source) if source.exists() else copy.deepcopy(base)
            cfg['mode']='normal';cfg['mono_mode']=True
            cfg['manual_roi']['enabled']=False
            cfg['input_system'].update(max_frames=0,use_threaded_capture=False)
            if not source.exists():
                cfg['input_system']['image'].update(type='sequence',directory_path=(DATA/name).as_posix(),sequence_pattern='frame')
            cfg['temporal']['enabled']=False;cfg['eskf']={'enabled':False}
            cfg['strategies']['dual_roi']['mono_robustness']={'enabled':mode=='after'}
            dest=OUT/'runs'/(name+'_'+mode)
            cfg['output']=dict(visualize=False,log_file=True,pose_csv=True,directory=dest.as_posix())
            path=OUT/'configs'/(name+'_'+mode+'.json');dump(path,cfg)
            jobs.append(dict(name=name+'_'+mode,sequence=name,mode=mode,config=path.as_posix(),config_sha256=sha(path),
                             binary=(OUT/('GPNP_'+mode)).as_posix()))
    dump(OUT/'experiment.json',dict(jobs=jobs,source_config_sha256=sha(OUT/'config_before.json'),
          dataset_sha256=sha(DATA/'suite_manifest.json'),before_binary_sha256=sha(OUT/'GPNP_before')))
def run(mode):
    for j in read(OUT/'experiment.json')['jobs']:
        if j['mode']!=mode: continue
        assert sha(j['config'])==j['config_sha256']
        dest=OUT/'runs'/j['name'];dest.mkdir(parents=True,exist_ok=True)
        print('RUN',j['name'],flush=True)
        with (dest/'stdout.log').open('w') as f:
            subprocess.run([j['binary'],j['config']],stdout=f,stderr=subprocess.STDOUT,check=True)
        n=len(list(csv.DictReader((dest/'poses.csv').open())))
        expected=len(read(DATA/j['sequence']/'manifest.json')['frames'])
        assert n==expected,(n,expected)
        print('DONE',j['name'],n,flush=True)
def prepare_fixed():
    original=read('output/corner_quality_20261016/jobs.json')
    boxes={(j['sequence'],j['frame'],j['cls']):j for j in original if j['strategy']=='BC'}
    cache=Path('output/tt_priority_20261019/image_cache.csv')
    mapping={r['source']:r['cached'] for r in csv.DictReader(cache.open())}
    jobs=[]
    for (sequence,frame,cls),primary in boxes.items():
        if cls:continue
        secondary=boxes.get((sequence,frame,1))
        if not secondary or not secondary['fully_visible']:continue
        roi=secondary['roi'];area=roi[2]*roi[3]
        if primary['fully_visible'] and primary['roi'][2]*primary['roi'][3]>=18000 and min(roi[2:])>=8:
            jobs.append(dict(kind='joint',primary=primary['roi'],secondary=roi,source=primary))
        # Deliberate full-frame primary isolates C1 fallback; it is not a YOLO
        # success-rate test. No GT pixels or pose values enter estimator input.
        if area>=80 and frame%3==0:
            jobs.append(dict(kind='fallback',primary=[0,0,1744,1744],secondary=roi,source=primary))
    lines=[]
    for i,j in enumerate(jobs):
        j['job']=i;j['sequence']=j['source']['sequence'];j['frame']=j['source']['frame']
        j['path']=j['source']['path'];j['tilt']=j['source']['tilt'];j.pop('source')
        path=mapping.get(j['path'],j['path'])
        lines.append(' '.join(map(str,[i,path,*j['primary'],*j['secondary']])))
    dump(OUT/'fixed_jobs.json',jobs)
    (OUT/'fixed_jobs.txt').write_bytes(('\n'.join(lines)+'\n').encode())
    for mode in ('before','after'):
        cfg=read(OUT/'configs'/('reference_'+mode+'.json'))
        dump(OUT/'configs'/('fixed_'+mode+'.json'),cfg)
    print('Fixed cases',len(jobs),'joint',sum(j['kind']=='joint' for j in jobs),flush=True)
def run_fixed():
    for mode in ('before','after'):
        print('FIXED',mode,flush=True)
        with (OUT/('fixed_'+mode+'.log')).open('w') as f:
            subprocess.run([str(OUT/('evaluate_'+mode)),str(OUT/'configs'/('fixed_'+mode+'.json')),
                str(OUT/'fixed_jobs.txt'),str(OUT/('fixed_'+mode+'.csv')),'1'],stdout=f,stderr=subprocess.STDOUT,check=True)
        assert len(list(csv.DictReader((OUT/('fixed_'+mode+'.csv')).open())))==len(read(OUT/'fixed_jobs.json'))
        print('DONE FIXED',mode,flush=True)
def timing():
    dest=OUT/'timing';dest.mkdir(exist_ok=True)
    for turn in range(4):
        for mode in (('before','after') if turn%2==0 else ('after','before')):
            print('TIMING',turn,mode,flush=True)
            with (dest/('%s_%d.log'%(mode,turn))).open('w') as f:
                subprocess.run([str(OUT/('evaluate_'+mode)),str(OUT/'configs'/('fixed_'+mode+'.json')),
                    str(OUT/'timing_jobs.txt'),str(dest/('%s_%d.csv'%(mode,turn))),'5'],
                    stdout=f,stderr=subprocess.STDOUT,check=True)
    dump(dest/'manifest.json',dict(rounds=4,repeats=5,cases=32,opencv_threads=1,
         binary_sha256={m:sha(OUT/('evaluate_'+m)) for m in ('before','after')},
         cases_sha256=sha(OUT/'timing_jobs.txt'),scope='excludes image decoding, initialization, warmup and output writing; includes TT diagnostic collection and all attempted PnP solves'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fixed','run_fixed','timing','before','after']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='fixed':prepare_fixed()
    elif a.action=='run_fixed':run_fixed()
    elif a.action=='timing':timing()
    else:run(a.action)
