"""Offline DualROI analysis. Ground truth is never used for runtime gating."""
import csv, json, re
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiment_dual_roi_priority import OUT, DATA, read, dump
from analyze_mono_robustness import analyze_job, angle, stats

def rows(p): return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def rotation(r): return np.array([[float(r['R%d%d'%(i,j)]) for j in range(3)] for i in range(3)])
def observations(r):
    return [list(map(float,p.split(':'))) for p in r['observations'].split(';') if p]
def virtual(gt,cls):
    src=Path('scripts/class%d_points.txt'%cls).read_text()
    p=np.array([list(map(float,m)) for m in re.findall(r'Corner_\d+:\s*([-\d.]+),\s*([-\d.]+)',src)])
    lo,hi=p.min(0),p.max(0)
    box=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    return cv2.perspectiveTransform(box[None],np.array(gt['H']))[0]
def fixed():
    jobs=read(OUT/'fixed_jobs.json'); manifests={n:read(DATA/n/'manifest.json')['frames'] for n in {j['sequence'] for j in jobs}}
    allrows=[]
    for mode in ('before','after'):
        for r in rows(OUT/('fixed_'+mode+'.csv')):
            j=jobs[int(r['job'])];gt=manifests[j['sequence']][j['frame']]
            a=dict(job=j['job'],mode=mode,kind=j['kind'],sequence=j['sequence'],frame=j['frame'],tilt=gt['tilt_deg'],
                success=int(r['success']),strategy=r['strategy'],route=r['route'],extract_ms=float(r['extract_ms']),
                total_ms=float(r['total_ms']),point_errors=[],mismatch=False)
            if a['success']:
                R=rotation(r);t=np.array([float(r[k]) for k in ('tx','ty','tz')])
                assert np.isfinite(R).all() and np.isfinite(t).all() and t[2]>0
                a.update(normal_error_deg=angle(R[:,2],np.array(gt['R'])[:,2]),
                    depth_error=float(abs(t[2]-gt['t_mm'][2])/gt['t_mm'][2]))
                groups={}
                for values in observations(r):
                    idx=int(values[0]);p=np.array(values[1:3]);c1=r['strategy'].startswith('DualRoi_C1')
                    cls=1 if c1 or idx>=10 else 0;identity=idx if c1 else idx%10
                    # Legacy joint class1 model rotated annotation rows by 7
                    # using its centroid. New DualROI preserves annotation rows.
                    if mode=='before' and cls==1 and not c1:identity=(identity+7)%10
                    truth=virtual(gt,cls) if r['strategy'].endswith('TT') else np.array(gt['class%d'%cls]['points_px'])
                    a['point_errors'].append(float(np.linalg.norm(p-truth[identity])))
                    groups.setdefault(cls,[]).append((p,identity,truth))
                a['point_rmse']=float(np.sqrt(np.mean(np.square(a['point_errors']))))
                for group in groups.values():
                    p=np.array([v[0] for v in group]);ids=np.array([v[1] for v in group]);truth=group[0][2]
                    dist=np.linalg.norm(p[:,None]-truth[None],axis=2);rr,cc=linear_sum_assignment(dist)
                    a['mismatch']|=bool(np.any(cc!=ids[rr]) and (dist[np.arange(len(ids)),ids].mean()-dist[rr,cc].mean())>.5)
            allrows.append(a)
    summaries=[]
    for kind in ('joint','fallback'):
        for mode in ('before','after'):
            for scope in ('all','tilt_le50'):
                rr=[r for r in allrows if r['mode']==mode and r['kind']==kind and (scope=='all' or r['tilt']<=50)]
                good=[r for r in rr if r['success']]
                summaries.append(dict(kind=kind,mode=mode,scope=scope,attempts=len(rr),success=len(good),
                    strategies=dict(Counter(r['strategy'] for r in good)),mismatches=sum(r['mismatch'] for r in good),
                    normal_error=stats([r['normal_error_deg'] for r in good]),depth_error=stats([r['depth_error'] for r in good]),
                    point_error=stats([v for r in good for v in r['point_errors']]),
                    severe_depth=sum(r['depth_error']>1 for r in good)))
    look={(r['mode'],r['job']):r for r in allrows};paired=[]
    for kind,scope in [(k,s) for k in ('joint','fallback') for s in ('all','tilt_le50')]:
        pairs=[(look['before',j['job']],look['after',j['job']]) for j in jobs
               if j['kind']==kind and (scope=='all' or j['tilt']<=50)]
        common=[(a,b) for a,b in pairs if a['success'] and b['success']]
        paired.append(dict(kind=kind,scope=scope,common=len(common),lost=sum(a['success'] and not b['success'] for a,b in pairs),
            recovered=sum(not a['success'] and b['success'] for a,b in pairs),
            normal_before=stats([a['normal_error_deg'] for a,b in common]),normal_after=stats([b['normal_error_deg'] for a,b in common]),
            depth_before=stats([a['depth_error'] for a,b in common]),depth_after=stats([b['depth_error'] for a,b in common]),
            point_before=stats([v for a,b in common for v in a['point_errors']]),
            point_after=stats([v for a,b in common for v in b['point_errors']]),
            mismatches_before=sum(a['mismatch'] for a,b in common),mismatches_after=sum(b['mismatch'] for a,b in common)))
    dump(OUT/'fixed_frames.json',allrows);dump(OUT/'fixed_summary.json',summaries);dump(OUT/'fixed_comparison.json',paired)
    raw={m:{int(r['job']):r for r in rows(OUT/('fixed_'+m+'.csv'))} for m in ('before','after')}
    errors={'before':[],'after':[]};cases=0
    for j in jobs:
        if j['kind']!='fallback' or j['tilt']>50:continue
        a,b=raw['before'][j['job']],raw['after'][j['job']]
        if not all(r['success']=='1' and r['strategy']=='DualRoi_C1BC' for r in (a,b)):continue
        gt=np.array(manifests[j['sequence']][j['frame']]['class1']['points_px'])
        points=[{int(v[0]):np.array(v[1:3]) for v in observations(r)} for r in (a,b)]
        cases+=1
        for idx in points[0].keys()&points[1].keys():
            for m,p in zip(('before','after'),points):errors[m].append(float(np.linalg.norm(p[idx]-gt[idx])))
    dump(OUT/'same_id_points.json',dict(cases=cases,before=stats(errors['before']),after=stats(errors['after'])))
    print(json.dumps(paired,indent=2))
def timing():
    jobs=read(OUT/'fixed_jobs.json')
    samples={m:[r for f in (OUT/'timing').glob(m+'_*.csv') for r in rows(f)] for m in ('before','after')}
    assert sum(map(len,samples.values()))==1280
    ok={m:{int(r['job']) for r in rr if r['success']=='1'} for m,rr in samples.items()}
    common=ok['before']&ok['after'];all_summary=[];common_summary=[]
    for m in samples:
        for kind in ('joint','fallback'):
            for same,target in ((False,all_summary),(True,common_summary)):
                rr=[r for r in samples[m] if jobs[int(r['job'])]['kind']==kind and (not same or int(r['job']) in common)]
                target.append(dict(mode=m,kind=kind,cases=len({r['job'] for r in rr}),samples=len(rr),
                    extract_ms=float(np.median([float(r['extract_ms']) for r in rr])),
                    total_ms=float(np.median([float(r['total_ms']) for r in rr])),
                    total_p95=float(np.percentile([float(r['total_ms']) for r in rr],95))))
    dump(OUT/'timing_summary.json',all_summary);dump(OUT/'timing_common.json',common_summary)
def pipeline():
    frames=[]
    for job in read(OUT/'experiment.json')['jobs']:
        log=OUT/'runs'/(job['name']+'_stdout.log')
        log.write_bytes((OUT/'runs'/job['name']/'stdout.log').read_bytes())
        _,rr,_=analyze_job(job,DATA,OUT)
        for r in rr:r.update(mode=job['mode'],sequence=job['sequence'])
        frames+=rr
    summary=[]
    for mode in ('before','after'):
        rr=[r for r in frames if r['mode']==mode];good=[r for r in rr if r['success']]
        summary.append(dict(mode=mode,frames=len(rr),success=len(good),strategies=dict(Counter(r['strategy'] for r in good)),
            normal_jumps=sum(r['z_jump'] for r in rr),depth_jumps=sum(r['depth_jump'] for r in rr),
            normal_error=stats([r['normal_error_deg'] for r in good]),depth_error=stats([r['depth_rel_error'] for r in good])))
    look={(r['mode'],r['sequence'],r['frame']):r for r in frames}
    common=[(r,look['after',r['sequence'],r['frame']]) for r in frames if r['mode']=='before' and r['success'] and look['after',r['sequence'],r['frame']]['success']]
    adjacent=[(a,b) for a,b in common if a['frame']>0 and look['before',a['sequence'],a['frame']-1]['success'] and look['after',a['sequence'],a['frame']-1]['success']]
    comparison=dict(common_success=len(common),common_adjacent=len(adjacent),
        normal_jumps_before=sum(a['z_jump'] for a,b in adjacent),normal_jumps_after=sum(b['z_jump'] for a,b in adjacent),
        depth_jumps_before=sum(a['depth_jump'] for a,b in adjacent),depth_jumps_after=sum(b['depth_jump'] for a,b in adjacent))
    dump(OUT/'pipeline_frames.json',frames);dump(OUT/'pipeline_summary.json',summary);dump(OUT/'pipeline_comparison.json',comparison)
    print(json.dumps(summary+[comparison],indent=2))
if __name__=='__main__':fixed();pipeline();timing()
