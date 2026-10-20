"""Offline error, common-case and timing statistics for stereo rollout."""
import csv,json,re
from collections import Counter
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from experiment_stereo_priority import OUT,read,dump
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def stats(x):
    a=np.array(x,float)
    return dict(n=len(x),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max())) if len(x) else dict(n=0)
def angle(a,b):return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
def vector(s):return [list(map(float,p.split(':'))) for p in s.split(';') if p]
def rotation(r):return np.array([[float(r['R%d%d'%(i,j)]) for j in range(3)] for i in range(3)])
def virtual(job,cls,view):
    # Offline projection of the physical bounding square. These are virtual
    # reference points, not four corners painted on the target.
    src=Path('data/NewMuBan(reordered)/0_degrees.txt').read_text()
    p=np.array([list(map(float,m)) for m in re.findall(r'Corner_\d+:\s*([-\d.]+),\s*([-\d.]+)',src)])
    c=(p.min(0)+p.max(0))/2;v=(p-c)*(6 if cls==0 else .46)
    lo,hi=v.min(0),v.max(0);v=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    if cls:v+=np.array([85,0])
    q=np.c_[v,np.zeros(4)]@np.array(job['R']).T+np.array(job['t'])-np.array([60*view,0,0])
    u=q@np.array([[900,0,400],[0,900,400],[0,0,1]]).T;return u[:,:2]/u[:,2:]
def physical():
    jobs=read(OUT/'physical_jobs.json');result=[]
    for mode in ('before','after','off','no_joint_gate'):
        path=OUT/('physical_'+mode+'.csv')
        if not path.exists():continue
        for r in rows(path):
            j=jobs[int(r['job'])];a=dict(job=j['job'],kind=j['kind'],frame=j['frame'],mode=mode,success=int(r['success']),
                stereo=int(r['stereo']),strategy=r['strategy'],source=r['source'],route=r['route'],failure=r['failure'],points=[],id_anomaly=False)
            pairs=vector(r['pairs']);raw=vector(r['raw']);tiny='TT' in r['strategy'] or r['strategy']=='TinyTarget'
            for values in raw:
                idx=int(values[0]);cls=1 if int(r['class1']) or idx>=10 else 0;identity=idx%10
                truth=virtual(j,cls,0) if tiny else np.array(j['truth'][0][cls])
                # Legacy joint secondary uses annotation centroid ordering; map
                # its model index back to source annotation row before scoring.
                if mode in ('before','off') and cls==1 and not int(r['class1']) and not tiny:
                    physical=np.array(j['truth'][0][1]) # ordering is computed in TEMPLATE space below
                    text=Path(OUT/'physical/class1.txt').read_text();ann=np.array([list(map(float,m)) for m in re.findall(r'Corner_\d+:\s*([-\d.]+),\s*([-\d.]+)',text)])
                    delta=ann-ann.mean(0);ang=np.mod(np.arctan2(-delta[:,0],-delta[:,1]),2*np.pi)
                    start=int(np.argmin(np.minimum(ang,2*np.pi-ang)));identity=(identity+start)%10
                if 0<=identity<len(truth):a['points'].append(dict(cls=cls,id=identity,p=values[1:3],error=float(np.linalg.norm(np.array(values[1:3])-truth[identity]))))
            for cls in (0,1):
                group=[p for p in a['points'] if p['cls']==cls]
                if not group:continue
                truth=virtual(j,cls,0) if tiny else np.array(j['truth'][0][cls]);px=np.array([p['p'] for p in group]);ids=np.array([p['id'] for p in group])
                dist=np.linalg.norm(px[:,None]-truth[None],axis=2);rr,cc=linear_sum_assignment(dist)
                if np.any(ids[rr]!=cc) and dist[np.arange(len(ids)),ids].mean()-dist[rr,cc].mean()>.5:a['id_anomaly']=True
            if a['success']:
                R=rotation(r);t=np.array([float(r[k]) for k in ('tx','ty','tz')]);a.update(R=R.tolist(),t=t.tolist(),
                    depth_relative=float(abs(t[2]-j['t'][2])/j['t'][2]),normal_error=angle(R[:,2],np.array(j['R'])[:,2]))
            result.append(a)
    dump(OUT/'physical_metrics.json',result)
    summary=[]
    for kind in ('bc','tt','dual','fallback'):
        for mode in ('before','after','off','no_joint_gate'):
            rr=[r for r in result if r['kind']==kind and r['mode']==mode];good=[r for r in rr if r['success']]
            summary.append(dict(kind=kind,mode=mode,n=len(rr),success=len(good),stereo=sum(r['stereo'] for r in good),
                strategies=dict(Counter(r['strategy'] for r in good)),normal=stats([r['normal_error'] for r in good]),
                depth=stats([r['depth_relative'] for r in good]),corner=stats([p['error'] for r in good for p in r['points']]),
                id_anomaly=sum(r['id_anomaly'] for r in good),failures=dict(Counter(r['failure'] for r in rr if not r['success']))))
    dump(OUT/'physical_summary.json',summary)
    common=[]
    for kind in ('bc','tt','dual','fallback'):
        arms={m:{r['job']:r for r in result if r['kind']==kind and r['mode']==m} for m in ('before','after')}
        both=[i for i in arms['before'] if arms['before'][i]['success'] and arms['after'][i]['success']]
        item=dict(kind=kind,n=len(both),lost=sum(r['success'] and not arms['after'][i]['success'] for i,r in arms['before'].items()),
            recovered=sum(r['success'] and not arms['before'][i]['success'] for i,r in arms['after'].items()))
        for m in arms:
            item[m]=dict(normal=stats([arms[m][i]['normal_error'] for i in both]),depth=stats([arms[m][i]['depth_relative'] for i in both]))
        common.append(item)
    dump(OUT/'physical_common.json',common)
    same_ids=[]
    for kind in ('bc','tt','dual','fallback'):
        arms={m:{r['job']:r for r in result if r['kind']==kind and r['mode']==m and r['success']} for m in ('before','after')}
        errors={m:[] for m in arms}
        for i in arms['before'].keys()&arms['after'].keys():
            points={m:{(p['cls'],p['id']):p for p in arms[m][i]['points']} for m in arms}
            for key in points['before'].keys()&points['after'].keys():
                for m in arms:errors[m].append(points[m][key]['error'])
        same_ids.append(dict(kind=kind,before=stats(errors['before']),after=stats(errors['after'])))
    dump(OUT/'same_id_corner_summary.json',same_ids)
    jumps=[]
    for kind in ('bc','tt','dual','fallback'):
        arms={m:{r['frame']:r for r in result if r['kind']==kind and r['mode']==m} for m in ('before','after')}
        frames=[i for i in arms['before'] if all(arms[m][i]['success'] for m in arms)]
        adjacent=[(i-1,i) for i in frames if i-1 in frames]
        truth={j['frame']:j for j in jobs if j['kind']==kind}
        item=dict(kind=kind,common_adjacent_pairs=len(adjacent))
        for m in arms:
            zsteps=[];nsteps=[]
            for a,b in adjacent:
                zsteps.append(abs((arms[m][b]['t'][2]-arms[m][a]['t'][2])-(truth[b]['t'][2]-truth[a]['t'][2]))/truth[b]['t'][2])
                nsteps.append(angle(np.array(arms[m][a]['R'])[:,2],np.array(arms[m][b]['R'])[:,2])-
                              angle(np.array(truth[a]['R'])[:,2],np.array(truth[b]['R'])[:,2]))
            item[m]=dict(z_jumps=sum(v>.05 for v in zsteps),normal_jumps=sum(v>10 for v in nsteps),z_step_relative=stats(zsteps))
        jumps.append(item)
    dump(OUT/'jumps.json',jumps)
    # Verify the compatibility switch without relying on timing equality.
    baseline=rows(OUT/'physical_before.csv');off=rows(OUT/'physical_off.csv') if (OUT/'physical_off.csv').exists() else []
    ignored={'extract_ms','flow_ms','pnp_ms','total_ms'}
    differences=[dict(job=a['job'],keys=[k for k in a if k not in ignored and a[k]!=b[k]]) for a,b in zip(baseline,off)]
    dump(OUT/'off_compatibility.json',dict(cases=len(off),differences=[d for d in differences if d['keys']]))
    offset=[]
    for mode in ('offset_gate','offset_open'):
        path=OUT/(mode+'.csv')
        if not path.exists():continue
        rr=rows(path);valid=[r for r in rr if int(r['success'])]
        offset.append(dict(mode=mode,success=len(valid),strategies=dict(Counter(r['strategy'] for r in valid)),
            normal=stats([angle(rotation(r)[:,2],np.array(jobs[int(r['job'])]['R'])[:,2]) for r in valid]),
            depth=stats([abs(float(r['tz'])-jobs[int(r['job'])]['t'][2])/jobs[int(r['job'])]['t'][2] for r in valid]),
            routes=dict(Counter(r['route'] for r in rr))))
    dump(OUT/'offset_summary.json',offset)
def crop():
    manifest=read('data/stereo_crop/manifest.json')['frames'];summary=[]
    for mode in ('before','after'):
        folder=OUT/'crop'/mode
        if not (folder/'poses.csv').exists():continue
        pp=rows(folder/'poses.csv');metrics=[]
        for r in pp:
            frame=int(r['frame']);points=rows(folder/('disparity_f%d.csv'%frame));g=manifest[frame-1]
            e=[float(p['disparity_px'])-g['disparity'] for p in points]
            v=[float(p['right_y'])-float(p['left_y'])-g['dy_px'] for p in points]
            metrics.append(dict(frame=frame,ok=bool(points and int(points[0].get('frame_match_ok',1))),strategy=r['strategy'],failure=r['failure_reason'],
                                errors=e,vertical=v,n=len(e),pnp_ms=float(r['pnp_ms'])))
        dump(OUT/('crop_metrics_'+mode+'.json'),metrics)
        errors=[e for r in metrics for e in r['errors']];vertical=[e for r in metrics for e in r['vertical']]
        summary.append(dict(mode=mode,frames=len(metrics),success=sum(r['ok'] for r in metrics),points=len(errors),
                            absolute=stats([abs(e) for e in errors]),rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                            vertical=stats([abs(e) for e in vertical]),strategies=dict(Counter(r['strategy'] for r in metrics)),
                            failures=dict(Counter(r['failure'] for r in metrics if not r['ok'])),pnp_ms=sum(r['pnp_ms'] for r in metrics)))
    dump(OUT/'crop_summary.json',summary)
def timing():
    jobs=read(OUT/'physical_jobs.json');allrows={m:[] for m in ('before','after')}
    for m in allrows:
        for f in OUT.glob('timing_%s_*.csv'%m):allrows[m]+=rows(f)
    quality={m:{int(r['job']):int(r['success']) for r in rows(OUT/('physical_'+m+'.csv'))} for m in allrows}
    summary=[]
    for kind in ('bc','tt','dual','fallback'):
        for m,rr in allrows.items():
            for scope in ('all','common_success'):
                selected=[r for r in rr if jobs[int(r['job'])]['kind']==kind and (scope=='all' or all(quality[a][int(r['job'])] for a in quality))]
                summary.append(dict(kind=kind,mode=m,scope=scope,samples=len(selected),
                    cases=len({r['job'] for r in selected}),extract=stats([float(r['extract_ms']) for r in selected]),
                    pose=stats([float(r['pnp_ms']) for r in selected]),total=stats([float(r['total_ms']) for r in selected])))
    dump(OUT/'timing_summary.json',summary)
if __name__=='__main__':
    import sys
    for action in sys.argv[1:] or ['physical','crop','timing']:globals()[action]()
