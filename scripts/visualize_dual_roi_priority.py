"""Actual emitted points and estimated axes; no pose or observation adjustment."""
from pathlib import Path
import csv
import cv2
import numpy as np
from experiment_dual_roi_priority import OUT, DATA, read, dump
from analyze_dual_roi_priority import rotation, observations, virtual

jobs=read(OUT/'fixed_jobs.json');details=read(OUT/'fixed_frames.json')
look={(r['mode'],r['job']):r for r in details}
raw={m:{int(r['job']):r for r in csv.DictReader((OUT/('fixed_'+m+'.csv')).open())} for m in ('before','after')}
common=[j for j in jobs if look['before',j['job']]['success'] and look['after',j['job']]['success'] and j['tilt']<=50]
joint=[j for j in common if j['kind']=='joint' and raw['before'][j['job']]['strategy']=='DualRoi_BCJ']
bc=[j for j in common if j['kind']=='fallback' and all(raw[m][j['job']]['strategy']=='DualRoi_C1BC' for m in raw)]
tt=[j for j in common if raw['after'][j['job']]['strategy']=='DualRoi_C1TT']
selected=[]
if joint:selected.append(('primary_preserved',max(joint,key=lambda j:look['before',j['job']]['depth_error']-look['after',j['job']]['depth_error'])))
if bc:selected.append(('class1_bc',max(bc,key=lambda j:look['before',j['job']]['point_rmse']-look['after',j['job']]['point_rmse'])))
if tt:selected.append(('class1_tt',min(tt,key=lambda j:look['after',j['job']]['normal_error_deg'])))
rejected=[j for j in jobs if j['kind']=='fallback' and j['tilt']<=50 and look['before',j['job']]['success'] and not look['after',j['job']]['success']]
if rejected:selected.append(('rejected_small',min(rejected,key=lambda j:look['before',j['job']]['normal_error_deg'])))
dest=OUT/'visuals';dest.mkdir(exist_ok=True);captures=[]
for name,j in selected:
    image=cv2.imread(j['path']);gt=read(DATA/j['sequence']/'manifest.json')['frames'][j['frame']]
    K=np.array(read(DATA/j['sequence']/'manifest.json')['K'])
    box=j['primary'] if j['kind']=='joint' else j['secondary'];x,y,w,h=box
    margin=max(12,int(max(w,h)*(.1 if j['kind']=='joint' else .7)))
    x0,y0=max(0,x-margin),max(0,y-margin);x1,y1=min(image.shape[1],x+w+margin),min(image.shape[0],y+h+margin)
    panels=[]
    for mode in ('before','after'):
        r=raw[mode][j['job']];q=look[mode,j['job']]
        crop=image[y0:y1,x0:x1].copy();scale=min(640/crop.shape[1],640/crop.shape[0])
        crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
        canvas=np.zeros((735,660,3),np.uint8)+24;canvas[75:75+crop.shape[0],10:10+crop.shape[1]]=crop
        def pt(p):return tuple(np.rint((np.asarray(p)-[x0,y0])*scale+[10,75]).astype(int))
        for v in observations(r):
            idx=int(v[0]);p=v[1:3];c1=r['strategy'].startswith('DualRoi_C1') or idx>=10
            color=(0,220,60) if c1 else (50,70,255)
            cv2.circle(canvas,pt(p),4,color,-1,cv2.LINE_AA)
            cv2.putText(canvas,str(idx),tuple(np.array(pt(p))+[4,-4]),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
        if q['success']:
            R=rotation(r);t=np.array([float(r[k]) for k in ('tx','ty','tz')]);length=100 if j['kind']=='joint' else 5
            p=np.array([[0,0,0],[length,0,0],[0,length,0],[0,0,-length]])@R.T+t
            p=p@K.T;p=p[:,:2]/p[:,2:]
            for end,color,label in zip(p[1:],[(0,0,255),(0,255,0),(255,160,0)],['+X','+Y','-Z up']):
                cv2.arrowedLine(canvas,pt(p[0]),pt(end),color,2,cv2.LINE_AA,tipLength=.18)
                cv2.putText(canvas,label,pt(end),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
        title='%s | %s'%(mode,r['strategy'])
        detail='normal %.2fdeg | depth %.2f%%'%(q['normal_error_deg'],q['depth_error']*100) if q['success'] else 'REJECTED: no axes drawn'
        cv2.putText(canvas,title,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.56,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(canvas,detail,(10,49),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(canvas,'%s frame=%d | tilt GT %.2fdeg'%(j['sequence'],j['frame'],j['tilt']),(10,725),cv2.FONT_HERSHEY_SIMPLEX,.43,(230,230,230),1)
        panels.append(canvas)
    path=dest/(name+'.png');cv2.imwrite(str(path),np.hstack(panels))
    captures.append(dict(name=name,job=j['job'],sequence=j['sequence'],frame=j['frame'],image=path.as_posix(),
        before=look['before',j['job']],after=look['after',j['job']]))
dump(OUT/'visual_cases.json',captures)
lines=(OUT/'fixed_jobs.txt').read_text().splitlines()
(OUT/'visual_jobs.txt').write_bytes(('\n'.join(lines[j['job']] for _,j in selected)+'\n').encode())
print('Visuals:',[name for name,_ in selected])
