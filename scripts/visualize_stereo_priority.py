"""Measured observations and estimated poses; GT is only used for annotations."""
from pathlib import Path
import csv
import cv2,numpy as np
from experiment_stereo_priority import OUT,read,dump
from analyze_stereo_priority import rows,rotation,vector

def prepare_visuals():
    data=read(OUT/'physical_metrics.json');jobs=read(OUT/'physical_jobs.json')
    by={m:{r['job']:r for r in data if r['mode']==m} for m in ('before','after')}
    selection=[]
    for kind in ('bc','dual','fallback'):
        common=[i for i in by['before'] if jobs[i]['kind']==kind and by['before'][i]['success'] and by['after'][i]['success']]
        chosen=max(common,key=lambda i:by['before'][i]['normal_error']-by['after'][i]['normal_error'])
        selection.append(dict(name=kind+'_improved',job=chosen))
    tt=[i for i in by['after'] if jobs[i]['kind']=='tt' and by['after'][i]['success']]
    selection.append(dict(name='tt_normal_limit',job=max(tt,key=lambda i:by['after'][i]['normal_error'])))
    selection.append(dict(name='tt_three_pairs',job=next(i for i in by['after'] if jobs[i]['kind']=='tt' and by['after'][i]['failure']=='insufficient_stereo_pairs')))
    lines=(OUT/'physical_jobs.txt').read_text().splitlines()
    (OUT/'visual_jobs.txt').write_bytes(('\n'.join(lines[j['job']] for j in selection)+'\n').encode())
    dump(OUT/'visual_cases.json',selection)
def draw_case(row,job,metrics):
    images=[cv2.imread(job[k]) for k in ('left','right')]
    axis_pixels=[None,None]
    R=rotation(row);t=np.array([float(row[k]) for k in ('tx','ty','tz')]);tiny='TT' in row['strategy'] or row['strategy']=='TinyTarget'
    if int(row['success']):
        length=5 if int(row['class1']) else 75
        axis=np.array([[0,0,0],[length,0,0],[0,length,0],[0,0,-length]],float)
        K=np.array([[900.,0,400],[0,900.,400],[0,0,1]])
        for view in (0,1):
            q=axis@R.T+t-np.array([60*view,0,0]);u=q@K.T;u=u[:,:2]/u[:,2:]
            axis_pixels[view]=u
    transforms=[(np.zeros(2),np.array([.75,.75])) for _ in range(2)]
    if tiny or int(row['class1']):
        cropped=[]
        for view in (0,1):
            box=job[('right_secondary' if view else 'secondary') if int(row['class1']) else ('right_primary' if view else 'primary')]
            limits=np.array([[box[0],box[1]],[box[0]+box[2],box[1]+box[3]]],float)
            if axis_pixels[view] is not None:limits=np.vstack([limits,axis_pixels[view]])
            lo=limits.min(0)-20;hi=limits.max(0)+20;c=(lo+hi)/2;side=max(hi-lo)
            x,y=np.maximum(0,np.floor(c-side/2)).astype(int);xx,yy=np.minimum(800,np.ceil(c+side/2)).astype(int)
            transforms[view]=(np.array([x,y]),np.array([600./(xx-x),600./(yy-y)]))
            cropped.append(cv2.resize(images[view][y:yy,x:xx],(600,600),interpolation=cv2.INTER_NEAREST))
        images=cropped
    else:images=[cv2.resize(im,(600,600)) for im in images]
    for view in (0,1):
        origin,scale=transforms[view]
        for values in vector(row['pairs']):
            idx=int(values[0]);c=(70,255,70) if int(row['class1']) or idx>=10 else (0,210,255)
            p=np.round((np.array(values[1+view*2:3+view*2])-origin)*scale).astype(int)
            cv2.circle(images[view],tuple(p),4,c,1,cv2.LINE_AA)
            cv2.putText(images[view],str(idx),tuple(p+[5,-5]),cv2.FONT_HERSHEY_SIMPLEX,.43,c,1,cv2.LINE_AA)
        if axis_pixels[view] is not None:
            uv=np.round((axis_pixels[view]-origin)*scale).astype(int)
            for k,c in enumerate(((0,0,255),(0,255,0),(255,0,0)),1):
                cv2.arrowedLine(images[view],tuple(uv[0]),tuple(uv[k]),c,2,cv2.LINE_AA,tipLength=.15)
                cv2.putText(images[view],('+X','+Y','-Z up')[k-1],tuple(uv[k]+[4,-4]),cv2.FONT_HERSHEY_SIMPLEX,.4,c,1,cv2.LINE_AA)
    joined=np.hstack(images)
    joined=cv2.copyMakeBorder(joined,78,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
    text='%s | %s | actual L/R pairs=%d | %s'%(metrics['mode'],row['strategy'],len(vector(row['pairs'])),row['source'])
    cv2.putText(joined,text,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
    text=('normal error=%.2f deg depth error=%.2f%%'%(metrics['normal_error'],100*metrics['depth_relative'])) if int(row['success']) else 'REJECTED: '+row['failure']
    cv2.putText(joined,text,(8,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(joined,'left camera / right camera; emitted TT vertices are virtual square references' if tiny else 'left camera / right camera; labels are template IDs',(8,69),cv2.FONT_HERSHEY_SIMPLEX,.42,(210,210,210),1)
    return joined
def make():
    dest=OUT/'visuals';dest.mkdir(exist_ok=True);jobs=read(OUT/'physical_jobs.json');metrics=read(OUT/'physical_metrics.json')
    by={m:{int(r['job']):r for r in rows(OUT/('physical_'+m+'.csv'))} for m in ('before','after')}
    mm={m:{r['job']:r for r in metrics if r['mode']==m} for m in by}
    for case in read(OUT/'visual_cases.json'):
        i=case['job'];panels=[draw_case(by[m][i],jobs[i],mm[m][i]) for m in by]
        cv2.imwrite(str(dest/(case['name']+'.png')),np.vstack(panels))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,12))
    for row,kind in enumerate(('bc','tt','dual','fallback')):
        jj=[j for j in jobs if j['kind']==kind];xx=[j['frame'] for j in jj]
        axes[row,0].plot(xx,[j['t'][2] for j in jj],color='black',label='GT')
        for m,c in (('before','tab:orange'),('after','tab:blue')):
            zz=[mm[m][j['job']].get('t',[0,0,np.nan])[2] for j in jj]
            ee=[mm[m][j['job']].get('normal_error',np.nan) for j in jj]
            axes[row,0].plot(xx,zz,'.-',label=m,color=c);axes[row,1].plot(xx,ee,'.-',label=m,color=c)
        axes[row,0].set_title(kind+' / Z (mm)');axes[row,1].set_title(kind+' / normal error (deg)')
        for ax in axes[row]:ax.grid(alpha=.25);ax.legend();ax.set_xlabel('frame; gaps indicate rejection')
    fig.tight_layout();fig.savefig(dest/'z_and_normal.png',dpi=150);plt.close(fig)
if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare_visuals()
    else:make()
