"""Prepare physical stereo projections and frozen experiments; GT stays offline."""
import argparse, copy, csv, hashlib, json, re, subprocess
from pathlib import Path

OUT=Path('output/stereo_priority_20261020')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def robust(enabled):return dict(enabled=enabled,binary_corner=True,tiny_target=True,joint_quality_check=True,min_bc_points=6)
def prepare():
    import cv2,numpy as np
    data=OUT/'physical';data.mkdir(exist_ok=True)
    src=Path('data/NewMuBan(reordered)/0_degrees.txt').read_text()
    points=np.array([list(map(float,m)) for m in re.findall(r'Corner_\d+:\s*([-\d.]+),\s*([-\d.]+)',src)])
    center=(points.min(0)+points.max(0))/2
    # Put the small white pattern in the dark opening of class0; coincident
    # centers would paint white on white and make class1 unobservable.
    models=[(points-center)*6.,(points-center)*.46+np.array([85.,0.])]
    # All image geometry, ordinary BC and Dual models share these physical points.
    sf=4;size=400;board=np.full((size*sf,size*sf),35,np.uint8)
    for p in models:cv2.fillPoly(board,[np.round((p+200)*sf).astype(np.int32)],210)
    rng=np.random.RandomState(20261020)
    texture=rng.normal(0,5,board.shape)
    board=np.clip(board.astype(float)+texture,0,255).astype(np.uint8)
    cv2.imwrite(str(data/'board.png'),cv2.resize(board,(400,400),interpolation=cv2.INTER_AREA))
    for cls,p in enumerate(models):
        (data/('class%d.txt'%cls)).write_bytes(('\n'.join('Corner_%d: %.9f, %.9f'%(i,x+200,y+200) for i,(x,y) in enumerate(p))+'\n').encode())
    base=read(OUT/'config_before.json');cfg=copy.deepcopy(base)
    cfg['camera']=dict(fx=900.,fy=900.,cx=400.,cy=400.,baseline_mm=60.)
    cfg['template_geometry']=dict(enabled=True,class1_center_in_class0_mm=[85.,0.])
    cfg['temporal']['enabled']=False
    s=cfg['strategies'];s['board_template']['template_path']=(data/'board.png').as_posix()
    s['tiny_max_area']=4000 # all far-target cases enter TT despite rotated YOLO-style boxes
    s['board_template']['template_real_width_mm']=s['board_template']['template_real_height_mm']=400.
    s['binary_corner']['pixel_to_meter_scale_class0']=.006
    s['binary_corner']['pixel_to_meter_scale_class1']=.00046
    # The TT four points represent the bounding square, not four material vertices.
    s['tiny_target']['square_size_m_class0']=float(np.ptp(models[0][:,0]))/1000
    s['tiny_target']['square_size_m_class1']=float(np.ptp(models[1][:,0]))/1000
    for cls in (0,1):s['dual_roi']['class%d_points_path'%cls]=(data/('class%d.txt'%cls)).as_posix()
    s['dual_roi']['primary_span_ratio']=.98;s['dual_roi']['secondary_expand_pixels']=0
    cfg['stereo_frontend'].update(disparity_only=False,disparity_full_chain=False,dual_roi_legacy_fallback=False,on_failure='reject')
    K=np.array([[900.,0,400],[0,900.,400],[0,0,1]])
    def Rxyz(rx,ry,rz):
        a,b,c=np.radians([rx,ry,rz]);ca,sa=np.cos(a),np.sin(a);cb,sb=np.cos(b),np.sin(b);cc,sc=np.cos(c),np.sin(c)
        return np.array([[cc,-sc,0],[sc,cc,0],[0,0,1]])@np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])@np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
    def project(p,R,t):
        q=np.c_[p,np.zeros(len(p))]@R.T+t;u=q@K.T;return u[:,:2]/u[:,2:]
    def roi(p):
        lo=np.floor(p.min(0)-5).astype(int);hi=np.ceil(p.max(0)+5).astype(int)
        lo=np.maximum(lo,0);hi=np.minimum(hi,800)
        return [int(lo[0]),int(lo[1]),int(max(1,hi[0]-lo[0])),int(max(1,hi[1]-lo[1]))]
    jobs=[];lines=[]
    for kind in ('bc','tt','dual','fallback'):
        for i in range(24):
            z={'bc':1500.,'tt':8500.,'dual':650.,'fallback':850.}[kind]*(1+.18*np.sin(i*.3))
            R=Rxyz(8+25*np.sin(i*.2),12+18*np.cos(i*.22),i*15.)
            t=np.array([25*np.sin(i*.25),20*np.cos(i*.3),z]);tr=t-np.array([60.,0,0])
            world_box=np.array([[-200,-200],[200,-200],[200,200],[-200,200]],float)
            source=np.array([[0,0],[1600,0],[1600,1600],[0,1600]],np.float32)
            images=[];truth=[]
            for view,tv in enumerate((t,tr)):
                uv=project(world_box,R,tv);H=cv2.getPerspectiveTransform(source,uv.astype(np.float32))
                image=cv2.warpPerspective(board,H,(800,800),flags=cv2.INTER_LINEAR,borderValue=35)
                if i%3==1:
                    image=cv2.GaussianBlur(image,(3,3),.6)
                    if view:image=np.clip(image*.85+10,0,255).astype(np.uint8)
                if i%3==2:
                    image=cv2.GaussianBlur(image,(5,5),1.)
                    if view:image=np.clip(image*.75+20+rng.normal(0,2,image.shape),0,255).astype(np.uint8)
                images.append(image);truth.append([project(p,R,tv).tolist() for p in models])
            job=len(jobs);lp=data/('left_%03d.png'%job);rp=data/('right_%03d.png'%job)
            cv2.imwrite(str(lp),images[0]);cv2.imwrite(str(rp),images[1])
            a,b=roi(np.array(truth[0][0])),roi(np.array(truth[1][0]));c,d=roi(np.array(truth[0][1])),roi(np.array(truth[1][1]))
            if kind=='fallback':a=b=[0,0,800,800]
            dual=kind in ('dual','fallback')
            if not dual:c=d=[0,0,0,0]
            item=dict(job=job,kind=kind,frame=i,left=lp.as_posix(),right=rp.as_posix(),R=R.tolist(),t=t.tolist(),
                      primary=a,secondary=c,right_primary=b,right_secondary=d,truth=truth,degradation=i%3,
                      tilt_deg=float(np.degrees(np.arccos(np.clip(R[2,2],-1,1)))))
            jobs.append(item);lines.append(' '.join(map(str,[job,lp.as_posix(),rp.as_posix(),int(dual),0,*a,*c,*b,*d])))
    dump(OUT/'physical_jobs.json',jobs);(OUT/'physical_jobs.txt').write_bytes(('\n'.join(lines)+'\n').encode())
    # Off switch replay must match the frozen implementation exactly.
    for mode in ('before','after','off','no_joint_gate'):
        c=copy.deepcopy(cfg);c['stereo_frontend']['robustness']=robust(mode not in ('before','off'))
        if mode=='no_joint_gate':c['stereo_frontend']['robustness']['joint_quality_check']=False
        dump(OUT/'configs'/('physical_'+mode+'.json'),c)
    for mode in ('offset_gate','offset_open'):
        c=copy.deepcopy(cfg);c['template_geometry']['class1_center_in_class0_mm']=[82.,0.]
        c['stereo_frontend']['robustness']=robust(True)
        c['stereo_frontend']['robustness']['joint_quality_check']=mode=='offset_gate'
        dump(OUT/'configs'/('physical_'+mode+'.json'),c)
    # End-to-end YOLO experiment on existing artificial stereo: disparity only.
    for mode in ('before','after','off'):
        c=copy.deepcopy(base);c['mono_mode']=False;c['mode']='normal';c['manual_roi']['enabled']=False
        c['input_system'].update(max_frames=0,use_threaded_capture=False)
        c['input_system']['image'].update(type='directory',directory_path='data/stereo_crop',left_pattern='left',right_pattern='right')
        c['temporal']['enabled']=False;c['eskf']={'enabled':False}
        c['output']=dict(visualize=False,log_file=True,pose_csv=True,directory=(OUT/'crop'/mode).as_posix())
        c['stereo_frontend']['robustness']=robust(mode=='after')
        c['stereo_frontend'].update(disparity_only=True,disparity_full_chain=True)
        dump(OUT/'configs'/('crop_'+mode+'.json'),c)
    tracked=[Path('config/tracker_config.json'),Path('data/stereo_crop/manifest.json'),*Path('data/stereo_crop').glob('*.png'),
             *data.glob('*'),OUT/'GPNP_before',OUT/'evaluate_before']
    dump(OUT/'provenance.json',dict(hashes={p.as_posix():sha(p) for p in tracked},seed=20261020,
        scope='physical fixed ROI with 5px padding; no GT coordinates or poses enter estimator; crop uses actual YOLO',
        physical_cases=len(jobs),crop_cases=len(read('data/stereo_crop/manifest.json')['frames'])))
    print('prepared',len(jobs),'physical pairs',flush=True)
def run(action):
    if action.startswith('crop'):
        for mode in (('before','after') if action=='crop' else (action.split('_',1)[1],)):
            dest=OUT/'crop'/mode;dest.mkdir(parents=True,exist_ok=True);print('CROP',mode,flush=True)
            with (dest/'console.log').open('w') as f:
                subprocess.run([str(OUT/('GPNP_'+mode)),str(OUT/'configs'/('crop_'+mode+'.json'))],stdout=f,stderr=subprocess.STDOUT,check=True)
    elif action.startswith('physical'):
        for mode in (('before','after','off','no_joint_gate') if action=='physical' else (action.split('_',1)[1],)):
            binary='before' if mode=='before' else 'after';print('PHYSICAL',mode,flush=True)
            with (OUT/('physical_'+mode+'.log')).open('w') as f:
                subprocess.run([str(OUT/('evaluate_'+binary)),str(OUT/'configs'/('physical_'+mode+'.json')),
                                str(OUT/'physical_jobs.txt'),str(OUT/('physical_'+mode+'.csv')),'1'],stdout=f,stderr=subprocess.STDOUT,check=True)
    elif action=='offset':
        lines=(OUT/'physical_jobs.txt').read_text().splitlines()
        (OUT/'offset_jobs.txt').write_text('\n'.join(lines[48:72])+'\n')
        for mode in ('offset_gate','offset_open'):
            with (OUT/(mode+'.log')).open('w') as f:
                subprocess.run([str(OUT/'evaluate_after'),str(OUT/'configs'/('physical_'+mode+'.json')),
                                str(OUT/'offset_jobs.txt'),str(OUT/(mode+'.csv')),'1'],stdout=f,stderr=subprocess.STDOUT,check=True)
    elif action=='timing':
        for turn in range(3):
            for mode in (('before','after') if turn%2==0 else ('after','before')):
                print('TIMING',turn,mode,flush=True)
                with (OUT/('timing_%s_%d.log'%(mode,turn))).open('w') as f:
                    subprocess.run([str(OUT/('evaluate_'+mode)),str(OUT/'configs'/('physical_'+mode+'.json')),
                        str(OUT/'timing_jobs.txt'),str(OUT/('timing_%s_%d.csv'%(mode,turn))),'5'],stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','physical','physical_before','physical_after','physical_off','physical_no_joint_gate','crop','crop_before','crop_after','offset','timing']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.action)
