#!/usr/bin/env python3
"""Freeze mono experiment configs or analyze full-precision pose CSV against GT.

python scripts/analyze_mono_robustness.py prepare|analyze --dataset ... --out ...
No change is made to config/tracker_config.json or to pose estimation algorithms.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def prepare(args):
    dataset,out=Path(args.dataset),Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"configs").mkdir(exist_ok=True)
    config=read("config/tracker_config.json")
    profiles=read("config/experiments/mono_robustness_suite.json")["profiles"]
    jobs=[(p["name"],p["name"],True) for p in profiles]
    jobs += [(n+"_temporal_off",n,False) for n in ("normal_ambiguity","oblique_clean")]
    frozen=[]
    for name,sequence,temporal in jobs:
        cfg=json.loads(json.dumps(config))
        cfg["mode"]="normal"
        cfg["mono_mode"]=True
        # This historical broad-tilt benchmark includes 75deg views, outside the
        # later near-nadir prior. Keep its baseline reproducible and unconstrained.
        cfg["mono_pose_prior"]=dict(enabled=False,max_tilt_deg=30.0)
        cfg["temporal"]["enabled"]=temporal
        cfg["stereo_frontend"].update(method="dual_bc",dual_roi_flow=False,disparity_only=False,disparity_full_chain=False)
        cfg["input_system"]["max_frames"]=0
        cfg["input_system"]["use_threaded_capture"]=False
        cfg["input_system"]["image"].update(type="sequence",directory_path=(dataset/sequence).as_posix(),sequence_pattern="frame")
        cfg["input_system"]["imu"]["enabled"]=False
        cfg["input_system"]["altimeter"]["enabled"]=False
        cfg["output"]=dict(visualize=False,log_file=True,pose_csv=True,directory=(out/"runs"/name).as_posix())
        cfg["manual_roi"]["enabled"]=False
        cfg.pop("fusion",None)
        cfg.pop("eskf",None)
        path=out/"configs"/(name+".json")
        dump(path,cfg)
        frozen.append(dict(name=name,sequence=sequence,config=path.as_posix(),sha256=sha(path),temporal=temporal))
    try:
        head=subprocess.check_output(["git","-c","safe.directory="+Path.cwd().as_posix(),"rev-parse","HEAD"],text=True).strip()
    except Exception: head="unknown"
    dump(out/"experiment.json",dict(dataset=dataset.as_posix(),jobs=frozen,head=head,source_config_sha256=sha("config/tracker_config.json"),
                                   model_sha256=sha(config["yolo"]["model_path"]),binary_sha256=sha("build-codex-bc-bench/GPNP"),
                                   diagnostic_only=True,notes="Existing uncommitted algorithm edits retained; only opt-in CSV export added for this experiment."))
    script=["#!/usr/bin/env bash","set -euo pipefail","mkdir -p '"+out.as_posix()+"/runs'"]
    for j in frozen:
        script += ["echo 'RUN "+j["name"]+"'", "./build-codex-bc-bench/GPNP '"+j["config"]+"' > '"+(out/"runs"/(j["name"]+"_stdout.log")).as_posix()+"' 2>&1"]
    with (out/"run_all.sh").open("w",encoding="utf-8",newline="\n") as f:
        f.write("\n".join(script)+"\n")
    print(f"Prepared {len(frozen)} mono jobs at {out}")


def angle(a,b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30),-1,1))))


def rotangle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))


def arrow(K,R,t,length=100.):
    points=np.array([t,t+R[:,2]*length])@K.T
    return points[1,:2]/points[1,2]-points[0,:2]/points[0,2]


def stats(values):
    v=np.asarray(values,dtype=float)
    v=v[np.isfinite(v)]
    if not len(v): return dict(n=0,median=None,p95=None,max=None)
    return dict(n=len(v),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(v.max()))


def validate_and_oracle(args):
    from generate_synthetic_dataset import load_points, project_points
    dataset,out=Path(args.dataset),Path(args.out)
    suite=read(dataset/"suite_manifest.json")
    K=np.array(suite["K"]); width,height=suite["target_size_px"]
    scale=np.array(suite["board_size_mm"])/[width,height]
    points=np.array(load_points(suite["config"]["class0_points"]))
    center=(points.min(axis=0)+points.max(axis=0))/2
    obj=np.column_stack([(points-center)*scale,np.zeros(len(points))])
    bbox=np.array([[points[:,0].min(),points[:,1].min()],[points[:,0].max(),points[:,1].min()],
                   [points[:,0].max(),points[:,1].max()],[points[:,0].min(),points[:,1].max()]])
    square=np.column_stack([(bbox-center)*scale,np.zeros(4)])
    rng=np.random.default_rng(20261015)
    maxerror=0.; mindepth=1e30; total=0; counts={}
    (out/"analysis").mkdir(exist_ok=True)
    with (out/"oracle_input.txt").open("w",encoding="ascii",newline="\n") as f:
        for profile in suite["profiles"]:
            manifest=read(dataset/profile["name"]/"manifest.json")
            assert len(list((dataset/profile["name"]).glob("frame_*.png")))==profile["frames"]
            counts[profile["name"]]=len(manifest["frames"])
            for gt in manifest["frames"]:
                R=np.array(gt["R"]); t=np.array(gt["t_mm"])
                assert np.linalg.norm(R.T@R-np.eye(3))<1e-12 and abs(np.linalg.det(R)-1)<1e-12
                proj=(obj@R.T+t)@K.T; proj=proj[:,:2]/proj[:,2:]
                maxerror=max(maxerror,float(np.max(np.linalg.norm(proj-np.array(gt["class0"]["points_px"]),axis=1))))
                mindepth=min(mindepth,gt["min_board_depth_mm"])
                assert gt["min_board_depth_mm"]>=suite["config"]["minimum_board_depth_mm"]-1e-8
                assert gt["normal"][2]>0  # One-sided board, no physically rendered backside.
                total+=1
                if profile["name"]=="oblique_degraded": continue  # Same geometry as oblique_clean.
                for model,tag in [(obj,"c0_10"),(square,"square4")]:
                    uv=(model@R.T+t)@K.T; uv=uv[:,:2]/uv[:,2:]
                    if not np.all((uv>=0)&(uv<np.array(suite["config"]["canvas"]))): continue
                    for noise in (0.,.25):
                        xy=uv+rng.normal(0,noise,uv.shape)
                        header=[profile["name"],("exact_" if noise==0 else "noise025_")+tag,str(gt["frame"]),str(len(model)),
                                *map(str,[K[0,0],K[1,1],K[0,2],K[1,2]]),*map(str,R.flatten()),*map(str,t)]
                        values=np.column_stack([model,xy]).flatten()
                        f.write(" ".join(header)+" "+" ".join(map(str,values))+"\n")
    clean=read(dataset/"oblique_clean"/"manifest.json")["frames"]
    degraded=read(dataset/"oblique_degraded"/"manifest.json")["frames"]
    assert all(a["R"]==b["R"] and a["t_mm"]==b["t_mm"] and a["H"]==b["H"] for a,b in zip(clean,degraded))
    assert maxerror<1e-7
    validation=dict(frames=total,counts=counts,max_projection_error_px=maxerror,min_board_depth_mm=mindepth,
                    paired_geometry_identical=True,source_manifest_sha256=sha(dataset/"suite_manifest.json"),
                    python_opencv=cv2.__version__,numpy=np.__version__,
                    hashes={str(p):sha(p) for p in [Path("main.cpp"),Path("src/pose/MonoPnPSolver.cpp"),Path("src/tracker/MonoTracker.cpp"),
                                                    Path("scripts/generate_approach_sequence.py"),Path("scripts/generate_mono_robustness_suite.py"),
                                                    Path("config/experiments/mono_robustness_suite.json")]})
    dump(out/"analysis"/"dataset_validation.json",validation)
    print(json.dumps(validation,indent=2))


def analyze_job(job,dataset,out):
    manifest=read(dataset/job["sequence"]/"manifest.json")
    K=np.array(manifest["K"])
    truths=manifest["frames"]
    path=out/"runs"/job["name"]/"poses.csv"
    rows=list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows)!=len(truths) or [int(r["frame"]) for r in rows]!=list(range(1,len(truths)+1)):
        raise ValueError(f"Missing/duplicated/reordered frame rows: {path}: {len(rows)} vs {len(truths)}")
    measurements=[]
    last=None
    for i,(row,gt) in enumerate(zip(rows,truths)):
        rec=dict(sequence=job["name"],frame=i,tracker_frame=i+1,success=int(row["success"]),strategy=row["strategy"],
                 is_class1=int(row["is_class1"]),gate=int(row["gate_status"]),warm=int(row["warm_start"]),
                 n_matched=int(row["n_matched"]),fallback=int(row["fallback"]),
                 extract_ms=float(row["extract_ms"]),pnp_ms=float(row["pnp_ms"]),
                 tilt_gt=gt["tilt_deg"],size_gt=gt["effective_size_px"],
                 c0_visible=sum(gt["class0"]["visible"]),c1_visible=sum(gt["class1"]["visible"]),
                 z_jump=0,gap_jump=0,arrow_flip=0,short_arrow_flip=0,strategy_switch=0,depth_jump=0)
        if rec["success"]:
            R=np.array([[float(row[f"R{a}{b}"]) for b in range(3)] for a in range(3)])
            t=np.array([float(row[k]) for k in ("tx","ty","tz")])
            Rg,tg=np.array(gt["R"]),np.array(gt["t_mm"])
            if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)) or np.linalg.norm(R.T@R-np.eye(3))>1e-5 or abs(np.linalg.det(R)-1)>1e-5:
                raise ValueError(f"Invalid rotation {job['name']}:{i}")
            n=R[:,2]
            a=arrow(K,R,t)
            ag=arrow(K,Rg,tg)
            rec.update(normal_error_deg=angle(n,Rg[:,2]),rotation_error_deg=rotangle(R@Rg.T),
                       depth_mm=float(t[2]),depth_gt_mm=float(tg[2]),depth_rel_error=float(abs(t[2]-tg[2])/tg[2]),
                       arrow_length_px=float(np.linalg.norm(a)),arrow_gt_length_px=float(np.linalg.norm(ag)),
                       arrow_error_deg=angle(a,ag),normal_x=float(n[0]),normal_y=float(n[1]),normal_z=float(n[2]))
            if last:
                prev,pr,pgt,pR,pa,pt=last
                gap=i-prev
                step=angle(n,pR[:,2]); realstep=angle(Rg[:,2],np.array(pgt["R"])[:,2])
                rec.update(gap=gap,normal_step_deg=step,normal_gt_step_deg=realstep,
                           normal_step_excess_deg=step-realstep,rotation_step_deg=rotangle(R@pR.T),
                           depth_step_ratio=float(abs(t[2]-pt[2])/pt[2]),
                           depth_gt_step_ratio=float(abs(tg[2]-pgt["t_mm"][2])/pgt["t_mm"][2]),
                           strategy_switch=int(pr["strategy"]!=rec["strategy"] or pr["is_class1"]!=rec["is_class1"]))
                anomaly=step>20 and step-realstep>15
                rec["z_jump"]=int(gap==1 and anomaly)
                rec["gap_jump"]=int(gap>1 and anomaly)
                rec["depth_jump"]=int(gap==1 and rec["depth_step_ratio"]>.3
                                      and rec["depth_step_ratio"]-rec["depth_gt_step_ratio"]>.2)
                arrow_step=angle(a,pa)
                short=min(np.linalg.norm(a),np.linalg.norm(pa))<5
                rec.update(arrow_step_deg=arrow_step)
                rec["arrow_flip"]=int(gap==1 and arrow_step>90 and not short)
                rec["short_arrow_flip"]=int(gap==1 and arrow_step>90 and short)
            last=(i,rec,gt,R,a,t)
        measurements.append(rec)
    valid=[r for r in measurements if r["success"]]
    adjacent=[r for r in valid if r.get("gap")==1]
    groups=defaultdict(list)
    for r in valid: groups[r["strategy"]].append(r)
    stdout=(out/"runs"/(job["name"]+"_stdout.log")).read_text(encoding="utf-8",errors="replace")
    no_detection=set(map(int,re.findall(r"\[Frame (\d+)\] YOLO未检测到目标",stdout)))
    summary=dict(name=job["name"],sequence=job["sequence"],frames=len(rows),success=len(valid),success_rate=len(valid)/len(rows),
                 consecutive_success_pairs=len(adjacent),z_jumps=sum(r["z_jump"] for r in valid),
                 gap_jumps=sum(r["gap_jump"] for r in valid),arrow_flips=sum(r["arrow_flip"] for r in valid),
                 short_arrow_flips=sum(r["short_arrow_flip"] for r in valid),
                 normal_error_deg=stats([r["normal_error_deg"] for r in valid]),
                 rotation_error_deg=stats([r["rotation_error_deg"] for r in valid]),
                 depth_rel_error=stats([r["depth_rel_error"] for r in valid]),
                 gt_normal_step_deg=stats([angle(truths[i]["normal"],truths[i-1]["normal"]) for i in range(1,len(truths))]),
                 normal_bad20=sum(r["normal_error_deg"]>20 for r in valid),normal_bad45=sum(r["normal_error_deg"]>45 for r in valid),
                 normal_good10=sum(r["normal_error_deg"]<=10 for r in valid),
                 depth_jumps=sum(r["depth_jump"] for r in valid),
                 normal_backwards=sum(r["normal_error_deg"]>90 for r in valid),
                 recovered_frames=sum(r["gate"]==2 for r in valid),
                 recovered_z_jumps=sum(r["gate"]==2 and r["z_jump"] for r in valid),
                 no_detection_frames=len(no_detection),algorithm_failed_frames=len(rows)-len(valid)-len(no_detection),
                 switch_z_jumps=sum(r["strategy_switch"] and r["z_jump"] for r in valid),
                 full_c0_frames=sum(sum(g["class0"]["visible"])==10 for g in truths),
                 depth_capped_frames=sum(g["depth_capped"] for g in truths),
                 strategy_counts=dict(Counter(r["strategy"] for r in valid)),
                 per_strategy={k:dict(success=len(v),z_jumps=sum(r["z_jump"] for r in v),normal_error_deg=stats([r["normal_error_deg"] for r in v])) for k,v in groups.items()})
    return manifest,measurements,summary


def oracle_summary(args):
    out=Path(args.out)
    groups=defaultdict(list)
    with (out/"analysis"/"oracle_poses.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f): groups[(row["sequence"],row["condition"],row["temporal"])].append(row)
    summaries=[]
    for (sequence,condition,temporal),rows in groups.items():
        valid=[r for r in rows if r["success"]=="1"]
        summaries.append(dict(sequence=sequence,condition=condition,temporal=int(temporal),frames=len(rows),success=len(valid),
                              normal_error_deg=stats([float(r["normal_error_deg"]) for r in valid]),
                              z_jumps=sum(r["z_jump"]=="1" for r in valid)))
    dump(out/"analysis"/"oracle_summary.json",summaries)
    print(f"Summarized {sum(len(v) for v in groups.values())} oracle solver calls")


def visualize(experiment,results,out,dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(6,2,figsize=(15,18),constrained_layout=True)
    for index,job in enumerate(experiment["jobs"][:6]):
        manifest,rows,summary=results[job["name"]]
        frames=np.arange(len(rows))
        for label,color in [("normal_error_deg","tab:blue"),("normal_step_deg","tab:red")]:
            v=[r.get(label,np.nan) for r in rows]
            axes[index,0].plot(frames,v,label=label,lw=1,color=color)
        axes[index,0].plot(frames,[g["tilt_deg"] for g in manifest["frames"]],label="GT tilt",color="gray",lw=.7)
        for r in rows:
            if r["z_jump"]: axes[index,0].axvline(r["frame"],color="red",alpha=.3,lw=.6)
        axes[index,0].set(title=job["name"]+f" ({summary['success']}/{summary['frames']})",ylabel="angle (deg)",ylim=(0,185))
        axes[index,0].legend(fontsize=7)
        for component,color in [("x","r"),("y","g"),("z","b")]:
            axes[index,1].plot(frames,[r.get("normal_"+component,np.nan) for r in rows],color=color,lw=1,label="est n"+component)
            j="xyz".index(component)
            axes[index,1].plot(frames,[g["normal"][j] for g in manifest["frames"]],color=color,ls="--",lw=.7)
        axes[index,1].set(ylabel="Z axis in camera (unit vector)",ylim=(-1.1,1.1))
        axes[index,1].legend(fontsize=7)
    fig.savefig(out/"figures"/"normal_traces.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for index,name in enumerate(["normal_ambiguity","oblique_clean"]):
        for col,suffix in enumerate(["","_temporal_off"]):
            _,rows,summary=results[name+suffix]
            ax=axes[index,col]
            ax.plot([r.get("normal_error_deg",np.nan) for r in rows],lw=1,label="normal error")
            ax.axhline(20,color="gray",ls="--",lw=.8,label="20 deg")
            for r in rows:
                if r["z_jump"]: ax.axvline(r["frame"],color="red",alpha=.25,lw=.7)
            ax.set(title=f"{name} temporal={'ON' if not suffix else 'OFF'} jumps={summary['z_jumps']}",
                   xlabel="frame (0 based)",ylabel="normal error (deg)",ylim=(0,95))
            ax.legend(fontsize=8)
    fig.savefig(out/"figures"/"temporal_comparison.png",dpi=150)
    plt.close(fig)
    # Each case shows three adjacent actual frames. Solid=estimated, dashed=GT.
    selected=[]
    for job in experiment["jobs"][:6]:
        manifest,rows,_=results[job["name"]]
        jumps=[r for r in rows if r["z_jump"]]
        if jumps:
            selected.append((job,max(jumps,key=lambda r:r["normal_step_deg"])["frame"]))
        else:
            good=[r for r in rows if r["success"]]
            if good: selected.append((job,max(good,key=lambda r:r["normal_error_deg"])["frame"]))
    for job,index in selected:
        manifest,rows,_=results[job["name"]]
        K=np.array(manifest["K"])
        raw=list(csv.DictReader((out/"runs"/job["name"]/"poses.csv").open(encoding="utf-8")))
        panels=[]
        for i in range(max(0,index-1),min(len(rows),index+2)):
            gt=manifest["frames"][i]; row=raw[i]; measure=rows[i]
            image=cv2.imread(str(dataset/job["sequence"]/gt["path"]))
            def draw_axes(R,t,gt=False):
                pts=np.array([t,t+R[:,0]*100,t+R[:,1]*100,t+R[:,2]*100])@K.T
                pts=pts[:,:2]/pts[:,2:]
                # OpenCV clips huge offscreen lines but protect integer conversion.
                pts=np.clip(pts,-20000,20000).astype(int)
                for j,color in enumerate([(0,0,255),(0,255,0),(255,0,0)]):
                    if gt:
                        for u in np.arange(0,1,.12):
                            a=(pts[0]*(1-u)+pts[j+1]*u).astype(int)
                            b=(pts[0]*(1-min(u+.06,1))+pts[j+1]*min(u+.06,1)).astype(int)
                            cv2.line(image,tuple(a),tuple(b),color,2,cv2.LINE_AA)
                    else: cv2.arrowedLine(image,tuple(pts[0]),tuple(pts[j+1]),color,4,cv2.LINE_AA,tipLength=.12)
            draw_axes(np.array(gt["R"]),np.array(gt["t_mm"]),True)
            if measure["success"]:
                R=np.array([[float(row[f"R{a}{b}"]) for b in range(3)] for a in range(3)])
                t=np.array([float(row[k]) for k in ("tx","ty","tz")])
                draw_axes(R,t)
            # Zoom the diagnostic panel around the GT target; full images remain
            # untouched in the dataset. This keeps 20px targets inspectable.
            corners=np.array(gt["class0"]["points_px"])
            lo,hi=corners.min(axis=0),corners.max(axis=0)
            pad=max(70.,float((hi-lo).max())*.4)
            x1,y1=np.maximum(0,lo-pad).astype(int)
            x2,y2=np.minimum(np.array(manifest["config"]["canvas"]),hi+pad).astype(int)
            if x2>x1 and y2>y1: image=image[y1:y2,x1:x2]
            factor=min(680/image.shape[1],680/image.shape[0])
            image=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
            dh,dw=680-image.shape[0],680-image.shape[1]
            image=cv2.copyMakeBorder(image,dh//2,dh-dh//2,dw//2,dw-dw//2,cv2.BORDER_CONSTANT,value=(30,30,30))
            image=cv2.copyMakeBorder(image,86,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
            title=f"{job['name']} frame={i} {row['strategy']}"
            cv2.putText(image,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
            text=f"normal err={measure.get('normal_error_deg',float('nan')):.1f} step={measure.get('normal_step_deg',float('nan')):.1f} gate={measure['gate']}"
            cv2.putText(image,text,(8,48),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(image,"solid=estimate dashed=GT; X red Y green Z blue (crop)",(8,72),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            panels.append(image)
        cv2.imwrite(str(out/"figures"/(job["name"]+"_case.jpg")),cv2.hconcat(panels))
    strips=[]
    for job in experiment["jobs"][:6]:
        manifest,_,_=results[job["name"]]
        panels=[]
        for index in [min(30,len(manifest["frames"])-1),len(manifest["frames"])//2,len(manifest["frames"])-1]:
            gt=manifest["frames"][index]
            image=cv2.imread(str(dataset/job["sequence"]/gt["path"]))
            image=cv2.resize(image,(400,400),interpolation=cv2.INTER_AREA)
            image=cv2.copyMakeBorder(image,42,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
            cv2.putText(image,f"{job['name']} frame={index}",(6,17),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(image,f"tilt={gt['tilt_deg']:.1f} size={gt['effective_size_px']:.1f}px",(6,35),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            panels.append(image)
        strips.append(cv2.hconcat(panels))
    cv2.imwrite(str(out/"figures"/"dataset_overview.jpg"),cv2.vconcat(strips))


def analyze(args):
    out,dataset=Path(args.out),Path(args.dataset)
    (out/"analysis").mkdir(exist_ok=True)
    (out/"figures").mkdir(exist_ok=True)
    experiment=read(out/"experiment.json")
    results={}
    for job in experiment["jobs"]:
        results[job["name"]]=analyze_job(job,dataset,out)
    allrows=[r for _,rows,_ in results.values() for r in rows]
    fields=sorted(set(k for r in allrows for k in r))
    with (out/"analysis"/"frame_metrics.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(allrows)
    summaries=[s for _,_,s in results.values()]
    dump(out/"analysis"/"summary.json",summaries)
    dump(out/"analysis"/"anomalies.json",[r for r in allrows if r["z_jump"] or r["gap_jump"] or r["depth_jump"]])
    visualize(experiment,results,out,dataset)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("action",choices=["prepare","analyze","validate","oracle-summary"])
    p.add_argument("--dataset",default="tests/data/mono_robustness_20261015")
    p.add_argument("--out",default="output/mono_robustness_20261015")
    a=p.parse_args()
    {"prepare":prepare,"analyze":analyze,"validate":validate_and_oracle,"oracle-summary":oracle_summary}[a.action](a)
