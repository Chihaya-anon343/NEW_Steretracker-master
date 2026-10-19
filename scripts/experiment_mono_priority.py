"""Reproducible ROI / reliable corners / partial BC ablation; GT only in reports."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT=Path("output/mono_priority_20261019")
DATA=Path("tests/data/mono_robustness_20261015")
MODES=("baseline","roi","quality","partial")

def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def dump(path,value):Path(path).write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare():
    if (OUT/"completion.json").exists():
        raise RuntimeError("Completed experiment is archived; use a new output directory before preparing again")
    base=read(OUT/"before/config/tracker_config.json")
    latest=read("config/tracker_config.json")
    shutil.copyfile("build-codex-bc-bench/GPNP",OUT/"GPNP_after")
    for folder in ("configs","runs"): (OUT/folder).mkdir(exist_ok=True)
    jobs=[]
    for profile in read(DATA/"suite_manifest.json")["profiles"]:
        for mode in MODES:
            cfg=json.loads(json.dumps(base));cfg["mode"]="normal";cfg["mono_mode"]=True
            cfg["manual_roi"]["enabled"]=False;cfg["eskf"]={"enabled":False}
            cfg["input_system"]["max_frames"]=0;cfg["input_system"]["use_threaded_capture"]=False
            cfg["input_system"]["image"].update(type="sequence",directory_path=(DATA/profile["name"]).as_posix(),sequence_pattern="frame")
            cfg["roi_selection"]=dict(latest["roi_selection"],enabled=mode!="baseline",verify_mono=mode!="baseline")
            cfg["strategies"]["binary_corner"]["mono_identity"].update(quality_filter=mode in ("quality","partial"),partial_matching=mode=="partial",min_reliable_points=8,max_partial_hypotheses=16,min_model_coverage=.60)
            name=profile["name"]+"_"+mode
            cfg["output"]=dict(visualize=False,log_file=True,pose_csv=True,directory=(OUT/"runs"/name).as_posix())
            path=OUT/"configs"/(name+".json");dump(path,cfg)
            binary=OUT/("GPNP_before" if mode=="baseline" else "GPNP_after")
            jobs.append(dict(name=name,mode=mode,sequence=profile["name"],frames=profile["frames"],config=path.as_posix(),config_sha256=sha(path),binary=binary.as_posix(),binary_sha256=sha(binary)))
    dump(OUT/"experiment.json",dict(dataset=DATA.as_posix(),jobs=jobs))
    points=read("output/mono_normalized_20261016/corner_jobs.json");dump(OUT/"corner_jobs.json",points)
    def line(j):return "%d %s %d 0 %d %d %d %d"%(j["job"],j["path"],j["cls"],*j["roi"])
    (OUT/"corner_jobs.txt").write_bytes(("\n".join(map(line,points))+"\n").encode())
    timing=[j for j in points if (j["cls"]==0 and j["frame"]%15==0)or(j["cls"]==1 and j["frame"]%60==0)]
    (OUT/"timing_jobs.txt").write_bytes(("\n".join(map(line,timing))+"\n").encode())
    print("Prepared",len(jobs),"sequences and",len(points),"fixed ROI jobs")

def run():
    experiment=read(OUT/"experiment.json")
    def one(j):
        assert sha(j["binary"])==j["binary_sha256"] and sha(j["config"])==j["config_sha256"]
        dest=OUT/"runs"/j["name"]/"poses.csv";log=OUT/"runs"/(j["name"]+"_stdout.log")
        completed=False
        if dest.exists() and log.exists():
            rows=list(csv.DictReader(dest.open()))
            completed=len(rows)==j["frames"] and [int(r["frame"]) for r in rows]==list(range(1,j["frames"]+1)) and log.read_text(errors="replace").rstrip().endswith("tracking_log.txt")
        if not completed:
            print("RUN",j["name"],flush=True)
            with log.open("w") as stream:subprocess.run([j["binary"],j["config"]],stdout=stream,stderr=subprocess.STDOUT,check=True)
        rows=list(csv.DictReader(dest.open()));assert len(rows)==j["frames"]
        print("DONE",j["name"],len(rows),flush=True)
        return dict(name=j["name"],rows=len(rows),sha256=sha(dest))
    with ThreadPoolExecutor(max_workers=2)as pool:results=list(pool.map(one,experiment["jobs"]))
    dump(OUT/"completion.json",results)

def analyze():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from analyze_mono_robustness import analyze_job,stats
    frames=[];summaries=[]
    for job in read(OUT/"experiment.json")["jobs"]:
        _,rows,_=analyze_job(job,DATA,OUT)
        raw=list(csv.DictReader((OUT/"runs"/job["name"]/"poses.csv").open()))
        for row,detail in zip(rows,raw):
            row.update(mode=job["mode"],sequence=job["sequence"],failure=detail["failure_reason"],roi_selected=int(detail.get("roi_selected",-1)),bc_rejected=int(detail.get("bc_rejected",0)))
        frames+=rows
    for mode in MODES:
        for scope in ("all","tilt_le30"):
            rows=[r for r in frames if r["mode"]==mode and(scope=="all"or r["tilt_gt"]<=30)];good=[r for r in rows if r["success"]]
            summaries.append(dict(mode=mode,scope=scope,frames=len(rows),success=len(good),normal_jumps=sum(r["z_jump"]for r in rows),depth_jumps=sum(r["depth_jump"]for r in rows),normal_error_deg=stats([r["normal_error_deg"]for r in good]),depth_rel_error=stats([r["depth_rel_error"]for r in good]),failures=dict(Counter(r["failure"]for r in rows if not r["success"])),reselected=sum(r["roi_selected"]>0 for r in rows),quality_rejected=sum(r["bc_rejected"]for r in rows)))
    dump(OUT/"pose_frames.json",frames);dump(OUT/"pose_summary.json",summaries)
    jobs={j["job"]:j for j in read(OUT/"corner_jobs.json")};corners=[];quality=[]
    for r in csv.DictReader((OUT/"corners.csv").open()):
        job=jobs[int(r["job"])];n=int(r["n"]);ids=[int(r["id"+str(i)])for i in range(n)]
        pred=np.array([[float(r["x"+str(i)]),float(r["y"+str(i)])]for i in range(n)]);gt=np.array(job["gt"])
        err=np.linalg.norm(pred-gt[ids],axis=1).tolist()if n else []
        mismatch=False
        if n and len(set(ids))==n:
            rr,cc=linear_sum_assignment(np.linalg.norm(pred[:,None]-gt[None],axis=2))
            mismatch=bool(np.any(cc!=np.array(ids)[rr])and np.mean(err)-np.mean(np.linalg.norm(pred[rr]-gt[cc],axis=1))>.5)
        corners.append(dict(job=job["job"],sequence=job["sequence"],frame=job["frame"],cls=job["cls"],visible=job["fully_visible"],tilt=job["tilt"],mode=r["mode"],success=r["success"]=="1",n=n,ids=ids,errors=err,mismatch=mismatch,status=r["status"],path=r["path"],hypotheses=int(r["hypotheses"])))
    for mode in ("baseline","quality","partial"):
        for scope in("full_visible","tilt_le30"):
            rows=[r for r in corners if r["mode"]==mode and r["cls"]==0 and r["visible"]and(scope!="tilt_le30"or r["tilt"]<=30)];good=[r for r in rows if r["success"]]
            quality.append(dict(mode=mode,scope=scope,attempts=len(rows),success=len(good),partial_outputs=sum(r["n"]<10 for r in good),mismatch_frames=sum(r["mismatch"]for r in good),point_error_px=stats([v for r in good for v in r["errors"]]),all_observed_within2=sum(max(r["errors"])<=2 for r in good),statuses=dict(Counter(r["status"]for r in rows))))
    dump(OUT/"corner_frames.json",corners);dump(OUT/"corner_summary.json",quality)
    print(json.dumps(summaries+quality,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","run","analyze"]);args=p.parse_args();globals()[args.action]()
