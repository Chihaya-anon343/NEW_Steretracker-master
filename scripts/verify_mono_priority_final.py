"""Serial timing, final-binary numerical checks, and actual debug panels."""
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path("output/mono_priority_20261019")

def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def dump(path,value):Path(path).write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(binary,config,name):
    path=ROOT/"final_configs"/(name+".json");dump(path,config)
    log=ROOT/"final_runs"/(name+"_stdout.log")
    started=time.perf_counter()
    with log.open("w")as stream:subprocess.run([str(binary),str(path)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    return log,time.perf_counter()-started

def check_numeric(csv_path,reference):
    actual=list(csv.DictReader(csv_path.open()));expected=list(csv.DictReader(reference.open()))
    assert len(actual)==len(expected)
    max_difference=0.
    for a,b in zip(actual,expected):
        for key in("frame","success","strategy","n_matched"):assert a[key]==b[key],(csv_path,key,a[key],b[key])
        if a["success"]=="1":
            for key in("tx","ty","tz",*["R%d%d"%(i,j)for i in range(3)for j in range(3)]):
                difference=abs(float(a[key])-float(b[key]));max_difference=max(max_difference,difference)
                assert difference<1e-7,(csv_path,key,difference)
    return dict(frames=len(actual),max_numeric_difference=max_difference)

def main():
    for folder in("final_configs","final_runs","visuals"):(ROOT/folder).mkdir(exist_ok=True)
    binary=Path("build-codex-bc-bench/GPNP")
    jobs=read(ROOT/"experiment.json")["jobs"];timing=[];verification=[]
    finish="--finish" in sys.argv
    saved=read(ROOT/"pipeline_timing.json") if finish else []
    # One process at a time; image IO/model startup excluded from stage means.
    for mode in("baseline","roi","quality","partial"):
        original=next(j for j in jobs if j["sequence"]=="reference"and j["mode"]==mode)
        cfg=read(original["config"]);name="serial_reference_"+mode
        cfg["output"]["directory"]=(ROOT/"final_runs"/name).as_posix()
        selected=ROOT/"GPNP_before"if mode=="baseline"else binary
        if finish:
            stored=next(r for r in saved if r["mode"]==mode)
            assert stored["binary_sha256"]==sha(selected)
            timing.append(stored)
            if mode!="baseline":verification.append(dict(name=name,**check_numeric(ROOT/"final_runs"/name/"poses.csv",ROOT/"runs"/original["name"]/"poses.csv")))
            continue
        print("RUN",name,flush=True);log,wall=run(selected,cfg,name)
        text=log.read_text(errors="replace");metrics={}
        for key,label in(("yolo_ms","YOLO 平均用时"),("track_ms","单帧跟踪平均用时"),("extract_ms","特征提取平均用时"),("pnp_ms","位姿解算平均用时")):
            match=re.search(re.escape(label)+r"[^:\n]*:\s*([\d.]+) ms/帧\s*\(共 (\d+) 帧\)",text)
            assert match,(label,log);metrics[key]=float(match[1]);metrics[key+"_calls"]=int(match[2])
        timing.append(dict(mode=mode,process_wall_s=wall,binary_sha256=sha(selected),**metrics))
        if mode!="baseline":verification.append(dict(name=name,**check_numeric(ROOT/"final_runs"/name/"poses.csv",ROOT/"runs"/original["name"]/"poses.csv")))
    dump(ROOT/"pipeline_timing.json",timing)
    original=next(j for j in jobs if j["sequence"]=="oblique_degraded"and j["mode"]=="partial")
    cfg=read(original["config"]);name="final_oblique_degraded_partial";cfg["output"]["directory"]=(ROOT/"final_runs"/name).as_posix()
    if not finish:
        print("RUN",name,flush=True);run(binary,cfg,name)
    verification.append(dict(name=name,**check_numeric(ROOT/"final_runs"/name/"poses.csv",ROOT/"runs"/original["name"]/"poses.csv")))
    base=read("config/tracker_config.json")
    for name,sequence,frame,manual in(("roi_recovery","oblique_degraded",75,None),("depth_recovery","oblique_degraded",149,None),("eight_corners","normal_ambiguity",173,[855,856,27,33])):
        cfg=json.loads(json.dumps(base));cfg["mode"]="debug";cfg["mono_mode"]=True
        path="tests/data/mono_robustness_20261015/%s/frame_%06d.png"%(sequence,frame)
        cfg["input"]["left"]=cfg["input"]["right"]=path
        cfg["manual_roi"]["enabled"]=manual is not None
        if manual:
            cfg["manual_roi"]["left"]=cfg["manual_roi"]["right"]=dict(zip(("x","y","width","height"),manual))
            cfg["strategies"]["tiny_max_area"]=1  # isolate BC/PnP on this small ROI
        cfg["output"]=dict(directory=(ROOT/"visuals"/name).as_posix(),visualize=True,log_file=True,pose_csv=True)
        print("RUN debug",name,flush=True);run(binary,cfg,"debug_"+name)
        rows=list(csv.DictReader((ROOT/"visuals"/name/"poses.csv").open()));assert len(rows)==2
        assert rows[0]["success"]=="1",(name,rows[0])
        if manual:assert rows[0]["n_matched"]=="8",rows[0]
        verification.append(dict(name=name,first_frame_success=int(rows[0]["success"]),matched=int(rows[0]["n_matched"]),manual_roi=manual is not None))
    dump(ROOT/"final_numerical_verification.json",dict(binary_sha256=sha(binary),checks=verification))
    print(json.dumps(timing+verification,indent=2),flush=True)

if __name__=="__main__":main()
