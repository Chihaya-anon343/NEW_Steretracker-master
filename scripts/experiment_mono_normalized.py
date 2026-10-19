#!/usr/bin/env python3
"""Frozen A/B/C experiment: current BC, normalized, normalized+original edge fit.

GT enters only ROI preparation and offline evaluation, never feature extraction.
Run each generated shell script through the project's Docker toolchain.
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from analyze_mono_robustness import read, dump, sha, analyze_job, stats

MODES={"current":dict(frontend="legacy",refine_original=False),
       "normalized":dict(frontend="normalized",refine_original=False),
       "refined":dict(frontend="normalized",refine_original=True)}


def configuration(base,mode):
    cfg=json.loads(json.dumps(base))
    identity=cfg["strategies"]["binary_corner"]["mono_identity"]
    identity.update(enabled=True,class1_enabled=False,method="hybrid",**MODES[mode])
    return cfg


def prepare(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    snapshot=out/"config_before.json"
    if not snapshot.exists():shutil.copyfile("config/tracker_config.json",snapshot)
    base=read(snapshot)
    jobs=[j for j in read("output/corner_quality_20261016/jobs.json") if j["strategy"]=="BC"]
    dump(out/"corner_jobs.json",jobs)
    runs=["#!/usr/bin/env bash","set -euo pipefail",
          "bash scripts/run_mono_normalized_corners.sh '"+out.as_posix()+"'"]
    for mode in MODES:
        p=out/"corners"/mode;p.mkdir(parents=True,exist_ok=True)
        dump(p/"config_snapshot.json",configuration(base,mode))
        (p/"jobs.txt").write_bytes(("\n".join("%d %s %d 0 %d %d %d %d" %
            (j["job"],j["path"],j["cls"],*j["roi"]) for j in jobs)+"\n").encode())
    (out/"run_corners.sh").write_bytes(("\n".join(runs)+"\n").encode())
    # Systematic sampling across each sequence and the full size/tilt range.
    timing=[j for j in jobs if (j["cls"]==0 and j["frame"]%15==0) or (j["cls"]==1 and j["frame"]%60==0)]
    (out/"timing_jobs.txt").write_bytes(("\n".join("%d %s %d 0 %d %d %d %d" %
        (j["job"],j["path"],j["cls"],*j["roi"]) for j in timing)+"\n").encode())
    print("Prepared",len(jobs),"corner jobs per mode and",len(timing),"timing jobs")


def corners(args):
    out=Path(args.out);jobs={j["job"]:j for j in read(out/"corner_jobs.json")}
    summaries=[];frames=[];reference=None
    for mode in MODES:
        raw=list(csv.DictReader((out/"corners"/mode/"detections.csv").open(encoding="utf8")))
        assert len(raw)==len(jobs) and len({r["job"] for r in raw})==len(jobs)
        if reference is None:reference={int(r["job"]):r for r in raw}
        rows=[]
        for r in raw:
            j=jobs[int(r["job"])];n=int(r["n"])
            pred=np.array([[float(r[f"x{i}"]),float(r[f"y{i}"])] for i in range(n)])
            ok=r["success"]=="1" and n==10 and np.isfinite(pred).all()
            item=dict(mode=mode,job=j["job"],sequence=j["sequence"],frame=j["frame"],cls=j["cls"],
                visible=j["fully_visible"],tilt=j["tilt"],size=j["size"],output=bool(ok),
                status=r["identity_status"],path=r["identity_path"],extract_ms=float(r["extract_ms"]),mismatch=False)
            if ok:
                gt=np.array(j["gt"]);cost=np.linalg.norm(pred[:,None]-gt[None,:],axis=2)
                ri,ci=linear_sum_assignment(cost**2);direct=np.diag(cost);assigned=cost[ri,ci]
                rmse=float(np.sqrt(np.mean(direct**2)));best=float(np.sqrt(np.mean(assigned**2)))
                item.update(direct=direct.tolist(),assigned=assigned.tolist(),rmse=rmse,
                    mismatch=bool(not np.array_equal(ci,np.arange(10)) and rmse-best>max(1.,.01*j["size"])))
            if j["cls"]==1:
                old=reference[j["job"]]
                assert all(r[k]==old[k] for k in ["success","n"]+[f"{a}{i}" for a in ("x","y") for i in range(10)])
            rows.append(item)
        frames.extend(rows)
        for cls in (0,1):
            for scope in ("full_visible","tilt_le30","all"):
                selected=[r for r in rows if r["cls"]==cls and (scope=="all" or r["visible"])
                          and (scope!="tilt_le30" or r["tilt"]<=30)]
                valid=[r for r in selected if r["output"]]
                summaries.append(dict(mode=mode,cls=cls,scope=scope,attempts=len(selected),outputs=len(valid),
                    mismatches=sum(r["mismatch"] for r in valid),
                    direct=stats([v for r in valid for v in r["direct"]]),
                    assigned=stats([v for r in valid for v in r["assigned"]]),
                    within2=sum(max(r["direct"])<=2 for r in valid),
                    extract_ms=stats([r["extract_ms"] for r in selected]),
                    statuses=dict(Counter(r["status"] for r in selected)),paths=dict(Counter(r["path"] for r in valid))))
    dump(out/"corner_summary.json",summaries);dump(out/"corner_frames.json",frames)
    indexed={mode:{r["job"]:r for r in frames if r["mode"]==mode} for mode in MODES}
    common=[job for job in jobs if jobs[job]["cls"]==0 and jobs[job]["fully_visible"]
            and all(indexed[mode][job]["output"] for mode in MODES)]
    dump(out/"corner_paired.json",dict(common_frames=len(common),modes={mode:dict(
        direct=stats([v for job in common for v in indexed[mode][job]["direct"]]),
        assigned=stats([v for job in common for v in indexed[mode][job]["assigned"]]),
        mismatches=sum(indexed[mode][job]["mismatch"] for job in common)) for mode in MODES}))
    print(json.dumps([s for s in summaries if s["cls"]==0 and s["scope"]=="full_visible"],indent=2))


def prepare_poses(args):
    out=Path(args.out);base=read(out/"config_before.json");ds=Path("tests/data/mono_robustness_20261015")
    (out/"configs").mkdir(exist_ok=True);(out/"runs").mkdir(exist_ok=True)
    jobs=[];runs=["#!/usr/bin/env bash","set -euo pipefail"]
    for p in read(ds/"suite_manifest.json")["profiles"]:
        for mode in MODES:
            name=p["name"]+"_"+mode;cfg=configuration(base,mode)
            cfg["mode"]="normal";cfg["mono_mode"]=True;cfg["manual_roi"]["enabled"]=False
            cfg["output"]=dict(visualize=False,log_file=True,pose_csv=True,directory=(out/"runs"/name).as_posix())
            cfg["input_system"]["max_frames"]=0
            cfg["input_system"]["image"].update(type="sequence",directory_path=(ds/p["name"]).as_posix(),sequence_pattern="frame")
            path=out/"configs"/(name+".json");dump(path,cfg)
            jobs.append(dict(name=name,sequence=p["name"],mode=mode,config=path.as_posix(),config_sha256=sha(path)))
            runs.append("./build-codex-bc-bench/GPNP '"+path.as_posix()+"' > '"+(out/"runs"/(name+"_stdout.log")).as_posix()+"' 2>&1")
    dump(out/"pose_experiment.json",dict(dataset=ds.as_posix(),jobs=jobs,binary_sha256=sha("build-codex-bc-bench/GPNP")))
    (out/"run_poses.sh").write_bytes(("\n".join(runs)+"\n").encode())
    print("Prepared",len(jobs),"full mono runs")


def poses(args):
    out=Path(args.out);experiment=read(out/"pose_experiment.json");rows=[];summary=[]
    for j in experiment["jobs"]:
        _,frames,_=analyze_job(j,Path(experiment["dataset"]),out)
        for r in frames:r["mode"]=j["mode"];r["sequence"]=j["sequence"]
        rows.extend(frames)
    for mode in MODES:
        for scope in ("all","tilt_le30"):
            for name in ["all"]+sorted({r["sequence"] for r in rows}):
                group=[r for r in rows if r["mode"]==mode and (scope=="all" or r["tilt_gt"]<=30)
                       and (name=="all" or r["sequence"]==name)]
                valid=[r for r in group if r["success"]]
                summary.append(dict(mode=mode,scope=scope,sequence=name,frames=len(group),success=len(valid),
                    normal_jumps=sum(r["z_jump"] for r in group),depth_jumps=sum(r["depth_jump"] for r in group),
                    normal_error_deg=stats([r["normal_error_deg"] for r in valid]),
                    depth_rel_error=stats([r["depth_rel_error"] for r in valid]),
                    extract_ms=stats([r["extract_ms"] for r in group]),strategies=dict(Counter(r["strategy"] for r in valid))))
    dump(out/"pose_frames.json",rows);dump(out/"pose_summary.json",summary)
    print(json.dumps([s for s in summary if s["sequence"]=="all"],indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","corners","prepare_poses","poses"])
    p.add_argument("--out",default="output/mono_normalized_20261016")
    args=p.parse_args();globals()[args.action](args)
