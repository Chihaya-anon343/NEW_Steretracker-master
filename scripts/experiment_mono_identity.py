#!/usr/bin/env python3
"""Prepare/analyze isolated BC identity ablations and mono YOLO pose A/B.
No GT enters the C++ matching path; GT is used only in this analysis script.
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from analyze_mono_robustness import read,dump,sha,analyze_job,stats as pose_stats
from analyze_corner_quality import stats

DEFAULT=dict(enabled=True,class1_enabled=False,method="hybrid",normalized_size=128,min_span_px=16.,min_roi_span_ratio=.25,min_iou=.75,
             min_score_gap=.05,max_fit_error_px=4.,max_fit_error_ratio=.03)


def prepare(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/"corners").mkdir(exist_ok=True)
    if not (out/"config_before.json").exists():shutil.copyfile("config/tracker_config.json",out/"config_before.json")
    base=read(out/"config_before.json")
    jobs=[j for j in read("output/corner_quality_20261016/jobs.json") if j["strategy"]=="BC"]
    dump(out/"corner_jobs.json",jobs)
    modes=["legacy","tt_fourway","projective","hybrid"]
    for mode in modes:
        folder=out/"corners"/mode;folder.mkdir(exist_ok=True)
        cfg=json.loads(json.dumps(base));c=dict(DEFAULT)
        c["class1_enabled"]=True # explicit experimental ablation; production mono defaults false
        c["enabled"]=mode!="legacy";c["method"]=mode if mode!="legacy" else "hybrid"
        cfg["strategies"]["binary_corner"]["mono_identity"]=c
        dump(folder/"config_snapshot.json",cfg)
        (folder/"jobs.txt").write_text("\n".join("%d %s %d 0 %d %d %d %d" % (j["job"],j["path"],j["cls"],*j["roi"]) for j in jobs)+"\n",encoding="utf8")
    reuse=args.reuse_baseline
    if reuse:
        source=Path(reuse)/"corners/legacy/detections.csv"
        assert len(list(csv.DictReader(source.open(encoding="utf8"))))==len(jobs)
        shutil.copyfile(source,out/"corners/legacy/detections.csv")
        dump(out/"baseline_reuse.json",dict(source=Path(reuse).as_posix(),corner_csv_sha256=sha(source),
            notes="Legacy identity disabled; original pose/config/point baselines retained, inactive new thresholds have no effect."))
    run=["#!/usr/bin/env bash","set -euo pipefail",
         "bash scripts/run_corner_extraction.sh '"+(out/"corners/legacy").as_posix()+"'"+(" --build-only" if reuse else "")]
    binary=(out/"corners/legacy/evaluate_corner_extraction").as_posix()
    for mode in modes[1:]:
        p=(out/"corners"/mode).as_posix()
        run.append("'"+binary+"' '"+p+"/config_snapshot.json' '"+p+"/jobs.txt' '"+p+"/detections.csv' > '"+p+"/extraction.log' 2>&1")
    (out/"run_corners.sh").write_bytes(("\n".join(run)+"\n").encode("utf8"))
    print("Prepared",len(jobs),"BC jobs per mode,",len(modes),"modes")


def analyze_corners(args):
    out=Path(args.out);jobs={j["job"]:j for j in read(out/"corner_jobs.json")}
    baseline={int(x["job"]):x for x in csv.DictReader(open("output/corner_quality_20261016/detections.csv",encoding="utf8"))}
    results=[];frame_rows=[];compatibility=True
    for mode in ("legacy","tt_fourway","projective","hybrid"):
        raw=list(csv.DictReader((out/"corners"/mode/"detections.csv").open(encoding="utf8")))
        assert len(raw)==len(jobs) and len({r["job"] for r in raw})==len(jobs)
        items=[]
        for r in raw:
            j=jobs[int(r["job"])];n=int(r["n"])
            pred=np.array([[float(r["x%d"%i]),float(r["y%d"%i])] for i in range(n)])
            ok=r["success"]=="1" and n==10 and np.isfinite(pred).all()
            item=dict(mode=mode,job=j["job"],cls=j["cls"],sequence=j["sequence"],frame=j["frame"],
                visible=j["fully_visible"],size=j["size"],tilt=j["tilt"],output=bool(ok),
                status=r["identity_status"],path=r["identity_path"],extract_ms=float(r["extract_ms"]),identity_ms=float(r["identity_ms"]))
            if ok:
                gt=np.array(j["gt"]);cost=np.linalg.norm(pred[:,None]-gt[None,:],axis=2)
                ri,ci=linear_sum_assignment(cost**2);direct=np.diag(cost);assigned=cost[ri,ci]
                rmse=float(np.sqrt(np.mean(direct**2)));best=float(np.sqrt(np.mean(assigned**2)))
                item.update(direct=direct.tolist(),assigned=assigned.tolist(),rmse=rmse,
                    mismatch=bool(not np.array_equal(ci,np.arange(10)) and rmse-best>max(1.,.01*j["size"])))
            if mode=="legacy":
                old=baseline[j["job"]]
                assert old["success"]==r["success"] and old["n"]==r["n"]
                for i in range(n):
                    compatibility &= abs(float(old["x%d"%i])-pred[i,0])<1e-7 and abs(float(old["y%d"%i])-pred[i,1])<1e-7
            items.append(item)
        frame_rows.extend(items)
        for cls in (0,1):
            for visibility in (True,False):
                selected=[x for x in items if x["cls"]==cls and x["visible"]==visibility]
                if not selected:continue
                groupings=[("all",selected)]
                groupings += [(name,[x for x in selected if x["sequence"]==name]) for name in sorted({x["sequence"] for x in selected})]
                for name,group in groupings:
                    valid=[x for x in group if x["output"]]
                    results.append(dict(mode=mode,cls=cls,visible=visibility,group=name,attempts=len(group),outputs=len(valid),
                        rejections=len(group)-len(valid),mismatch_frames=sum(x["mismatch"] for x in valid),
                        all_points_within2=sum(max(x["direct"])<=2 for x in valid),
                        direct=stats([e for x in valid for e in x["direct"]]),
                        assigned=stats([e for x in valid for e in x["assigned"]]),
                        extract_ms=stats([x["extract_ms"] for x in group]),identity_ms=stats([x["identity_ms"] for x in group]),
                        statuses=dict(Counter(x["status"] for x in group)),paths=dict(Counter(x["path"] for x in valid))))
    dump(out/"corner_summary.json",results);dump(out/"corner_frames.json",frame_rows)
    dump(out/"corner_verification.json",dict(rows=4*len(jobs),legacy_matches_historical_points=bool(compatibility)))
    assert compatibility
    print(json.dumps([s for s in results if s["group"]=="all" and s["visible"]],ensure_ascii=False,indent=2))


def prepare_poses(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/"configs").mkdir(exist_ok=True);(out/"runs").mkdir(exist_ok=True)
    if not (out/"config_before.json").exists():shutil.copyfile("config/tracker_config.json",out/"config_before.json")
    base=read(out/"config_before.json");jobs=[]
    ds=Path("tests/data/mono_robustness_20261015")
    for profile in read(ds/"suite_manifest.json")["profiles"]:
        for mode in ("legacy","hybrid"):
            name=profile["name"]+"_"+mode
            cfg=json.loads(json.dumps(base));cfg["mode"]="normal";cfg["mono_mode"]=True
            cfg["manual_roi"]["enabled"]=False
            cfg["output"]=dict(visualize=False,log_file=True,pose_csv=True,directory=(out/"runs"/name).as_posix())
            cfg["input_system"]["max_frames"]=0;cfg["input_system"]["image"].update(type="sequence",directory_path=(ds/profile["name"]).as_posix(),sequence_pattern="frame")
            identity=dict(DEFAULT);identity["enabled"]=mode!="legacy"
            cfg["strategies"]["binary_corner"]["mono_identity"]=identity
            path=out/"configs"/(name+".json");dump(path,cfg)
            reused=False
            if args.reuse_baseline and mode=="legacy":
                source_root=Path(args.reuse_baseline);source=source_root/"runs"/name
                old_cfg=read(source_root/"configs"/(name+".json"))
                def comparable(c):
                    c=json.loads(json.dumps(c));c.pop("output",None)
                    c["strategies"]["binary_corner"].pop("mono_identity",None);return c
                assert comparable(old_cfg)==comparable(cfg),"Baseline runtime config changed"
                raw=list(csv.DictReader((source/"poses.csv").open(encoding="utf8")))
                assert len(raw)==profile["frames"] and [int(r["frame"]) for r in raw]==list(range(1,profile["frames"]+1))
                destination=out/"runs"/name;destination.mkdir(exist_ok=True)
                shutil.copyfile(source/"poses.csv",destination/"poses.csv")
                shutil.copyfile(source_root/"runs"/(name+"_stdout.log"),out/"runs"/(name+"_stdout.log"))
                reused=True
            jobs.append(dict(name=name,sequence=profile["name"],mode=mode,config=path.as_posix(),reused=reused))
    dump(out/"pose_experiment.json",dict(dataset=ds.as_posix(),jobs=jobs,binary_sha256=sha("build-codex-bc-bench/GPNP")))
    run=["#!/usr/bin/env bash","set -euo pipefail"]
    for j in jobs:
        if j["reused"]:continue
        run += ["echo 'RUN "+j["name"]+"'","./build-codex-bc-bench/GPNP '"+j["config"]+"' > '"+(out/"runs"/(j["name"]+"_stdout.log")).as_posix()+"' 2>&1"]
    (out/"run_poses.sh").write_bytes(("\n".join(run)+"\n").encode("utf8"))
    print("Prepared",len(jobs),"mono pose runs")


def analyze_poses(args):
    out=Path(args.out);e=read(out/"pose_experiment.json");summaries=[];frames=[]
    for j in e["jobs"]:
        manifest,rows,summary=analyze_job(j,Path(e["dataset"]),out)
        for scope in ("all","tilt_le30"):
            selected=rows if scope=="all" else [r for r in rows if r["tilt_gt"]<=30]
            valid=[r for r in selected if r["success"]]
            summaries.append(dict(sequence=j["sequence"],mode=j["mode"],scope=scope,frames=len(selected),success=len(valid),
                good10=sum(r["normal_error_deg"]<=10 for r in valid),bad20=sum(r["normal_error_deg"]>20 for r in valid),
                normal_jumps=sum(r["z_jump"] for r in selected),depth_jumps=sum(r["depth_jump"] for r in selected),
                normal_error_deg=pose_stats([r["normal_error_deg"] for r in valid]),
                depth_rel_error=pose_stats([r["depth_rel_error"] for r in valid]),
                extract_ms=pose_stats([r["extract_ms"] for r in selected]),strategies=dict(Counter(r["strategy"] for r in valid))))
        frames.extend(rows)
    dump(out/"pose_summary.json",summaries);dump(out/"pose_frames.json",frames)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","analyze_corners","prepare_poses","analyze_poses"])
    p.add_argument("--out",default="output/mono_identity_20261016")
    p.add_argument("--reuse-baseline");args=p.parse_args();globals()[args.action](args)
