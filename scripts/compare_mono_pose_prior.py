#!/usr/bin/env python3
"""Compare the 30deg near-nadir MonoPnP prior on frozen 2026-10-15 inputs.

prepare -> Docker bash <out>/run_all.sh -> analyze
All six prior-ON runs are compared with historical OFF results. Two OFF replays
verify compatibility. Only GT tilt<=30deg is used to assess in-domain accuracy.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from analyze_mono_robustness import analyze_job, dump, read, sha, stats


def prepare_segments(args):
    old,out=Path(args.baseline),Path(args.out)
    experiment=read(old/"experiment.json"); ds=Path(experiment["dataset"])
    target=Path(args.segment_dataset)
    if target.exists() and any(target.iterdir()) and not args.reuse_segments: raise ValueError("Segment dataset already exists")
    target.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for original in experiment["jobs"][:6]:
        if original["name"] not in ("oblique_clean","fast_rotation","oblique_degraded"): continue
        manifest=read(ds/original["sequence"]/"manifest.json"); frames=manifest["frames"]
        starts=[i for i,g in enumerate(frames) if g["tilt_deg"]<=30 and (i==0 or frames[i-1]["tilt_deg"]>30)]
        for start in starts:
            stop=next((i for i in range(start,len(frames)) if frames[i]["tilt_deg"]>30),len(frames))
            sequence=f"{original['name']}_s{start:04d}_{stop:04d}"
            folder=target/sequence;folder.mkdir(exist_ok=args.reuse_segments)
            part=json.loads(json.dumps(manifest));part["frames"]=[]
            for index,g in enumerate(frames[start:stop]):
                gt=json.loads(json.dumps(g));gt["source_frame"]=g["frame"];gt["frame"]=index;gt["path"]=f"frame_{index:06d}.png"
                source=ds/original["sequence"]/g["path"]; destination=folder/gt["path"]
                if args.reuse_segments:
                    assert sha(source)==sha(destination),destination
                else: shutil.copyfile(source,destination)
                part["frames"].append(gt)
            part["profile"].update(name=sequence,frames=stop-start)
            dump(folder/"manifest.json",part)
            for enabled in (False,True):
                name=sequence+("_prior_on" if enabled else "_prior_off")
                cfg=read(original["config"])
                cfg["mono_pose_prior"]=dict(enabled=enabled,max_tilt_deg=30.0)
                cfg["input_system"]["image"]["directory_path"]=folder.as_posix()
                cfg["output"].update(directory=(out/"runs"/name).as_posix(),pose_csv=True,visualize=False)
                path=out/"configs"/(name+".json");dump(path,cfg)
                jobs.append(dict(name=name,sequence=sequence,group=original["name"],source_range=[start,stop],enabled=enabled,
                                 config=path.as_posix(),sha256=sha(path)))
    dump(out/"segment_experiment.json",dict(dataset=target.as_posix(),jobs=jobs,binary_sha256=sha("build-codex-bc-bench/GPNP")))
    script=["#!/usr/bin/env bash","set -euo pipefail"]
    for j in jobs:
        script += ["echo 'RUN "+j["name"]+"'","./build-codex-bc-bench/GPNP '"+j["config"]+"' > '"+(out/"runs"/(j["name"]+"_stdout.log")).as_posix()+"' 2>&1"]
    with (out/"run_segments.sh").open("w",encoding="utf-8",newline="\n") as f:f.write("\n".join(script)+"\n")
    print(f"Prepared {len(jobs)} segment jobs, all GT tilts <=30deg")


def analyze_segments(args):
    out=Path(args.out);e=read(out/"segment_experiment.json");ds=Path(e["dataset"])
    (out/"analysis").mkdir(exist_ok=True)
    assert sha("build-codex-bc-bench/GPNP")==e["binary_sha256"]
    combined={};individual=[]
    for j in e["jobs"]:
        manifest,rows,summary=analyze_job(j,ds,out)
        assert all(g["tilt_deg"]<=30 for g in manifest["frames"])
        if j["enabled"]:assert all(r["normal_z"]>=np.cos(np.deg2rad(30))-1e-8 for r in rows if r["success"])
        key=(j["group"],j["enabled"])
        combined.setdefault(key,[]).extend(rows)
        individual.append(dict(job=j,summary=summary))
    groups=[]
    for group in sorted({k[0] for k in combined}):
        result=dict(sequence=group)
        for enabled in (False,True):
            rows=combined[(group,enabled)];valid=[r for r in rows if r["success"]]
            result["after" if enabled else "before"]=dict(frames=len(rows),success=len(valid),
                good10=sum(r["normal_error_deg"]<=10 for r in valid),bad20=sum(r["normal_error_deg"]>20 for r in valid),
                z_jumps=sum(r["z_jump"] for r in valid),normal_error_deg=stats([r["normal_error_deg"] for r in valid]))
        groups.append(result)
    dump(out/"analysis"/"segment_comparison.json",dict(groups=groups,jobs=individual))
    print(json.dumps(groups,indent=2))


def prepare(args):
    old=Path(args.baseline); out=Path(args.out)
    experiment=read(old/"experiment.json")
    (out/"configs").mkdir(parents=True,exist_ok=True)
    (out/"runs").mkdir(exist_ok=True)
    jobs=[]
    for original in experiment["jobs"][:6]:
        assert sha(original["config"])==original["sha256"],original["config"]
        for enabled in ([True,False] if original["name"] in ("reference","normal_ambiguity") else [True]):
            name=original["name"]+("_prior_on" if enabled else "_prior_off")
            cfg=read(original["config"])
            cfg["mono_pose_prior"]=dict(enabled=enabled,max_tilt_deg=30.0)
            cfg["output"].update(directory=(out/"runs"/name).as_posix(),pose_csv=True,visualize=False)
            path=out/"configs"/(name+".json"); dump(path,cfg)
            jobs.append(dict(name=name,sequence=original["sequence"],config=path.as_posix(),sha256=sha(path),enabled=enabled))
    source_files=[Path("main.cpp"),*Path("include").rglob("*.hpp"),*Path("src").rglob("*.cpp")]
    dump(out/"experiment.json",dict(dataset=experiment["dataset"],baseline=args.baseline,jobs=jobs,max_tilt_deg=30.,
                                  binary_sha256=sha("build-codex-bc-bench/GPNP"),model_sha256=experiment["model_sha256"],
                                  source_sha256={p.as_posix():sha(p) for p in source_files}))
    script=["#!/usr/bin/env bash","set -euo pipefail"]
    for j in jobs:
        script += ["echo 'RUN "+j["name"]+"'",
                   "./build-codex-bc-bench/GPNP '"+j["config"]+"' > '"+(out/"runs"/(j["name"]+"_stdout.log")).as_posix()+"' 2>&1"]
    with (out/"run_all.sh").open("w",encoding="utf-8",newline="\n") as f: f.write("\n".join(script)+"\n")
    print(f"Prepared {len(jobs)} jobs")


def scoped(rows,truths,inside):
    selected=[r for r,g in zip(rows,truths) if (g["tilt_deg"]<=30.+1e-8)==inside]
    valid=[r for r in selected if r["success"]]
    eligible_pairs=sum(i>0 and (truths[i-1]["tilt_deg"]<=30.+1e-8)==inside and
                       rows[i]["success"] and rows[i-1]["success"]
                       for i in range(len(rows)) if (truths[i]["tilt_deg"]<=30.+1e-8)==inside)
    jumps=sum(r["z_jump"] and r["frame"]>0 and
              (truths[r["frame"]-1]["tilt_deg"]<=30.+1e-8)==inside for r in valid)
    return dict(frames=len(selected),success=len(valid),good10=sum(r["normal_error_deg"]<=10 for r in valid),
                bad20=sum(r["normal_error_deg"]>20 for r in valid),bad90=sum(r["normal_error_deg"]>90 for r in valid),
                z_jumps=jumps,consecutive_success_pairs=eligible_pairs,
                normal_error_deg=stats([r["normal_error_deg"] for r in valid]),
                estimated_tilt_deg=stats([np.degrees(np.arccos(np.clip(r["normal_z"],-1,1))) for r in valid]))


def analyze(args):
    out=Path(args.out); old=Path(args.baseline)
    e=read(out/"experiment.json"); dataset=Path(e["dataset"])
    assert sha("build-codex-bc-bench/GPNP")==e["binary_sha256"]
    (out/"analysis").mkdir(exist_ok=True); (out/"figures").mkdir(exist_ok=True)
    originals={j["name"]:j for j in read(old/"experiment.json")["jobs"][:6]}
    old_results={name:analyze_job(j,dataset,old) for name,j in originals.items()}
    new_results={j["name"]:analyze_job(j,dataset,out) for j in e["jobs"]}
    comparisons=[]; allrows=[]; replays=[]
    for j in e["jobs"]:
        manifest,rows,summary=new_results[j["name"]]
        allrows+=rows
        if not j["enabled"]:
            before=list(csv.DictReader((old/"runs"/j["sequence"]/"poses.csv").open(encoding="utf-8")))
            after=list(csv.DictReader((out/"runs"/j["name"]/"poses.csv").open(encoding="utf-8")))
            same_status=all(a["success"]==b["success"] and a["strategy"]==b["strategy"] and a["gate_status"]==b["gate_status"] for a,b in zip(before,after))
            values=[abs(float(a[k])-float(b[k])) for a,b in zip(before,after) if a["success"]==b["success"]=="1"
                    for k in ["tx","ty","tz",*[f"R{i}{q}" for i in range(3) for q in range(3)]]]
            replays.append(dict(sequence=j["sequence"],same_status=same_status,max_pose_element_difference=max(values,default=0.)))
            continue
        _,baseline,baseline_summary=old_results[j["sequence"]]
        direct_prior_violations=0; angle_violations=0; invalid_depths=0
        raw=list(csv.DictReader((out/"runs"/j["name"]/"poses.csv").open(encoding="utf-8")))
        for r in raw:
            if r["success"]!="1": continue
            n=np.array([float(r["R02"]),float(r["R12"]),float(r["R22"])])
            t=np.array([float(r[k]) for k in ("tx","ty","tz")])
            angle_violations+=n[2]<np.cos(np.deg2rad(30))-1e-8
            direct_prior_violations+=n@t<=0
            invalid_depths+=t[2]<=0
        assert angle_violations==direct_prior_violations==invalid_depths==0
        comparisons.append(dict(sequence=j["sequence"],all_before=baseline_summary,all_after=summary,
                                in_domain_before=scoped(baseline,manifest["frames"],True),
                                in_domain_after=scoped(rows,manifest["frames"],True),
                                out_domain_before=scoped(baseline,manifest["frames"],False),
                                out_domain_after=scoped(rows,manifest["frames"],False),
                                angle_violations=int(angle_violations),front_side_violations=int(direct_prior_violations)))
    dump(out/"analysis"/"comparison.json",comparisons)
    dump(out/"analysis"/"disabled_replays.json",replays)
    fields=sorted(set(k for r in allrows for k in r))
    with (out/"analysis"/"frame_metrics.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(allrows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for i,name in enumerate(["reference","normal_ambiguity"]):
        for label,rows in [("OFF",old_results[name][1]),("ON",new_results[name+"_prior_on"][1])]:
            axes[i,0].plot([r.get("normal_error_deg",np.nan) for r in rows],label=label,lw=1)
            axes[i,1].plot([np.degrees(np.arccos(np.clip(r["normal_z"],-1,1))) if r["success"] else np.nan for r in rows],label=label,lw=1)
        axes[i,0].set(title=name,ylabel="normal error (deg)",xlabel="frame (0 based)")
        axes[i,1].set(title=name,ylabel="estimated tilt (deg)",xlabel="frame (0 based)")
        axes[i,1].axhline(30,color="red",ls="--",label="30 deg limit")
        axes[i,0].legend();axes[i,1].legend()
    fig.savefig(out/"figures"/"prior_comparison.png",dpi=150);plt.close(fig)
    print(json.dumps(dict(comparison=comparisons,disabled_replays=replays),ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("action",choices=["prepare","analyze","prepare-segments","analyze-segments"])
    p.add_argument("--baseline",default="output/mono_robustness_20261015")
    p.add_argument("--out",default="output/mono_pose_prior_bounded_20261016")
    p.add_argument("--segment-dataset",default="tests/data/mono_nadir_segments_20261016")
    p.add_argument("--reuse-segments",action="store_true",help="Verify and reuse unchanged segment PNGs")
    args=p.parse_args();{"prepare":prepare,"analyze":analyze,"prepare-segments":prepare_segments,"analyze-segments":analyze_segments}[args.action](args)
