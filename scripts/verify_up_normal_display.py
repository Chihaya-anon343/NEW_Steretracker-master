"""Replay actual visualizations and ensure pose outputs are display-independent."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

OUT=Path("output/up_normal_display_20261019")
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def dump(path,value):
    Path(path).write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode())

shutil.copy2("build-codex-bc-bench/GPNP",OUT/"GPNP_after")
current=read(OUT/"config_before.json")
bc=read("output/mono_priority_20261019/final_configs/debug_eight_corners.json")
tt=read("output/mono_priority_20261019/configs/reference_partial.json")
tt["mode"]="debug"
tt["input"]["left"]=tt["input"]["right"]="tests/data/mono_robustness_20261015/reference/frame_000000.png"
tt["output"]["visualize"]=True
dual=read("output/mono_priority_20261019/configs/reference_partial.json")
dual["mode"]="debug"
dual["input"]["left"]=dual["input"]["right"]="tests/data/mono_robustness_20261015/reference/frame_000043.png"
dual["output"]["visualize"]=True
cases=[("current_dual",current),("mono_bc",bc),("mono_tt",tt),("dual_roi",dual)]
checks=[]
for name,base in cases:
    for version in ("before","after"):
        cfg=json.loads(json.dumps(base))
        directory=OUT/(name+"_"+version)
        cfg["output"].update(directory=directory.as_posix(),pose_csv=True)
        path=OUT/(name+"_"+version+".json")
        if directory.exists():
            assert read(path)==cfg
            assert len(list(csv.DictReader((directory/"poses.csv").open())))==2
            print("REUSE",name,version,flush=True)
            continue
        dump(path,cfg)
        print("RUN",name,version,flush=True)
        with (OUT/(name+"_"+version+".log")).open("w") as stream:
            subprocess.run([str(OUT/("GPNP_"+version)),str(path)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    rows=[list(csv.DictReader((OUT/(name+"_"+v)/"poses.csv").open())) for v in ("before","after")]
    assert len(rows[0])==len(rows[1])==2
    mismatches=[]
    for a,b in zip(*rows):
        for key in a:
            if key not in ("extract_ms","pnp_ms","roi_verify_ms") and a[key]!=b[key]:
                mismatches.append([a["frame"],key,a[key],b[key]])
    checks.append(dict(case=name,frames=len(rows[0]),strategies=[r["strategy"] for r in rows[1]],
                       successes=sum(r["success"]=="1" for r in rows[1]),differences=mismatches))
    assert not mismatches,checks[-1]
assert Path("config/tracker_config.json").read_bytes()==(OUT/"config_before.json").read_bytes()
dump(OUT/"verification.json",dict(checks=checks,root_config_preserved=True,
    binary_sha256={v:hashlib.sha256((OUT/("GPNP_"+v)).read_bytes()).hexdigest() for v in ("before","after")}))
print(json.dumps(checks),flush=True)
