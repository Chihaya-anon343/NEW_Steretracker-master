#!/usr/bin/env python3
"""Run independent mono sequences in at most two processes, resume verified jobs.

Only for pose quality. Never use concurrent process timings as speed results.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",default="output/mono_normalized_20261016")
    p.add_argument("--workers",type=int,default=2);args=p.parse_args()
    assert 1<=args.workers<=2
    out=Path(args.out);experiment=out/"pose_experiment.json"
    e=json.loads(experiment.read_text(encoding="utf8"));binary=Path("build-codex-bc-bench/GPNP")
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert digest(binary)==e["binary_sha256"],"Build differs from frozen experiment"
    counts={p["name"]:p["frames"] for p in json.loads((Path(e["dataset"])/"suite_manifest.json").read_text())["profiles"]}

    def run(j):
        assert digest(j["config"])==j["config_sha256"]
        dest=out/"runs"/j["name"];csvpath=dest/"poses.csv"
        logpath=out/"runs"/(j["name"]+"_stdout.log")
        complete=False
        if csvpath.exists() and logpath.exists() and csvpath.stat().st_mtime>experiment.stat().st_mtime:
            rows=list(csv.DictReader(csvpath.open()))
            log=logpath.read_text(errors="replace")
            complete=(len(rows)==counts[j["sequence"]] and
                      [int(r["frame"]) for r in rows]==list(range(1,len(rows)+1)) and
                      log.rstrip().endswith("tracking_log.txt"))
        if not complete:
            print("RUN",j["name"],flush=True)
            with logpath.open("w") as log:subprocess.run([str(binary),j["config"]],stdout=log,stderr=subprocess.STDOUT,check=True)
            rows=list(csv.DictReader(csvpath.open()))
            assert len(rows)==counts[j["sequence"]]
        print("DONE",j["name"],"verified existing" if complete else "new",flush=True)
        return dict(name=j["name"],rows=len(rows),reused_current_run=complete,
                    csv_sha256=digest(csvpath),config_sha256=j["config_sha256"])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run,e["jobs"]))
    (out/"pose_completion.json").write_text(json.dumps(dict(binary_sha256=e["binary_sha256"],jobs=results),indent=2))


if __name__=="__main__":main()
