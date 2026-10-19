"""Format only changed C++ ranges relative to this experiment's saved source."""
import difflib
import subprocess
from pathlib import Path

before=Path("output/mono_priority_20261019/before")
for old in before.rglob("*"):
    if old.suffix not in (".hpp",".cpp"):continue
    path=old.relative_to(before)
    if not path.exists():continue
    original=old.read_text(encoding="utf-8-sig").splitlines()
    current=path.read_text(encoding="utf-8-sig").splitlines()
    ranges=[]
    for tag,_,__,start,end in difflib.SequenceMatcher(None,original,current,autojunk=False).get_opcodes():
        if tag!="equal" and end>start:ranges.append("--lines=%d:%d"%(start+1,end))
    if ranges:subprocess.run(["clang-format","-i","--sort-includes=false",*ranges,str(path)],check=True)
subprocess.run(["clang-format","-i","--sort-includes=false","scripts/evaluate_mono_priority.cpp"],check=True)
