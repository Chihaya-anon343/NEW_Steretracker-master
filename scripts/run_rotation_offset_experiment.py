from pathlib import Path
import shlex
import subprocess
root=Path(__file__).resolve().parents[1]
out=root/'output/rotation_offset_20261020'
out.mkdir(parents=True,exist_ok=True)
build=root/'build-codex-bc-bench/tests'
subprocess.run(['cmake','--build',str(build.parent),'--target','test_dual_roi_flow','-j','4'],check=True)
opts=[]
for line in (build/'CMakeFiles/test_dual_roi_flow.dir/flags.make').read_text().splitlines():
    if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):opts+=shlex.split(line.split('=',1)[1])
obj=out/'experiment.o'
subprocess.run(['/usr/bin/c++',*opts,'-c',str(root/'scripts/experiment_rotation_offset.cpp'),'-o',str(obj)],check=True)
link=shlex.split((build/'CMakeFiles/test_dual_roi_flow.dir/link.txt').read_text())
link=[str(obj) if x.endswith('/unit/test_dual_roi_flow.cpp.o') else x for x in link]
link[link.index('-o')+1]=str(out/'experiment')
subprocess.run(link,cwd=build,check=True)
with (out/'console.log').open('w') as log:subprocess.run([str(out/'experiment')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
