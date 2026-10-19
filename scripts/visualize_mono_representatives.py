#!/usr/bin/env python3
"""Visualize actual saved extraction points and sequential PnP, without re-solving pose."""
import argparse
import csv
import json
import shutil
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_mono_robustness import read,dump,stats

ROOT=Path("output/mono_normalized_20261016")
OUT=ROOT/"representatives"
CASES=[
    ("reference",45,"常规清晰目标","clear target"),
    ("reference",55,"旧方法局部大误差样例","old large corner error"),
    ("oblique_clean",75,"应用范围内倾斜","tilted, clean"),
    ("oblique_degraded",75,"与上一帧同几何的退化图","same geometry, degraded"),
    ("fast_rotation",135,"快速面内旋转","fast rotation"),
    ("oblique_clean",165,"大ROI性能压力，超30度","large ROI, outside 30 deg"),
    ("grazing",60,"大倾角压力，超30度","grazing, outside 30 deg"),
    ("reference",15,"新BC拒绝边界样例","new isolated BC rejection"),
    ("oblique_degraded",149,"残余深度跳变样例","remaining depth jump"),
]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=read(ROOT/"corner_jobs.json");selected=[]
    for seq,frame,title,en in CASES:
        j=next(j for j in jobs if j["sequence"]==seq and j["frame"]==frame and j["cls"]==0)
        selected.append(dict(**j,title=title,title_en=en))
    dump(OUT/"selection.json",selected)
    replay=OUT/"replay"
    for mode in ("current","normalized","refined"):
        folder=replay/"corners"/mode;folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/"corners"/mode/"config_snapshot.json",folder/"config_snapshot.json")
        (folder/"jobs.txt").write_bytes(("\n".join("%d %s %d 0 %d %d %d %d" %
            (j["job"],j["path"],j["cls"],*j["roi"]) for j in selected)+"\n").encode())
    rows=read(ROOT/"corner_frames.json")
    good={m:{r["job"] for r in rows if r["mode"]==m and r["output"]} for m in ("current","normalized","refined")}
    common=[j for j in selected if all(j["job"] in good[m] for m in good)]
    (OUT/"resource_jobs.txt").write_bytes(("\n".join("%d %s %d 0 %d %d %d %d" %
        (j["job"],j["path"],j["cls"],*j["roi"]) for j in common)+"\n").encode())
    print("Selected",len(selected),"frames;",len(common),"common-success resource ROIs")


def points(row):
    return np.array([[float(row[f"x{i}"]),float(row[f"y{i}"])] for i in range(int(row["n"]))]).reshape(-1,2)


def pose_projection(K,R,t,length=100.):
    xyz=np.array([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
    cam=xyz@R.T+t
    q=cam@K.T
    if not np.isfinite(q).all() or (cam[:,2]<=0).any():return None
    return q[:,:2]/q[:,2:]


def render():
    selection=read(OUT/"selection.json");(OUT/"cards").mkdir(exist_ok=True)
    raw={m:{int(r["job"]):r for r in csv.DictReader((ROOT/"corners"/m/"detections.csv").open())}
         for m in ("current","refined")}
    metrics={(r["mode"],r["job"]):r for r in read(ROOT/"corner_frames.json")}
    pm={(r["mode"],r["sequence"],r["frame"]):r for r in read(ROOT/"pose_frames.json")}
    poses={}
    for seq,_,_,_ in CASES:
        for mode in ("current","refined"):
            poses[(mode,seq)]=list(csv.DictReader((ROOT/"runs"/(seq+"_"+mode)/"poses.csv").open()))
    manifests={seq:read(Path("tests/data/mono_robustness_20261015")/seq/"manifest.json") for seq,_,_,_ in CASES}
    records=[];thumbnails=[]
    timing=list(csv.DictReader((ROOT/"paired_timing.csv").open()))
    for j in selection:
        seq,f,job=j["sequence"],j["frame"],j["job"]
        image=cv2.cvtColor(cv2.imread(j["path"]),cv2.COLOR_BGR2RGB);gt=np.array(j["gt"])
        predicted={m:points(raw[m][job]) for m in raw}
        allp=np.vstack([gt]+[p for p in predicted.values() if len(p)])
        low=allp.min(axis=0);high=allp.max(axis=0);center=(low+high)/2
        span=max(64,float(max(high-low))*1.28)
        x0=max(0,int(center[0]-span/2));y0=max(0,int(center[1]-span/2))
        x1=min(image.shape[1],int(center[0]+span/2));y1=min(image.shape[0],int(center[1]+span/2))
        crop=image[y0:y1,x0:x1];offset=np.array([x0,y0])
        fig,axes=plt.subplots(2,3,figsize=(14,9.6),constrained_layout=True)
        for ax in axes.flat:
            ax.imshow(crop);ax.set_xlim(0,crop.shape[1]);ax.set_ylim(crop.shape[0],0);ax.set_xticks([]);ax.set_yticks([])
        for col,mode in enumerate((None,"current","refined")):
            ax=axes[0,col];g=gt-offset
            ax.scatter(g[:,0],g[:,1],s=35,facecolors="none",edgecolors="cyan",linewidths=1)
            for i,p in enumerate(g):ax.annotate(str(i),p,xytext=(3,-4),textcoords="offset points",color="cyan",fontsize=8)
            if mode:
                r=raw[mode][job];p=predicted[mode]-offset
                if len(p):
                    ax.scatter(p[:,0],p[:,1],s=35,marker="x",color="red",linewidths=1)
                    for i,point in enumerate(p):ax.annotate(str(i),point,xytext=(-9,7),textcoords="offset points",color="red",fontsize=7)
                    for a,b in zip(g,p):ax.plot([a[0],b[0]],[a[1],b[1]],color="yellow",lw=.8)
                metric=metrics[(mode,job)]
                line=f"RMSE={metric['rmse']:.2f}px, max={max(metric['direct']):.2f}px" if metric["output"] else "REJECT: "+metric["status"]
                ax.set_title(("Old BC" if mode=="current" else "New BC + original edge fit")+"\n"+line,fontsize=10)
            else:ax.set_title("EXTRACTION: annotated ROI / GT IDs\ncyan circles = GT; red crosses = observed",fontsize=10)
        manifest=manifests[seq];truth=manifest["frames"][f];K=np.array(manifest["K"])
        gtaxis=pose_projection(K,np.array(truth["R"]),np.array(truth["t_mm"]))
        for col,mode in enumerate((None,"current","refined")):
            ax=axes[1,col]
            if mode:
                row=poses[(mode,seq)][f];metric=pm[(mode,seq,f)]
                if int(row["success"]):
                    R=np.array([[float(row[f"R{a}{b}"]) for b in range(3)] for a in range(3)])
                    t=np.array([float(row[k]) for k in ("tx","ty","tz")])
                    axis=pose_projection(K,R,t)
                    text=f"{row['strategy']} | Z={t[2]:.1f}mm\nnormal err={metric['normal_error_deg']:.1f}deg, depth err={metric['depth_rel_error']*100:.1f}%"
                else:axis=None;text="POSE FAILED / "+row["strategy"]
                if gtaxis is not None:
                    for k in range(1,4):ax.plot(*(gtaxis[[0,k]]-offset).T,"--",color="white",lw=1,alpha=.8)
            else:axis=gtaxis;text=f"GT Z={truth['t_mm'][2]:.1f}mm\nX=red / Y=green / Z=blue; 100mm axes"
            if axis is not None:
                q=axis-offset
                for k,color in enumerate(("red","lime","dodgerblue"),1):
                    ax.plot(q[[0,k],0],q[[0,k],1],color=color,lw=2)
                    if 0<=q[k,0]<crop.shape[1] and 0<=q[k,1]<crop.shape[0]:ax.text(*q[k],"XYZ"[k-1],color=color,fontsize=9)
            ax.set_title(("FULL PIPELINE: ground truth" if mode is None else "Recorded sequential "+mode+" PnP")+"\n"+text,fontsize=10)
        fig.suptitle(f"{seq} / frame_{f:06d}.png (tracker row {f+1}) | {j['title_en']}\n"
            f"GT tilt={j['tilt']:.1f}deg, class0 span={j['size']:.0f}px; top row isolated BC, bottom row full YOLO + history + PnP",fontsize=12)
        path=OUT/"cards"/(seq+f"_f{f:06d}.png");fig.savefig(path,dpi=130);plt.close(fig)
        thumbnail=predicted["refined"]-offset
        thumbnails.append((crop,thumbnail,j))
        record=dict(job=job,sequence=seq,frame=f,path=j["path"],title=j["title"],tilt=j["tilt"],size=j["size"],
            fully_visible=j["fully_visible"],card=path.as_posix(),corners={},pose={},timing={})
        for mode in ("current","refined"):
            metric=metrics[(mode,job)];pose=pm[(mode,seq,f)]
            record["corners"][mode]={k:metric.get(k) for k in ("output","status","rmse","direct")}
            record["pose"][mode]={k:pose.get(k) for k in ("success","strategy","normal_error_deg","depth_mm","depth_gt_mm","depth_rel_error","z_jump","depth_jump")}
            ts=[float(r["extract_ms"]) for r in timing if int(r["job"])==job and r["mode"]==mode]
            if ts:record["timing"][mode]=stats(ts)
        records.append(record)
    fig,axes=plt.subplots(3,3,figsize=(13,13),constrained_layout=True)
    for ax,(crop,points_,j) in zip(axes.flat,thumbnails):
        ax.imshow(crop)
        if len(points_):ax.scatter(points_[:,0],points_[:,1],marker="x",color="red",s=20)
        ax.set_title(f"{j['sequence']} / {j['frame']:06d}\n{j['title_en']} | tilt={j['tilt']:.1f}deg",fontsize=9)
        ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(OUT/"overview.png",dpi=125);plt.close(fig)
    # Capture replays must reproduce the already published point results.
    for mode in ("current","normalized","refined"):
        full={int(r["job"]):r for r in csv.DictReader((ROOT/"corners"/mode/"detections.csv").open())}
        replay=list(csv.DictReader((OUT/"replay/corners"/mode/"detections.csv").open()))
        assert len(replay)==len(selection)
        for r in replay:
            old=full[int(r["job"])]
            assert all(old[k]==r[k] for k in ["success","n"]+[f"{a}{i}" for a in ("x","y") for i in range(10)])
    # Real C++ diagnostic images, never a reconstruction from GT.
    fig,axes=plt.subplots(len(selection),3,figsize=(9,3*len(selection)),constrained_layout=True)
    for row,j in enumerate(selection):
        for col,name in enumerate(("foreground","coarse","normalized")):
            path=OUT/"replay/diagnostics"/(f"job{j['job']}_refined_{name}.png")
            ax=axes[row,col]
            if path.exists():ax.imshow(cv2.imread(str(path),0),cmap="gray",vmin=0,vmax=255)
            else:ax.text(.5,.5,"Not produced / rejected",ha="center",va="center",transform=ax.transAxes)
            ax.set_title(f"{j['sequence']} {j['frame']:06d}\n{name}",fontsize=8);ax.set_xticks([]);ax.set_yticks([])
            if name=="normalized" and path.exists():
                meta=cv2.FileStorage(str(OUT/"replay/diagnostics"/(f"job{j['job']}_refined.json")),cv2.FILE_STORAGE_READ)
                H=meta.getNode("image_to_template").mat();pts=meta.getNode("points").mat();meta.release()
                if H is not None and pts is not None:
                    q=cv2.perspectiveTransform(pts.reshape(-1,1,2),H).reshape(-1,2)
                    ax.scatter(q[:,0],q[:,1],s=10,color="red")
                    for i,point in enumerate(q):ax.annotate(str(i),point,xytext=(2,-3),textcoords="offset points",color="lime",fontsize=7)
    fig.savefig(OUT/"normalization_stages.png",dpi=115);plt.close(fig)
    dump(OUT/"frames.json",dict(selection_policy="Scenario coverage plus an old large error, a new rejection and a residual depth jump; not a random statistical sample.",
        frames=records,replayed_points_equal_frozen_results=True))
    print("Rendered",len(records),"comparison cards, overview and actual C++ normalization stages")


def performance():
    rows=list(csv.DictReader((ROOT/"paired_timing.csv").open()));paired=read(ROOT/"timing_paired.json")
    resources=[]
    for mode in ("current","normalized","refined"):
        log=(OUT/("resources_"+mode+".log")).read_text()
        resource=json.loads(next(line[len("RESOURCE "):] for line in log.splitlines() if line.startswith("RESOURCE ")))
        resource["cpu_ms_per_call"]=resource["cpu_ms"]/resource["calls"];resources.append(resource)
    dump(OUT/"resources.json",dict(scope="One OpenCV thread; identical common-success ROIs; 50 passes per mode; independent process per mode. CPU interval excludes input decoding/warmup. Peak RSS includes libraries, input decoding, cached images and warmup; not algorithm-only allocation.",results=resources))
    modes=("current","normalized","refined");labels=("Old BC","Normalized","+ edge fit")
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x=np.arange(3);width=.36
    axes[0,0].bar(x-width/2,[paired["common_success_calls"][m]["median"] for m in modes],width,label="median")
    axes[0,0].bar(x+width/2,[paired["common_success_calls"][m]["p95"] for m in modes],width,label="P95")
    axes[0,0].set(xticks=x,xticklabels=labels,ylabel="ms",title="Isolated timing: 61 common-success ROIs");axes[0,0].legend()
    bottom=np.zeros(2)
    stages=[("normalize_ms","Normalize"),("corners_ms","Corner fitting"),("refine_ms","Edge refinement"),("validate_ms","H / shape validation"),(None,"Otsu / cleanup / output")]
    means={m:{k:float(np.mean([float(r[k]) for r in rows if r["mode"]==m and r["cls"]=="0"])) for k in ("extract_ms","normalize_ms","corners_ms","refine_ms","validate_ms")} for m in modes[1:]}
    for key,label in stages:
        values=[means[m][key] if key else means[m]["extract_ms"]-sum(means[m][k] for k in means[m] if k!="extract_ms") for m in modes[1:]]
        axes[0,1].bar([0,1],values,bottom=bottom,label=label);bottom+=values
    axes[0,1].set(xticks=[0,1],xticklabels=labels[1:],ylabel="mean ms",title="New pipeline stages: all 816 class0 calls");axes[0,1].legend(fontsize=7)
    axes[1,0].bar(x,[r["cpu_ms_per_call"] for r in resources]);axes[1,0].set(xticks=x,xticklabels=labels,ylabel="process CPU ms / call",title=f"Additional resource probe: {resources[0]['roi_count']} common ROIs")
    axes[1,1].bar(x,[r["peak_rss_mib_including_load_warmup"] for r in resources]);axes[1,1].set(xticks=x,xticklabels=labels,ylabel="MiB",title="Whole-process peak RSS, including loading")
    fig.savefig(OUT/"performance.png",dpi=145);plt.close(fig)
    report()


def report():
    records=read(OUT/"frames.json")["frames"];resources=read(OUT/"resources.json")["results"]
    lines=["# 单目 BC：代表帧可视化与性能开销", "", "本页只展示已经冻结的实验数据；可视化不重新解算位姿。文件名帧号从 0 开始，完整运行 CSV 的 frame 为文件名帧号＋1。", "",
        "## 如何阅读", "", "每张对比图上排：标注 ROI 下独立 BC 提取，青色圆圈为 GT，红色叉为观测角点，黄色线为位置误差。下排：原序列完整 YOLO＋历史状态＋PnP 的记录结果，坐标轴 X 红、Y 绿、Z 蓝，长度均为 100mm，白色虚线为 GT。轴超出裁剪范围时只显示可见部分。两排属于不同实验，不能把独立 BC 拒绝直接当作完整追踪失败。", "",
        "选帧覆盖清晰、倾斜、同几何退化、旋转、大 ROI、超范围倾角、新增拒绝、残余深度跳变，并加入一例旧方法局部大误差。它们是解释用样例，不是随机样本，整体结论仍以全量实验为准。", "",
        "![九帧总览](../output/mono_normalized_20261016/representatives/overview.png)", "", "## 性能开销", "",
        "已有独立交错计时：61 个共同成功 ROI、每组 732 次调用。原 BC 中位数/P95 为 4.016/56.006ms，新精化方案为 2.136/18.910ms。排除了图像读取、YOLO、PnP及可视化。", "",
        f"新增资源探针：{resources[0]['roi_count']} 个相同且共同成功的 ROI，每种方法单独启动进程，预热后重复 50 轮、每组 {resources[0]['calls']} 次。CPU 时间不含读取/预热；峰值 RSS 是整个进程的高水位，包含 OpenCV、读取缓存及预热，不能解读为算法独占内存。", "",
        "| 方法 | CPU ms/次 | 测量区间 wall ms/次 | 单核利用率 | 整进程峰值 RSS（MiB） |", "|---|---:|---:|---:|---:|"]
    for r in resources:lines.append(f"| {r['mode']} | {r['cpu_ms_per_call']:.3f} | {r['wall_ms']/r['calls']:.3f} | {r['single_core_utilization_pct']:.1f}% | {r['peak_rss_mib_including_load_warmup']:.1f} |")
    lines += ["", "单核利用率高表示该区间主要在计算；它不是整台电脑 CPU 占用百分比。资源探针的样本与全量计时样本不同，不能混合计算加速比；未测 GPU 利用率和完整系统 FPS。", "",
        "![耗时、阶段及资源](../output/mono_normalized_20261016/representatives/performance.png)", "",
        "新精化路径的阶段耗时（原独立计时的全部 816 次 class0 调用）：", "",
        "| 阶段 | 中位数 ms | P95 ms |", "|---|---:|---:|"]
    timing=next(r for r in read(ROOT/"timing_summary.json") if r["cls"]==0 and r["mode"]=="refined")
    for key,label in (("normalize_ms","粗归一化及四方向"),("corners_ms","标准空间角点拟合"),("refine_ms","原图边线精化"),("validate_ms","H / 形状验证")):
        s=timing["stages"][key];lines.append(f"| {label} | {s['median']:.3f} | {s['p95']:.3f} |")
    lines += ["", "阶段分位数不能直接相加；完整 BC 还包含 Otsu、连通域、填孔、形态学及结果封装。", "", "## 帧索引", "",
        "| 场景 / 文件帧号 | 倾角 | 角点 RMSE：旧→新（px） | BC 计时中位数：旧→新（ms） | 完整流程策略：旧→新 |", "|---|---:|---:|---:|---|"]
    def fmt(v):return "—" if v is None else f"{v:.2f}"
    for r in records:
        a,b=r["corners"]["current"],r["corners"]["refined"];t=r["timing"]
        time="未在原计时样本中" if "current" not in t else f"{t['current']['median']:.2f} → {t['refined']['median']:.2f}"
        lines.append(f"| {r['sequence']} / {r['frame']:06d} | {r['tilt']:.1f}° | {fmt(a.get('rmse'))} → {fmt(b.get('rmse')) if b['output'] else '拒绝'} | {time} | {r['pose']['current']['strategy']} → {r['pose']['refined']['strategy']} |")
    for r in records:
        lines += ["",f"## {r['sequence']} / frame_{r['frame']:06d}.png：{r['title']}","",
            f"[原始帧](../{r['path']})；倾角 {r['tilt']:.1f}°，class0 跨度 {r['size']:.0f}px。"+("超过当前 PnP 的 30°先验，仅作压力样例。" if r['tilt']>30 else ""),"",
            f"![{r['title']}](../{r['card']})",""]
        for m in ("current","refined"):
            p=r["pose"][m]
            if p["success"]:lines.append(f"- {m}：{p['strategy']}；Z={p['depth_mm']:.2f}mm，GT={p['depth_gt_mm']:.2f}mm；法向误差 {p['normal_error_deg']:.2f}°；深度相对误差 {p['depth_rel_error']*100:.2f}%。")
            else:lines.append(f"- {m}：完整位姿失败。")
        if r["frame"]==75 and r["sequence"]=="oblique_degraded":
            lines += ["", "2026-10-19 复核：完整流程选中的 YOLO 框在背景上，触发 TT 四点，图中上排精确的 BC 十点没有进入该 PnP。改用标注 ROI 且保留 30°先验后，BC 十点 PnP 成功，重投影误差约 0.289px。详见 [失败原因核查](mono_pnp_failure_diagnosis_20261019.md)。"]
        if r["frame"]==60 and r["sequence"]=="grazing":
            lines += ["", "2026-10-19 对照确认：保持 30°先验时拒绝；仅在诊断中关闭先验后，BC 十点求解成功，估计倾角 69.16°。这是超工作范围的约束拒绝。详见 [失败原因核查](mono_pnp_failure_diagnosis_20261019.md)。"]
        if r["frame"]==149 and r["sequence"]=="oblique_degraded":lines += ["", "这是保留的失败案例：完整流程退化到 TT 后仍输出严重错误的深度。新 BC 没有解决该后备路径问题，不能据其他样例宣称所有深度跳变已消除。"]
    lines += ["", "## C++ 实际归一化阶段", "", "下面逐行对应上述九帧，三列依次为前景、四方向粗归一化、H 验证后的归一化结果。拒绝帧缺少后续图像时明确留空。所有图均由实际 C++ 调试输出产生，并核对了重放角点与原冻结 CSV 完全一致。", "",
        "![归一化阶段](../output/mono_normalized_20261016/representatives/normalization_stages.png)","", "完整统计与配置见 [主实验报告](mono_bc_normalized_20261016.md)。"]
    Path("docs/mono_bc_representative_frames_20261016.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Wrote representative-frame report")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","render","performance"])
    args=p.parse_args();globals()[args.action]()
