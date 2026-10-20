# -*- coding: utf-8 -*-
"""Produce the reviewable report from completed contour experiment artifacts."""
import json
import hashlib
from pathlib import Path

ROOT=Path("output/contour_scan_20261019")
def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8-sig"))

quality=read("quality_comparison.json")
pipeline=read("pipeline_comparison.json")
visuals=read("visual_comparison.json")
timing=read("timing_class_summary.json")
assert quality["field_differences"]==0
assert all(r["difference_count"]==0 for r in pipeline)
assert all(r["identical"] for r in visuals)
config_hash=hashlib.sha256(Path("config/tracker_config.json").read_bytes()).hexdigest()
assert config_hash==read("before_hashes.json")["config/tracker_config.json"]
lookup={(r["extractor"],r["cls"],r["capture"],r["version"]):r for r in timing}
table=[]
round_table=[]
for kind in ("BC","TT"):
    for cls in (0,1):
        for capture in (0,1):
            key=(kind,cls,capture)
            if (*key,"before") not in lookup:
                continue
            a,b=lookup[(*key,"before")],lookup[(*key,"after")]
            gain=100*(1-b["median_ms"]/a["median_ms"])
            table.append("| %s | %d | %s | %d | %.3f → %.3f | %.3f → %.3f | %+.1f%% |" % (
                kind,cls,"开" if capture else "关",a["calls"],a["median_ms"],b["median_ms"],
                a["p95_ms"],b["p95_ms"],gain))
            deltas=[100*(1-y/x) for x,y in zip(a["round_mean_ms"],b["round_mean_ms"])]
            round_table.append("| %s | %d | %s | %s |" % (
                kind,cls,"开" if capture else "关"," / ".join("%+.1f%%"%v for v in deltas)))
frames=sum(r["frames"] for r in pipeline)
body="""# 减少轮廓重复扫描：修改与测试报告

2026-10-19。只调整特征提取内部的数据复用和遍历组织，不调整 ROI/图案阈值、可靠点数量、模板尺度、位姿先验或策略链。

## 修改

1. **TT 包围框复用。** 最大连通域仍按原像素面积选取；包围框直接读取同次连通域统计的 `stats`。删除最大域掩码构造、`findNonZero` 及其后的白色像素遍历。保持包含端点、中心取整、正方形大小和图像边界截断规则不变。
2. **BC 最终轮廓复用。** 归一化前端的一次调用内持有最终密集轮廓、包围框及按需计算的最小外接矩形。调试轮廓从该密集链按方向变化压缩，不再对同一最终掩码执行第二次 `findContours`；保留简化轮廓表示。`source_contour` 仅在开启调试采集时返回。
3. **BC 边缘支持点合并遍历。** 预计算每条边的参数，一次按顺序遍历密集轮廓，为各边收集支持点。支持点仍可属于多条边，各边内部顺序与旧版相同；直线拟合、交点与位移检查不变。几何判定量仍为 O(轮廓点数×边数)，主要减少重复读取和重复计算。

已有归一化流程本来就会把粗 H 和四方向匹配结果传给后续验证；本次没有重复实现这部分缓存。填洞前与形态学处理后的轮廓继续分别提取，BC→TT 不共享清理图。上下文不跨帧、不跨 ROI、不跨左右图和类别。

## 数值及兼容性验证

- Docker Release 主程序与五个相关回归测试目标构建通过。CTest 五目标全部通过：角点身份、提取器、DualROI、视差策略链及单目位姿约束。角点测试含新增轮廓复用测试，共 15 项通过。
- 固定 ROI 使用既有六段合成数据的 2,040 个标注 ROI，BC 全部运行，TT 在面积不超过 4,096px²的 ROI 上运行；两者均测试调试采集开/关。**每个版本 %d 次调用，前后对应字段差异为 0**。核对成功状态、角点坐标、匹配编号、角度/IoU、假设/拒绝点统计和轮廓调试栅格摘要；耗时字段不参与等值比较。
- 完整单目流程重放 reference、normal_ambiguity、oblique_degraded，**每版 %d 帧**；策略、类别、匹配数、可靠点统计、拒绝原因、ROI 选择、旋转矩阵和三维位置的 CSV 字段逐项相同。本轮有效位姿数及其轨迹保持不变，不是一次恢复率修复。
- 三组代表帧（ROI 恢复、错误深度恢复、八点 BC）共六张轮廓面板 PNG 前后文件字节完全一致。八点示例沿用先前手动标注 ROI、独立配置强制 BC 的诊断设置；另外两组使用实际 YOLO。

完整流程各序列：

| 序列 | 每版帧数 | 每版有效位姿 | 数值/状态字段差异 |
|---|---:|---:|---:|
%s

## 独立耗时

相同 85 个 ROI（68 个 class0、17 个 class1），TT 只运行其中面积≤4,096px²的子集。独立基准两版均使用 `g++ -O2` 编译；单 OpenCV 线程、每个 ROI/提取器/采集状态先预热，再重复四次；四轮串行运行，前后版本次序交替。独立计时期间没有同时运行构建、回归、序列重放或固定 ROI 对照。

计时包含提取和调试快照采集，排除图像解码、模型初始化、CSV 输出和结果校验哈希；不包含主程序多面板绘制及 PNG 写盘。下表每版调用数覆盖四轮；“中位数下降”正值表示耗时减少，负值表示增加。class1 BC 保持原前端，不能把它当作归一化前端收益。

| 提取器 | 类别 | 调试采集 | 每版调用数 | 中位数 ms（前→后） | P95 ms（前→后） | 中位数下降 |
|---|---:|---|---:|---|---|---:|
%s

四轮各自平均耗时的下降幅度，用于观察波动：

| 提取器 | 类别 | 调试采集 | 第1 / 2 / 3 / 4轮 |
|---|---:|---|---|
%s

总体变化需结合分组和轮间波动解释，不据此宣称整体 FPS 稳定提升；YOLO 推理未修改。本轮未测 RSS 或 GPU 占用。

## 复现与归档

实验目录 `output/contour_scan_20261019` 独立于原有鲁棒性实验；保留修改前源文件、头文件、配置和二进制，以及修改后基准程序/主程序。`before_hashes.json`、`after_hashes.json` 保存校验信息。主配置未由本轮修改。

- [固定 ROI 前后比较](../output/contour_scan_20261019/quality_comparison.json)
- [完整流程比较](../output/contour_scan_20261019/pipeline_comparison.json)
- [可视化比较](../output/contour_scan_20261019/visual_comparison.json)
- [分组计时](../output/contour_scan_20261019/timing_class_summary.json)
- [按 ROI 面积分组的计时](../output/contour_scan_20261019/timing_summary.json)
- [CTest 日志](../output/contour_scan_20261019/ctest.log)

```bash
scripts/docker-toolchain.sh bash scripts/build_contour_scan_benchmark.sh before
scripts/docker-toolchain.sh bash scripts/build_contour_scan_benchmark.sh after
scripts/docker-toolchain.sh python3 scripts/experiment_contour_scan.py quality
scripts/docker-toolchain.sh python3 scripts/experiment_contour_scan.py pipeline
scripts/docker-toolchain.sh python3 scripts/experiment_contour_scan.py visuals
# 等上面的任务全部结束，再独立计时
scripts/docker-toolchain.sh python3 scripts/experiment_contour_scan.py timing
python scripts/experiment_contour_scan.py summarize
python scripts/report_contour_scan.py
```

已存在的实验输出会被运行脚本拒绝覆盖；重新实验需要单独归档/调整输出目录。基准编译亦不应覆盖已完成实验归档。
""" % (quality["calls_per_version"],frames,
    "\n".join("| %s | %d | %d | %d |"%(r["sequence"],r["frames"],r["successes"],r["difference_count"]) for r in pipeline),
    "\n".join(table),"\n".join(round_table))
bc0=lookup[("BC",0,0,"before")];bc1=lookup[("BC",0,0,"after")]
tt0=lookup[("TT",0,0,"before")];tt1=lookup[("TT",0,0,"after")]
conclusion=("\n本轮数值输出保持一致。关闭调试采集时，class0 TT 中位耗时 %.3f→%.3fms，下降 %.1f%%；"
            "class0 BC 为 %.3f→%.3fms，下降 %.1f%%，幅度较小且存在轮间波动。\n" % (
                tt0["median_ms"],tt1["median_ms"],100*(1-tt1["median_ms"]/tt0["median_ms"]),
                bc0["median_ms"],bc1["median_ms"],100*(1-bc1["median_ms"]/bc0["median_ms"])))
body=body.replace("## 修改",conclusion+"\n## 修改",1)
refine=("\n边缘精化子阶段单独看：class0 关闭调试的中位耗时 %.4f→%.4fms，"
        "开启调试为 %.4f→%.4fms。**合并遍历没有测出独立加速收益**；"
        "不能把 BC 的整体波动归因于这一步。该组织方式保留了顺序读取及结果一致性，"
        "但仍需逐边几何判断。class1 BC 未改前端仍有明显轮间波动，也说明小幅总耗时差异需谨慎解释。\n" % (
            bc0["refine_median_ms"],bc1["refine_median_ms"],
            lookup[("BC",0,1,"before")]["refine_median_ms"],
            lookup[("BC",0,1,"after")]["refine_median_ms"]))
body=body.replace("总体变化需结合",refine+"\n总体变化需结合",1)
Path("docs/contour_scan_optimization_20261019.md").write_bytes(body.encode("utf-8"))
print("Wrote docs/contour_scan_optimization_20261019.md")
