#!/usr/bin/env python3
"""Write a TT P0/P1 report strictly from completed archived artifacts."""
import argparse
import json
from pathlib import Path

ROOT = Path("output/tt_priority_20261019")


def read(name, default=None):
    path = ROOT / name
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def number(value, digits=3):
    return "—" if value is None else ("%.*f" % (digits, value))


def stat(value):
    return number(value["median"]) + " / " + number(value["p95"])


def main(args):
    experiment = read("experiment.json")
    audit = read("model_audit.json")
    poses = read("pose_summary.json", [])
    compare = read("pose_comparison.json", [])
    corners = read("corner_summary.json", [])
    corner_compare = read("corner_comparison.json", [])
    timing = read("timing_summary.json", [])
    verify = read("verification.json")
    exact = read("diagnostic_only_comparison.json", [])
    exact_corner = read("diagnostic_only_corner_comparison.json", {})
    frames = read("pose_frames.json", [])
    modes = experiment["modes"]
    scopes = {"all": "全部倾角", "tilt_le30": "≤30°", "tilt_30_50": "30～50°", "tilt_gt50": ">50°"}
    lines = ["# TT P0/P1 优化与消融测试", "", "2026-10-19～2026-10-20。固定 ROI 来自既有六段合成数据；完整流程的序列范围见下文。不重新生成数据，不改变 BC 算法、模板物理尺寸或人工中心偏移。", "",
        "最终启用G：四方向编号、受限精化和严格TT入口。class0固定ROI编号异常诊断11→0，输出158→154；增加了提取耗时。完整流程有效位姿448→446，共同相邻帧对的法向跳变98→91、深度跳变1→1。低分数放行因引入新的严重尺度错误而默认关闭，尚未解决所有位姿异常。", "",
        "## 修改与版本", "",
        "| 版本 | 行为 |", "|---|---|",
        "| A | 修改前冻结主程序与冻结 TT 源码 |",
        "| B | 新程序，仅开启 TT 诊断；原有接受规则和角点输出不变 |",
        "| C | B＋单目 TT 专用 ROI 验证和实际 TT 退化链检查 |",
        "| D | C＋四方向粗归一化匹配与明确编号；歧义时拒绝 |",
        "| E | D＋有窗口、位移、梯度与四边形约束的亚像素精化 |",
        "| F | E＋独立的缩放像素中心映射修正消融 |",
        "| G | 最终保守候选：四方向＋受限精化，保持0.75严格入口，关闭实验性的低分数放行 |", "",
        "所有新距离阈值使用原 ROI 像素。稳健前端仅在显式单目上下文启用，保留现有双目提取行为。",
        "TT 专用验证保留原硬门限 0.75；仅在小目标分组（class0≤40px、class1≤24px）允许进入 0.65～0.75 的候选区间，仍须通过独立方向分数、方向分差和四点几何等检查。",
        "C～F 的低分数放行是探索性设置；它会引入类别/物理尺度混淆，最终G使用 `allow_small_target_path=false`。这些结果来自同一既有合成套件，尚不能视作独立实拍泛化验证。", "",
        "## 四点参考与物理模型审计", "",
        "四点参考来自源图十个标注点的轴对齐包围框，再使用生成器每帧保存的 H 投影。**它们是虚拟包围框顶点，不是四个实际可见物理角点。**",
        "下文的 virtual corner 误差只衡量 TT 观测与这一参考模型的偏差。最佳重排误差仅用于识别编号异常，不会修改待测算法输出或降低固定编号误差。", "",
        "| 类别 | 合成虚拟框宽×高 mm | TT 配置正方形 mm | 配置相对合成宽/高差异 |",
        "|---|---|---:|---|"]
    for item in audit["classes"]:
        lines.append("| class%d | %.4f × %.4f | %.3f | %+.2f%% / %+.2f%% |" % (
            item["cls"], *item["virtual_width_height_mm"], item["tt_square_side_mm"], *item["discrepancy_percent"]))
    lines += ["", "源图按 %.9f / %.9f mm/px 映射到生成器平面。合成 class1 相对 class0 中心为 %s mm，配置为 %s mm；本轮保留配置，不自动消除该差异。" % (
        *audit["source_mm_per_px"], audit["rendered_class1_offset_mm"], audit["configured_class1_offset_mm"]),
        "因此，尤其是 class1 的绝对深度及位置偏差不能全部归因于角点定位，也不能通过调参角点使错误尺寸模型获得较好 GT 误差。", "",
        "## 固定 ROI 提取", "",
        "使用源标注生成、裁剪到图像边界且面积≤%dpx²的 %d 个 TT ROI，来自全部六个序列并包含 class0/class1。固定 ROI 不使用 YOLO，不运行 PnP，不混入策略选择收益。" % (experiment["max_roi_area"], experiment["quality_jobs"]),
        "仅完整可见分组汇总如下；误差统计仅包含对应版本接受的四点输出，必须结合后面的共同成功集合比较。", "",
        "| 版本 | 类别 | 可见尝试 | 成功 | 编号异常诊断 | 虚拟点误差中位数/P95 px | 四点均≤2px帧 |",
        "|---|---:|---:|---:|---:|---|---:|"]
    for item in corners:
        if item["size"] == "all":
            lines.append("| %s | %d | %d | %d | %d | %s | %d |" % (
                item["mode"], item["cls"], item["attempts"], item["success"],
                item["numbering_mismatch_diagnostic"], stat(item["virtual_corner_error_px"]), item["all_virtual_corners_within2"]))
    lines += ["", "编号异常的诊断条件：最优空间分配不同于固定编号，且固定编号 RMSE 比最优分配 RMSE 大于 max(1px, 目标跨度×1%)。此项是诊断特征，不是 ID 真值的完整证明。", "",
        "共同成功集合（A 与对应版本均输出四点），用于避免拒绝困难帧产生的选择偏差：", "",
        "| 对比 A→ | 类别 | 配对可见帧 | 共同成功 | 恢复/丢失 | 共同虚拟点误差中位数/P95 px：A→新 | 共同编号异常：A→新 |",
        "|---|---:|---:|---:|---|---|---|"]
    for item in corner_compare:
        lines.append("| %s | %d | %d | %d | %d / %d | %s → %s | %d → %d |" % (
            item["mode"], item["cls"], item["paired_visible_attempts"], item["common_success"], item["recovered"], item["lost"],
            stat(item["common_virtual_error_before"]), stat(item["common_virtual_error_after"]),
            item["common_mismatch_before"], item["common_mismatch_after"]))
    lines += ["", "## YOLO 完整流程", "",
        "完整流程序列：%s；每个版本 %d 帧。" % ("、".join(sorted({j["sequence"] for j in experiment["jobs"]})),
            sum(j["frames"] for j in experiment["jobs"] if j["mode"] == "A")),
        "各版本使用相同相机、YOLO、策略阈值、物理模型与时序设置。先验启用=%s，倾角上限=%s°，temporal.enabled=%s；不把本轮50°结果直接与历史30°统计作前后对比。" % (
            experiment["prior"].get("enabled"), experiment["prior"].get("max_tilt_deg"), experiment["temporal"].get("enabled")),
        "下表按真实倾角分组。成功数包含既有 BC/DualROI 等策略，TT 变化可通过恢复/丢失帧与逐帧策略表核对。", "",
        "| 版本 | 倾角 | 帧数 | 有效位姿 | 深度跳变 | 法向跳变 | 深度相对误差中位数/P95 | 法向误差中位数/P95 ° |",
        "|---|---|---:|---:|---:|---:|---|---|"]
    for item in poses:
        lines.append("| %s | %s | %d | %d | %d | %d | %s | %s |" % (
            item["mode"], scopes[item["scope"]], item["frames"], item["success"], item["depth_jumps"], item["normal_jumps"],
            stat(item["depth_rel_error"]), stat(item["normal_error_deg"])))
    lines += ["", "深度相对误差使用比例（0.10表示10%）。跳变沿用已有实验定义：只计相邻两个源帧均成功的帧对。深度跳变要求估计相对步长>0.30且超过真值步长0.20；法向跳变要求估计夹角>20°且超过真值夹角15°。失败造成的帧间缺口另行保存，不计入这两个相邻跳变数。", "",
        "| 对比 A→ | 配对帧 | 共同成功 | 恢复 | 丢失 | 共同法向误差 A→新 | 共同深度误差 A→新 |",
        "|---|---:|---:|---:|---:|---|---|"]
    for item in compare:
        lines.append("| %s | %d | %d | %d | %d | %s → %s | %s → %s |" % (
            item["mode"], item["paired_frames"], item["common_success"], len(item["recovered"]), len(item["lost"]),
            stat(item["common_normal_error_before"]), stat(item["common_normal_error_after"]),
            stat(item["common_depth_error_before"]), stat(item["common_depth_error_after"])))
    lines += ["", "相同且两端都成功的相邻帧对，排除新增断帧对跳变计数的影响：", "",
              "| 对比A→ | 共同相邻帧对 | 法向跳变 A→新 | 深度跳变 A→新 |", "|---|---:|---|---|"]
    for item in compare:
        lines.append("| %s | %d | %d → %d | %d → %d |" % (item["mode"], item.get("common_adjacent_pairs", 0),
            item.get("common_adjacent_normal_jumps_before", 0), item.get("common_adjacent_normal_jumps_after", 0),
            item.get("common_adjacent_depth_jumps_before", 0), item.get("common_adjacent_depth_jumps_after", 0)))
    lines += ["", "单独统计深度相对误差超过100%的严重异常，避免被成功率和中位数掩盖：", "",
              "| 版本 | 严重深度异常数 | 新恢复帧中的严重深度异常 |", "|---|---:|---:|"]
    for item in poses:
        if item["scope"] == "all":
            comparison = next((r for r in compare if r["mode"] == item["mode"]), {})
            lines.append("| %s | %d | %d |" % (item["mode"], item.get("depth_error_over100pct", 0),
                         comparison.get("recovered_depth_error_over100pct", 0)))
    lines += ["", "各序列有效数：", "", "| 序列 | 版本 | 帧数 | 有效位姿 | 成功策略 |", "|---|---|---:|---:|---|"]
    for mode in modes:
        for sequence in sorted({r["sequence"] for r in frames if r["mode"] == mode}):
            selected = [r for r in frames if r["mode"] == mode and r["sequence"] == sequence]
            counts = {}
            for r in selected:
                if r["success"]:
                    counts[r["strategy"]] = counts.get(r["strategy"], 0) + 1
            lines.append("| %s | %s | %d | %d | %s |" % (
                sequence, mode, len(selected), sum(r["success"] for r in selected), counts))
    lines += ["", "## 诊断兼容与耗时", "",
        "A/B 完整流程比较仅比较共有的非计时字段；新增诊断列不参与旧版本等值判定。", "",
        "| 序列 | 帧数 | 非计时字段差异数 |", "|---|---:|---:|"]
    for item in exact:
        lines.append("| %s | %d | %d |" % (item["sequence"], item["frames"], item["differences"]))
    lines += ["", "固定 ROI 的 A/B 比较：%d / %d 次调用，成功状态、点数、坐标、编号及成功调用的角度/IoU差异数为 %d；另有 %d 次失败调用清除了旧的角度/IoU残留。" % (
        exact_corner.get("rows_before", 0), exact_corner.get("rows_after", 0), exact_corner.get("difference_count", 0),
        len({row[0] for row in exact_corner.get("failed_call_stale_state_resets", [])})),
        "主配置状态：**%s**。" % ("实验结束后仅新增已验证的G版本两块配置，其他配置与冻结前相同" if verify.get("root_config_verified_activation") else
            ("逐字节保持冻结值" if verify["root_config_unchanged"] else "发生未验证的变化，需核对来源")), ""]
    lines += ["固定 ROI 等值比较覆盖成功状态、点数、坐标、编号，以及成功调用的角度/IoU。原版失败调用会残留上次成功的角度和IoU；新版清空这两项，作为已修复的状态缺陷单独列出，不将其算成角点数值变化。", ""]
    if (ROOT / "image_cache.complete.json").exists():
        cache = read("image_cache.complete.json")
        lines += ["重复实验使用容器内 BMP 解码像素缓存。缓存共 %d 张图，逐张与原 PNG 的解码像素作完全一致校验；未对图像进行插值、缩放或有损压缩。缓存清单及运行时路径单独归档，原数据和冻结配置不覆盖。" % cache["images"], ""]
    if timing:
        lines += ["独立计时使用相同抽样 ROI，单 OpenCV 线程、每 ROI 先预热，按轮交替版本顺序串行运行。耗时含 TT 提取与诊断，排除图像解码、CSV写出、模型初始化和最终可视化写盘。", "",
            "| 版本 | 类别 | 调用数 | 提取中位数/P95 ms | 各轮均值 ms |", "|---|---:|---:|---|---|"]
        for item in timing:
            lines.append("| %s | %d | %d | %s | %s |" % (
                item["mode"], item["cls"], item["calls"], stat(item["extract_ms"]),
                " / ".join(number(v) for k, v in sorted(item["round_mean_ms"].items()))))
    else:
        lines += ["未执行独立重复计时。完整流程 CSV 的耗时用于发现显著回退，不能据此宣称稳定加速或整体 FPS 提升。"]
    selected = next((r for r in poses if r["mode"] == args.selected and r["scope"] == "all"), None)
    selected_comparison = next((r for r in compare if r["mode"] == args.selected), None)
    lines += ["", "## 结论与边界", ""]
    if selected and selected_comparison:
        lines += ["候选 %s 在本轮 %d 帧中输出 %d 个有效位姿，相对 A 恢复 %d 帧、丢失 %d 帧；深度跳变 %d 次、法向跳变 %d 次。是否采用须同时审查恢复帧的误差、共同成功集合和拒绝帧，不能只比较成功率。" % (
            args.selected, selected["frames"], selected["success"], len(selected_comparison["recovered"]), len(selected_comparison["lost"]),
            selected["depth_jumps"], selected["normal_jumps"])]
    if args.selected == "G":
        lines += ["G新增拒绝的两帧为 oblique_degraded 第123帧（原TT，法向误差约5.25°、深度误差13.80%）和第158帧（原BC，约2.95°、25.95%）。不能将它们都称为错误位姿；后者在ROI验证阶段使用TT作为候选证据，入口拒绝后BC也失去执行机会。", "",
            "深度跳变总数2→1中的减少发生在第123帧失效后，第124帧不再组成相邻成功帧对。共同418个相邻成功帧对的深度跳变仍为1→1，**不能将总计下降解释为修复深度跳变**；同一集合的法向跳变98→91。", "",
            "探索版E相对A恢复12帧、丢失2帧，但恢复帧中3帧是内图案被当作class0后的严重尺度错误。因此最终保留严格入口，实验性低分数放行默认关闭；这些设置没有按帧号写特例。", ""]
    visual_path = ROOT / "visualization_verification.json"
    if visual_path.exists():
        lines += ["## 代表性四点与三维轴", "",
            "以下是标注ROI下强制TT的单帧冷启动诊断，两侧均使用真实C++位姿，不代替上面的YOLO完整序列结果。彩色圆圈及数字为提取点，绿色加号为虚拟参考；轴为红+X、绿+Y、蓝−Z朝上，100mm。", ""]
        for example in read("visualization_verification.json"):
            before, after = example["measurements"]
            lines += ["%s：虚拟参考四点RMSE %.2f→%.2fpx。" % (example["name"], before["virtual_corner_rmse_px"], after["virtual_corner_rmse_px"]), "",
                      "![%s](../%s)" % (example["name"], example["figure"]), ""]
        lines += ["新增原生精化面板展示初值、输出点、粗归一化图和规范模板。显示补边只用于展示ROI外的虚拟顶点，不参与提取或PnP。", ""]
    lines += ["本轮保留最小外接矩形的物理近似。四点接近虚拟参考角点、低重投影误差均不保证法向准确；强透视下的模型偏差与低分辨率不可辨性仍存在。class1 尺寸和中心配置差异保持原样。", "",
        "F 是单独消融，不应在未审查结果时与其他修改一起默认开启。序列时序约束也没有加入本轮变量。", "",
        "## 复现与证据", "",
        "- [实验冻结与版本定义](../output/tt_priority_20261019/experiment.json)",
        "- [物理模型审计](../output/tt_priority_20261019/model_audit.json)",
        "- [固定 ROI 分组结果](../output/tt_priority_20261019/corner_summary.json)",
        "- [固定 ROI 共同成功集合](../output/tt_priority_20261019/corner_comparison.json)",
        "- [完整流程分组结果](../output/tt_priority_20261019/pose_summary.json)",
        "- [恢复、丢失和共同成功集合](../output/tt_priority_20261019/pose_comparison.json)",
        "- [逐帧测量与拒绝信息](../output/tt_priority_20261019/pose_frames.json)",
        "- [验证摘要](../output/tt_priority_20261019/verification.json)", "",
        "- [专项及兼容回归日志](../output/tt_priority_20261019/ctest.log)",
        "- [最终保守入口专项回归](../output/tt_priority_20261019/final_tt_ctest.log)",
        "- [主配置启用记录](../output/tt_priority_20261019/production_activation.json)", "",
        "- [归档校验与最终显示一致性](../output/tt_priority_20261019/final_verification.json)", "",
        "```bash", "# 主程序和两版评估器构建完成后冻结，不覆盖已存在归档", "python scripts/experiment_tt_priority.py prepare --sequences all",
        "scripts/docker-toolchain.sh python3 scripts/experiment_tt_priority.py run --sequences all",
        "scripts/docker-toolchain.sh python3 scripts/experiment_tt_priority.py quality",
        "# 独立计时需等构建、重放和回归测试结束", "scripts/docker-toolchain.sh python3 scripts/experiment_tt_priority.py timing --modes A G --evaluator output/tt_priority_20261019/evaluate_tt_priority_final --repeats 8",
        "python scripts/experiment_tt_priority.py analyze", "python scripts/report_tt_priority.py --selected " + args.selected, "```", ""]
    target = Path(args.output)
    target.write_bytes(("\n".join(lines)).encode("utf-8"))
    print("Wrote", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", default="G", choices=tuple("ABCDEFG"))
    parser.add_argument("--output", default="docs/tt_priority_optimization_20261019.md")
    main(parser.parse_args())
