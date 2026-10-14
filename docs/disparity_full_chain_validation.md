# 完整仅视差链验证结果

2026-10-13，Docker Release、OpenCV 4.5.4。主配置新增 `disparity_full_chain=true`，主数据集、面积阈值及其它用户设置保留；实验配置和产物放在 `output/disparity_full_chain_experiment/`。

## 构建与回归

主程序构建成功。最终源代码对应的测试顺序运行，日志保存在实验目录：

| 测试组 | 通过 |
|---|---:|
| 普通光流、ROI、几何、旧接口与位姿兼容 | 23/23 |
| Dual 原功能与新增左 TT／回退开关／负视差终止 | 17/17 |
| 完整仅视差链 | 11/11 |
| 合计 | **51/51** |

新视差链覆盖直接 TT、BC 匹配失败→TT、三角形仅三个 BC 点导致提取失败→TT、独立 class1 的 BC/TT/BC→TT、参数恢复、已提供右 ROI 的接口、无效输入、遮挡失败、可靠负视差终止、旧开关默认、时序切换和配置解析。旋转测试包含 12/16/24 px 目标，0°/25°/90°，视差 23.5 px，垂直偏移 -1.774 px。

Dual 左 TT 测试要求无需调用右 ROI 提供函数就成功，并验证一帧一日志。tier1-only、关闭 class1 回退以及负视差时不调用 TT／右检测；纯左 TT 失败时仍可按允许的旧开关延迟获取右 ROI。原位姿测试继续通过。

仅视差结果均检查 PnP 时间为零、位姿成功标志为 false、来源为 Unspecified。连续处理后仍为无成功位姿缓存状态。

## 现有 60 对数据

用最终主程序和独立 `ordinary.json` 批量处理原 `data/stereo_crop`，关闭可视化，保留日志：

| 指标 | 结果 |
|---|---:|
| 帧匹配通过 | **60/60** |
| 保留匹配点 | **598** |
| 视差 MAE | **0.009003 px** |
| 最大绝对误差 | **0.0363 px** |
| 左／右 YOLO 次数 | **60／0** |
| PnP 平均时间 | **0 ms** |

对照此前 `output/dual_roi_flow_experiment/ordinary_regression` 的 598 对点，匹配数量和原编号保持一致。CSV 新增元数据，不再逐文件字节对比。原六列共有 13 个数值单元格末位变化：其中 7 个视差，最大差异 0.0001 px；左右坐标最大差异 0.001 px。原因是共用搜索 ROI 包含 BC 与 TT 的 padding，金字塔边界略有变化。精度没有实际退化。

逐点对照见 `ordinary_comparison.csv`，汇总见 `ordinary_summary.json`；最终日志新增“视差匹配成功: 60 (100%)”，位姿成功率仍为 0，二者独立。

## 小目标矩阵

使用带随机纹理的合成方形目标，尺寸 8/12/16/24/32/48 px，视差 5.5/35.25/75.5 px，垂直偏移 -1.774/0/+1.774 px，共 **54 组**。所有组要求四点全部通过，并使用原 FB、NCC、极线、唯一性和覆盖检查。

结果 **54/54** 通过，各四点，最大视差绝对误差 **0.044983 px**。这是受控合成对应验证，不证明真实极小目标的成功率。矩阵原始结果在 `tiny/matrix.csv`。

## TT 前端对照

同一合成输入、图像常驻内存，目标尺寸 16/24/48 px，视差 35.25 px、垂直偏移 +1.774 px。各路径预热 8 次，然后各测 30 次，交替顺序；最终测量未并行运行构建或其它实验。

基线为“左右 TT 提取＋相同匹配质量复核”，右 ROI 用已知几何提供；新路径为完整仅视差执行器的“左 TT＋右图光流”。两者都必须四点通过。该对照不是未加质检的旧 TT 原始提取耗时，也不包含 YOLO、磁盘读取、可视化或融合。

| 目标尺寸 | 双侧 TT＋复核平均 | 左 TT 光流平均 | 平均耗时减少 | 双侧／左侧 P95 |
|---|---:|---:|---:|---:|
| 16 px | 3.978 ms | 2.170 ms | 45.4% | 5.952／4.120 ms |
| 24 px | 3.764 ms | 2.229 ms | 40.8% | 5.930／3.595 ms |
| 48 px | 5.190 ms | 2.983 ms | 42.5% | 7.459／4.066 ms |

所有测量各 30/30 通过，每次四点；两路径误差接近，最大约 0.027 px。原始耗时和误差在 `tiny/samples.csv`，汇总在 `tiny/summary.json`。这些数据只说明当前微基准中的收益，不代表项目整体 FPS 或其它设备。

## 大目标强制 TT 的限制测试

独立 `forced_tt.json` 将 tiny 阈值临时提高到 200 万，仅处理原数据前 12 帧，以验证主程序入口能直接选择 TT。**主配置阈值仍为 900，未改变。**

这组大目标原本适合 BC，强制 TT 后仅 **1/12** 帧通过，其余因不足四对合格匹配拒绝；每帧均无 PnP，左 YOLO 12、右 0。失败帧的 CSV 使用 `frame_match_ok=0`，保留的部分匹配不算成功。

因此，补齐策略链表示 TT 能被执行并经过质量检查，不表示 TT 能接管所有 BC 失败场景。TT 使用最小外接矩形四角点；对于不规则大目标，矩形角点可能没有稳定图像纹理。当前没有通过降低四点或光度门槛提高表面成功率。真实小目标、照明差异、运动模糊与遮挡的成功率仍需真实双目序列验证。

## 可视化与复现

已检查 `preview/tiny_stereo_f21.png`：四对点编号、左右连线、PnP skipped 和 frame_match=OK 显示完整，无位姿轴或投影。逐点状态图为 `preview/tiny_flow_status_f21.png`。原 BC 和 Dual 可视化接口测试继续通过。

```bash
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/test_disparity_chain
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/test_stereo_flow
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/test_dual_roi_flow
bash scripts/docker-toolchain.sh env ST_CHAIN_EXPERIMENT=1 ./build-codex-bc-bench/tests/test_disparity_chain
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/GPNP output/disparity_full_chain_experiment/ordinary.json
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/GPNP output/disparity_full_chain_experiment/forced_tt.json
```

完整使用说明见 [完整仅视差策略链](disparity_full_chain.md)。
