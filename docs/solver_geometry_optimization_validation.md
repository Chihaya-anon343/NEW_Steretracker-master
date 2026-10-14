# GPnP 与平面匹配过滤优化验证

日期：2026-10-13。实施用户批准的第 1、3 项，模板坐标系第 2 项保持原样。

## 修改内容

- GPnP 改为三维旋转增量＋三维平移的六参数优化，保留 Eigen LM 和原射线／平移先验目标。
- 补齐相机、初值、数组长度、有限值、索引、一对一对应和退化几何检查；不再以最小长度掩盖数组错位。
- 显式记录初值来源。仅数值终止状态 1–4 可进入几何验收；预算耗尽等状态明确拒绝。
- 求解器检查正深度和重投影，日志区分数值终止、候选有效、求解器接受与帧接受；左目回退保留独立来源。
- 新增可开关的平面过滤。水平双目拟合非恒定视差平面；一般外参使用平面单应候选与对称传输残差。
- 主配置启用过滤，门槛 0.25 px，最少 6 点，内点比例 0.7，候选预算 128。旧配置省略开关仍关闭；四点 TT 跳过该检查。
- 几何不确定时保留原逐点判断并标记状态。过滤后重新检查点数、比例和覆盖，保留原角点编号。
- CSV 原十二列不变，诊断图新增平面残差和拒绝原因。未加入可选重匹配或 GPnP 稳健权重。

## Docker 测试

| 测试目标 | 通过／总计 |
|---|---:|
| test_pose_solvers | 16／16 |
| test_stereo_flow | 29／29 |
| test_dual_roi_flow | 17／17 |
| test_disparity_chain | 11／11 |
| 合计 | 73／73 |

新增 13 项注册测试覆盖六参数零／小旋转、非法索引、重复身份、错位数组、非有限初值与观测、非正初始深度、相机参数、预算耗尽、共线模型与退化图像、显式左目来源、倾斜平面、单应过滤、低冗余及不确定共识。特别构造局部受扰的中间点：它通过 LK 逐点质检后被几何过滤剔除，剩余原编号为 0、1、3、4、5、6，没有重排。

这是四组相关专项与兼容测试，不是整个项目测试数量。物理一致的合成投影用于 GPnP 检查，人工裁剪序列不用于物理位姿精度证明。完整日志见 [实验目录](C:/D/Codes/NEW_Steretracker-master_COPY/output/solver_geometry_optimization)。

## 完整序列对照协议

使用同一 150 对数据和模板。保存修改前 ELF 为 GPNP_before，冻结修改后 ELF 为 GPNP_after。两版本分别完整预热一次，随后按 before/run1、after/run1、before/run2、after/run2、before/run3、after/run3 交替测量。

正式运行关闭可视化、开启日志，均为仅视差。精度各取 run1 的 150 帧；耗时各汇总三轮 450 个测量，预热不计入。没有将重复图像当作额外独立场景。运行时没有编译干扰。

[provenance.json](C:/D/Codes/NEW_Steretracker-master_COPY/output/solver_geometry_optimization/provenance.json) 保存 300 张输入、manifest、模型、模板、相关源码、配置和程序哈希，运行前后校验一致。原有配置值及模板构造代码未改变，仅增加五个平面过滤设置。

## 精度与计算代价

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 帧通过 | 150／150 | 150／150 |
| 保留点对 | 1559 | 1558 |
| 视差 MAE | 0.008942 px | 0.008538 px |
| 视差 RMSE | 0.019152 px | 0.010321 px |
| 绝对误差 P95 | 0.0175 px | 0.0175 px |
| 最大绝对误差 | 0.6371 px | 0.0496 px |
| 大于 0.5 px 的点 | 1 | 0 |
| 匹配均值 | 2.385 ms | 2.489 ms |
| 匹配 P95 | 5.656 ms | 5.813 ms |
| 端到端 FPS 三轮均值 | 1.809 | 1.805 |

RMSE 减少 46.11%，匹配均值增加 0.104 ms（4.34%）。这次改动增加几何检查计算，收益是抑制尾部异常，不是加速。端到端 FPS 接近，不据此宣称系统性能提升。

三策略仍分别为 Dual 23 帧、BC 101 帧、TT 26 帧，全部通过。Dual 保留 446 对、TT 保留 104 对；BC 从 1009 降到 1008。退化、全图重试及旧前端回退仍为零，单轮金字塔对数 150、粗搜索尝试 141。

三轮每轮均左 YOLO=150、右 YOLO=0、PnP=0。两版本各有 300 次重复 CSV 比较，均逐字节一致。[完整统计](C:/D/Codes/NEW_Steretracker-master_COPY/output/solver_geometry_optimization/comparison.json)

## 异常点及兼容核对

唯一变化为第 29 帧原角点 9 被删除；其余 1558 行的坐标、视差、编号、区域、模板字段、策略与帧标志逐字段一致，没有新增对应。误差改善来自异常点剔除，保留点的位置未重新优化。

第 29 帧仍通过，匹配由 10 对变为 9 对。该点 FB=0.628563 px、NCC=0.963796，平面残差约 0.637501 px，被标记为 planar_geometry_outlier。其余点编号仍为 0–8。独立 debug 两次结果一致，但不计入正式计时。

![第29帧过滤诊断](C:/D/Codes/NEW_Steretracker-master_COPY/docs/assets/algorithm_v1_3/frame29_flow_status.png)

记录与核对见 [verification.json](C:/D/Codes/NEW_Steretracker-master_COPY/output/solver_geometry_optimization/verification.json)。模板坐标系、模板尺寸、面积阈值和已有配置值保持原样。

## 门槛重放

对原版已通过逐点检查的 1559 对点按区域重放真实过滤函数，一次预热、十轮计时：

| 门槛 | 保留／剔除 | 每区域过滤均值 |
|---|---:|---:|
| 0.10 px | 1558／1 | 0.0612 ms |
| 0.20 px | 1558／1 | 0.0598 ms |
| 0.25 px | 1558／1 | 0.0584 ms |
| 0.30 px | 1558／1 | 0.0626 ms |
| 0.50 px | 1559／0 | 0.0626 ms |

重放读取已舍入的 CSV 像素坐标，其误差数字与原始匹配中的 disparity_px 存在细小差异；正式精度采用完整程序对照。重放不含图像、YOLO、LK 或 GPnP，不能将每区域时间当作端到端延迟。[重放原始结果](C:/D/Codes/NEW_Steretracker-master_COPY/output/solver_geometry_optimization/threshold_ablation.csv)

0.25 px 是本次初始设置，不代表真实模糊和标定误差场景的最优门槛。0.5 px 的模型能吸收该异常，因此模型残差门槛不等同于对真值误差直接截断。

## 复现

Windows 主机经 Git Bash 使用经过挂载检查的项目 Docker 包装脚本；此环境全局脚本未找到，采用同功能的 scripts/docker-toolchain.sh。

```text
python scripts/compare_solver_geometry_optimization.py
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/benchmark_planar_geometry output/stereo_mixed_strategy_150/run1 output/solver_geometry_optimization/benchmark_manifest.json
```

benchmark_manifest.json 是从原 manifest 提取 frames/disparity 的等价数值快照；OpenCV 4.5.4 的 JSON 读取器不支持原文件的 null，因此未直接用它读取原 manifest。原数据集未修改。

## 剩余边界

- 四点 TT 缺乏该过滤所需的冗余，继续原四点全部通过规则。
- 无可靠几何共识只标记不确定，不构成平面验收通过的证据。
- 水平分支检查视差平面残差，垂直方向仍由原极线、FB、NCC 等保护；不是完整逐点真值保证。
- 逐点质量权重、可选重匹配及模板坐标统一未在本次实施。
- 真实相机标定、同步、物理尺度与位姿准确度仍需独立实拍验证。
