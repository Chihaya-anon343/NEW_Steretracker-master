# 单目 PnP：单面靶板与 30°近垂直观测先验

日期：2026-10-16。场景由用户确认：硬板单面有图案、背面贴地，相机安装在无人机上并近似垂直向下；总倾角上限取 **30°**，保留运动与安装误差余量。本次沿用既有模板坐标、角点提取、物理尺寸与 class1 偏移。

## 1. 约束的物理含义

采用现有 `P_camera=R P_template+t`。从图案正面观察，模板 +X 向右、+Y 向下，因此右手系 **+Z 指向板背面／地面**，不是朝天空。相机 +Z 是向前的光轴；向下观察时，两者大致同向。

候选使用三项检查：

1. **单面可见**：相机必须位于图案正面一侧。模板系相机中心为 `C=-Rᵀt`，要求 `C_z<0`，等价于 `R.col(2)·t>0`。
2. **总倾角**：`n=R.col(2)`，要求 `acos(n_z)≤30°`，即 `R(2,2)≥cos(30°)`。这是合成倾角，不能分别把 pitch、roll 限制为 30°后就认为满足；面内旋转不受限制。
3. **正深度与输入有效性**：所有实际使用的模型点都必须位于相机前方；验证有限值、旋转矩阵和模板 `Z=0` 模型。仅检查 `t_z>0` 不足以代替逐点正深度。

这要求靶板近似水平，并且相机光轴与靶板法向的总偏差在 30°内。地面坡度、机体倾斜和安装偏角共同占用这个余量。本次不需要 IMU 输入。

## 2. 接入位置与行为

配置：[tracker_config.json](../config/tracker_config.json)。

```json
"mono_pose_prior": {
  "enabled": true,
  "max_tilt_deg": 30.0
}
```

省略该节或设为 `enabled=false` 保留原有求解路径。角度必须为 `(0,90)` 内有限数值；非法类型与越界值在配置读取／求解器构造时拒绝。主配置除新增这一节外，其余字段与修改前逐项相同，用户当前 debug 模式、输入路径等均保留。

主要实现：

- [PlanarPosePrior.hpp](../include/pose/PlanarPosePrior.hpp)：配置解析与统一物理判据。
- [MonoPnPSolver.cpp](../src/pose/MonoPnPSolver.cpp)：先验在候选入池和最终输出之前验收；保留正常的重投影与 seed 选择。对超出倾角锥、但仍为正面且正深度的候选，先在锥内重新优化，再进行验收。
- [BoundedMonoRefinement.hpp](../include/pose/BoundedMonoRefinement.hpp)：约束重优化接口；实现位于上述求解器源文件。
- [MonoTracker.cpp](../src/tracker/MonoTracker.cpp)：将配置传入单目解算器，覆盖普通 BC、TT、DualRoi、class1 及冷解恢复。
- [StereoTracker.cpp](../src/tracker/StereoTracker.cpp)：其已有 MonoPnP 单侧退化求解器接收相同配置。

开启后，四点无 seed 的首帧／冷解也进入 ITERATIVE 与 IPPE 候选比较，并采用已有相对重投影容差。全部候选被拒时返回失败，不经旧版平坦 8 px 回退绕开先验。EPnP 初值可继续精化，但其未经先验验收的姿态不能直接成为输出。

边界重优化把旋转表示为“法向倾斜二维向量＋面内转角”，限制倾斜向量模长 ≤30°，同时优化三个平移分量。使用像素重投影残差、有限差分雅可比和带阻尼的投影 LM 更新，最多 25 次迭代；每次候选更新均投影回倾角锥。输出仍须通过原有重投影、单面与全部使用点正深度验收。这是在物理可行域内拟合 R/t，不是事后裁剪显示角度。

约束覆盖的是 **MonoPnPSolver**，不宣称约束了双目 GPNP、双目 InitialPnP 或仅视差模式。后两者的原流程保留。新增配置不改变既有 R/t、模型编号和坐标轴定义；不通过事后翻转 Z 列或旋转蓝色箭头修补结果。日志用 `[MonoPnP][Prior]` 记录候选拒绝原因与倾角。

## 3. 能解决什么，仍有哪些限制

单面可见约束只能排除相机落到背面一侧的解。**两个平面 PnP 候选可能都能看到正面**，仅检查法向符号不能完全消除二义性。新增 30°法向锥进一步缩小候选范围，但两个小倾角候选仍可能同时通过。

本次不会强行选择“更接近零倾角”的候选，因为用户确认的是倾角上限，不是真实倾角为零。满足物理约束后继续比较重投影与时序 seed；因此错误姿态持续存在、先验内跳变和角点错误仍可能出现。

IPPE 作者说明了小目标／近仿射情况下可能存在两个均可解释观测的姿态，并指出需要额外信息消除歧义；时间平滑也不保证选中真实轨迹。[IPPE 原作者说明](https://github.com/tobycollins/IPPE#resolving-the-flip-ambiguity)。OpenCV 的 `solvePnPGeneric` 可返回 IPPE 多候选。[OpenCV PnP 文档](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)。

另外，三维 Z 轴朝地面的物理方向确定，不代表图像中的蓝线始终指向固定像素方向。蓝轴是 `π(t+100n)-π(t)`；相机位置和投影长度变化仍会改变二维方向，尤其在投影很短时。本次不把二维箭头方向变化直接判成三维翻面。

## 4. 验证设计

专用测试：[test_mono_pose_prior.cpp](../tests/unit/test_mono_pose_prior.cpp)。7 组全部通过，覆盖总倾角边界与面内自由旋转、背面、全部使用点正深度、非法配置／模型、四点与十点冷热启动、超范围候选无法通过退化回退逃逸，以及边界附近重新拟合而非误换镜像分支。原有 `test_pose_solvers` 的 16 组测试也全部通过，包含既有单目棘轮冻结回归和 GPNP 边界用例。

构建／测试均通过项目 Docker 工具链执行。检查时发现工程原先未启用 CTest 注册；本次先实际执行两个测试二进制，再补齐测试构建分支中的 `enable_testing()` 和新增目标依赖。最终 `ctest -R pose` 实际发现并通过两个测试目标，内部共 23 组用例。

最终版完整流程基于上一轮冻结图像及配置，运行 6 组先验 ON（1020 帧），加参考组和近正视组的 OFF 重放（300 帧）。新增实验脚本：[compare_mono_pose_prior.py](../scripts/compare_mono_pose_prior.py)。

上一轮数据包含 50°～75°观测，明显超出本次假设，不能用整体输出率降低来声称精度提高。因此同时作两层评估：

- 保留原混合轨迹，单列真值倾角 ≤30°的帧；这些帧的缓存仍可能受到前面超范围输入影响，所以只作重入场景诊断。
- 从大倾角干净、快速路径和退化图像中截取 **10 段连续且全程 ≤30°**的片段，共 155 帧；每段在 OFF/ON 两种情况下都从空缓存启动，共处理 310 帧，隔离超范围缓存影响。

片段只复制原图，不修改图像或角点；manifest 保留原始帧号。最终版复用片段前逐文件验证图像哈希。包括原始序列的以下半开区间：

| 原序列 | 连续片段 `[start,stop)` |
|---|---|
| 大倾角干净 | `[0,14)`、`[54,80)`、`[126,150)` |
| 快速路径 | `[0,5)`、`[27,33)`、`[106,114)`、`[135,143)` |
| 退化图像 | `[0,14)`、`[54,80)`、`[126,150)` |

## 5. 实测结果与适用边界

最终版共完成 **1630 帧完整处理**：原序列与兼容性重放 1320 帧、连续片段对照 310 帧。下面的片段组均从空缓存开始运行，不夹杂超出先验的输入。

| 数据组 | 输出位姿 OFF→ON | 法向误差 P95 OFF→ON | 法向误差 >20°帧数 OFF→ON | 异常 Z 轴跳变 OFF→ON |
|---|---:|---:|---:|---:|
| 参考幅度，全程 ≤30°，120 帧 | 119→119 | 15.95°→15.12° | 4→3 | 1→1 |
| 小目标近正视，全程 ≤30°，180 帧 | 178→178 | 31.20°→28.76° | 36→35 | 9→7 |
| 大倾角干净的合规连续片段，64 帧 | 64→64 | 29.92°→29.92° | 9→9 | 1→1 |
| 快速路径的合规连续片段，27 帧 | 27→27 | 6.36°→5.88° | 1→0 | 1→0 |
| 退化图像的合规连续片段，64 帧 | 61→60 | 12.12°→19.70° | 2→3 | 4→2 |

合计 455 个满足先验的输入，输出 449→448，>20°错误帧 52→50，异常跳变 16→11。收益主要体现在输出方向符合物理假设及部分大错误被限制，**不是所有场景的角度精度都提高**。

快速路径片段的最大法向误差从 **33.30°降为 7.13°**；退化图像片段的最大误差从 **97.58°降为 32.21°**，但其 P95 和 >20°错误帧变差，并多拒绝了一帧。这一代价必须与减少大幅错误同时看待。小目标近正视的最大误差为 37.01°→30.38°，仍有多个先验内错误解。

![先验开关对比](../output/mono_pose_prior_bounded_20261016/figures/prior_comparison.png)

### 5.1 为什么最终采用约束重优化

初版只作硬筛选，结果单独保留在 [初版实验目录](../output/mono_pose_prior_20261016)。它在近正视组获得较低尾部误差，但在接近 30°的退化图像片段中，会删掉受噪声影响而略超界的好解，然后选中错误镜像分支：P95 从原版 12.12°升到 **53.16°**，>20°错误帧从 2 增至 17。

约束重优化将这项退化降到 P95 **19.70°**、>20°错误帧 **3**。它允许在边界内继续拟合图像，不把略超边界的初始化简单丢弃，也没有靠放宽最终 30°门限实现收益。但在该组上仍未全面优于原版，因此本次不能宣称二义性已根治。

初版与最终版各自完成 1630 帧处理，配置、代码哈希、原始日志与逐帧结果分开保存。初版数据只用于诊断边界问题，不与最终结果混写。

### 5.2 物理一致性与兼容性

最终版的全部输出均按脚本检查单面方向和 30°上限；单位测试另验证所有模型点的正深度。原始 >30°样本只作先验外压力诊断，不能将拒绝数量解释为真实场景的精度提升，也不能保证每个先验外图像都会被识别为不符合场景：错误点集仍可能拟合出一个表面合法的姿态。

参考组和近正视组关闭先验重放时，成功状态、策略、门控与历史基线一致，R/t 元素的最大差为 **0**。测试证明省略／关闭配置保留了既有行为。

结论：30°物理约束已生效并覆盖各单目 PnP 路径，跳变有所减少；在这个较宽的倾角范围内，两个错误／正确候选仍可能同时合法，无法仅凭单面可见与上限唯一确定法向。后续若需要进一步消除先验内错误，需要更准确的相机倾斜信息，或改善二维对应与多帧判别；继续增大平滑强度不能保证方向正确。

## 6. 复现与产物

编译和实际执行测试：

```bash
export PATH=/usr/bin:/bin:$PATH
scripts/docker-toolchain.sh cmake --build build-codex-bc-bench --target GPNP test_mono_pose_prior test_pose_solvers -j 4
scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/test_mono_pose_prior
scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/test_pose_solvers
scripts/docker-toolchain.sh ctest --test-dir build-codex-bc-bench -R pose --output-on-failure
```

新增 CMake 目标首次生成后，如多目标 make 尚未识别它，再单独构建该目标即可。本次主程序和两个测试目标均已实际编译成功。

```powershell
$py = 'C:/Develop/Conda/envs/DCL/python.exe'
& $py scripts/compare_mono_pose_prior.py prepare
& $py scripts/compare_mono_pose_prior.py prepare-segments --reuse-segments
```

```bash
scripts/docker-toolchain.sh bash output/mono_pose_prior_bounded_20261016/run_all.sh
scripts/docker-toolchain.sh bash output/mono_pose_prior_bounded_20261016/run_segments.sh
```

```powershell
& $py scripts/compare_mono_pose_prior.py analyze
& $py scripts/compare_mono_pose_prior.py analyze-segments
```

首次生成片段不传 `--reuse-segments`；复用既有片段时该开关会验证图像哈希。不要对已有实验直接重复 prepare，以免改写来源记录。所有运行配置均为副本。

产物：[最终实验目录](../output/mono_pose_prior_bounded_20261016)、[全序列比较](../output/mono_pose_prior_bounded_20261016/analysis/comparison.json)、[纯先验内连续片段比较](../output/mono_pose_prior_bounded_20261016/analysis/segment_comparison.json)、[关闭先验兼容性重放](../output/mono_pose_prior_bounded_20261016/analysis/disabled_replays.json)、[新增回归日志](../output/mono_pose_prior_bounded_20261016/unit_tests.log)、[原有回归日志](../output/mono_pose_prior_bounded_20261016/pose_regression.log)、[CTest 日志](../output/mono_pose_prior_bounded_20261016/ctest.log)、[最终验证清单](../output/mono_pose_prior_bounded_20261016/verification.json)。
