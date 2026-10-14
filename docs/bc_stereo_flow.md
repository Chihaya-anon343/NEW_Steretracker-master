# 普通 BC 双目光流路径

2026-10-13 更新：主配置已开启完整仅视差链 `disparity_full_chain=true`，支持直接 TT、普通 BC→TT 和独立 class1；Dual 支持左 TT 优先退化。完整行为见 [完整仅视差策略链](disparity_full_chain.md)。下文关于仅普通 BC 的入口限制适用于该新开关关闭的兼容模式。

## 启用

运行 `./GPNP config/tracker_config_bc_flow.json`。缺少 `stereo_frontend` 节时保留原双目流程。主配置与光流示例的 Dual-ROI 阈值已按用户要求提高到 200 万像素²，供当前图像走普通 BC。示例输出目录为 `output/bc_flow`，图像输入与其他参数继承项目现有配置，运行前可按实际数据调整。

新模式只应用于普通 class0 BC 帧。左图执行一次 YOLO，按现有面积平滑、滞回和策略保持规则判定是否进入 BC；进入时只提取左图 BC，右图使用灰度图匹配。TinyTarget、Dual-ROI、class1 和单目模式保留原流程；需要双目检测的帧仍检测右图。手动 ROI 模式无需 YOLO。

## 算法

1. 左图 BC 提取与模板对应，恢复全图坐标，保留原角点编号。
2. 开启 InitialPnP 时求一次左目初始位姿，通过正深度与重投影检查后，投影出右图光流初值。
3. 初值不足时在极线容差带内进行有界 NCC 搜索；匹配失败时可进行一次粗搜索重试。左右金字塔各构建一次，复用于正反向 LK。
4. 检查边界、LK 状态、正反向误差、极线误差、NCC、视差、右点重复、匹配数量、有效比例与空间覆盖。
5. 通过检查的左右点及原角点编号交给现有 GPnP。结果通过正深度、旋转有效性与左右重投影检查后，才标记为双目成功。

匹配使用全图坐标，避免左右 ROI 原点不同造成虚假视差。平行双目使用 `d=uL-uR`，默认相机约定为 `t_rl=(+baseline,0,0)`，有效视差为正。一般外参使用标定极线约束；水平视差阈值仅用于平行双目。输入应与项目相机模型一致，当前改动未新增畸变补偿。

若多对可靠匹配呈负视差，返回 `negative_disparity_check_input_order`，提示检查左右输入顺序。不得直接对视差取绝对值。2026-10-12 更新的 `data/stereo_crop` 包含 60 对正视差图，manifest 记录 `uL-uR=d>0`，无需交换左右来源；其额外垂直平移约 1.774 px 超过代码默认 1.5 px 极线阈值，当前主配置按该合成实验设置为 3 px。

## 配置与失败处理

人工生成的左右图可以使用仅视差开关：

```json
"stereo_frontend": {
  "method": "left_bc_lk",
  "disparity_only": true,
  "on_failure": "reject"
}
```

主配置 `tracker_config.json` 和光流示例已开启此开关。设置 `disparity_only=false` 可恢复位姿解算；省略时默认为 false。仅视差模式完全跳过 InitialPnP、GPnP 和左目位姿回退，光流采用同坐标初值与极线粗搜索，无需物理尺度预测。匹配成功通过 `disparity_success` 表示，`success`、`gpnp_success` 与 `stereo_success` 保持 false，不更新位姿缓存、不提交融合相机观测。非普通 BC 帧跳过处理并说明原因；单目配置不支持该开关。

输出目录新增 `disparity_fN.csv`，列为原左角点编号、左右图全图坐标、带符号的 `disparity_px=uL-uR`。保留匹配检查和诊断图，主图显示左右匹配，不生成三维坐标轴或位姿投影。日志中的 PnP 状态显示 SKIP，位姿解算耗时为零。

`stereo_frontend.method` 为 `left_bc_lk` 或 `dual_bc`。`on_failure` 默认 `reject`：无合格双目位姿时返回失败，避免向融合模块提交错误测量。选择 `left_pnp` 允许经过检查的左目位姿回退，但不计为双目成功；BC 本身提取失败时，允许继续现有左目 TinyTarget 退化链。

主要默认阈值：LK 窗口 21、金字塔层数 3；FB 误差 1 px、极线误差 1.5 px；NCC 窗口 11、最低相关系数 0.8；平行双目视差 0.5–400 px；至少 4 对、有效比例 0.6、覆盖比例 0.25；最终平均重投影误差不超过 3 px，单点不超过 8 px。完整字段见示例配置，实际相机与目标尺寸改变后应重新验证阈值。

`coarse_epipolar_band` 默认为 true，搜索极线法向的 `±max_epipolar_error_px`，3 px 容差对应七条采样线。左侧 NCC 窗口只归一化一次，每个沿线位置保留法向最佳候选，唯一性检查按沿极线方向的距离区分相关峰；最终仍执行原有 FB、极线误差、NCC 与帧级质量检查。设置 false 可复现原单极线粗搜索，`coarse_retry=false` 则关闭两种粗搜索。

水平平行双目、沿线与法向均为 1 px 采样时，将整个容差带一次采样为灰度条带，使用批量 NCC 计算，并用积分图保留原弱纹理检查；其他几何或采样间距仍使用逐点 NCC。返回候选坐标依然按原采样位置解释，亚像素精化由 LK 完成。

`max_search_candidates` 仍限制沿线初始采样位置数，不代表整个容差带的 NCC 计算总数。法向采样最多九条；沿线步长大于 1 px 时，对最多四个独立峰进行有限局部精查，每峰最多 17 个沿线位置。因此单点单次搜索最多 `(max_search_candidates + 68) × 9` 次 NCC 计算，ROI→全图重试可能再执行一次。容差超过 4 px 时法向采样间距会增大，不能保证捕获所有窄相关峰。此改动未引入帧间视差缓存。

## 接口与可视化

原 `process(left,right,visualize,left_group,right_group)` 接口保留。配置开启且普通 BC 帧传入 `right_group=nullptr` 时使用光流；传入左右 ROI 的旧调用仍走原流程。主程序自动选择，输入、相机、IMU、雷达与融合接口保持兼容。

结果和日志增加 `stereo_method`、`pose_source`、`stereo_success`、`failure_reason`、`stereo_match_ms` 与粗搜索次数。`success` 表示可提交的位姿；双目是否成功应看 `stereo_success`。程序统计左右 YOLO 推理次数，供检查是否省去右图检测。

保留 `stereo_fN.png` 及原 BC 调试图文件名，新增 `binary_corner_flow_status_fN.png`。匹配图按原角点编号着色：有效观测彩色、拒绝点红叉、初值灰色、位姿投影白色。右侧显示实际灰度匹配结果，不伪造右图 BC 二值图。失败帧同样保存主图与原因。

## Docker 验证

Windows 主机通过 `scripts/docker-toolchain.sh` 转发命令到现有项目容器。默认容器 `st_copy_build`、挂载 `/tmp/NEW_Steretracker-master_COPY`；脚本先核对挂载确实指向当前项目。可用 `STEREOTRACKER_DOCKER_CONTAINER` 与 `STEREOTRACKER_DOCKER_WORKDIR` 指定其他已配置容器。

```bash
bash scripts/docker-toolchain.sh cmake -S . -B build-codex-bc-flow -DCMAKE_BUILD_TYPE=Debug -DGPNP_BUILD_TESTS=ON
bash scripts/docker-toolchain.sh cmake --build build-codex-bc-flow --target GPNP test_stereo_flow -j2
bash scripts/docker-toolchain.sh ./build-codex-bc-flow/tests/test_stereo_flow
bash scripts/docker-toolchain.sh ./build-codex-bc-flow/GPNP config/tracker_config_bc_flow.json
```

测试覆盖大视差、冷启动、遮挡、无效输入、极线拒绝、模板编号、普通 BC 集成、失败处理、左目回退、旧接口与可视化。合成数据验证逻辑和几何一致性；实际性能收益与位姿精度仍需真实标定双目序列比较。建议同序列分别运行 `dual_bc` 和 `left_bc_lk`，比较成功率、左右推理次数、总耗时、匹配耗时及位姿误差。

## 前端性能基准

`stereo_frontend.roi_pyramid=true` 开启方案 1，主配置与示例已开启；省略该字段时默认 false，保留全图金字塔。可配置 `roi_fallback_full`（默认 true）、`roi_max_area_ratio`（默认 0.85）、`roi_extra_margin_px`（默认 16）。裁剪区域由左目标、全部允许视差、有效初值和 LK/NCC 边界共同推导；左右共用原点并调整局部相机主点，返回值仍为全图坐标。

当前仅对默认水平平行双目采用 ROI；一般旋转外参保留全图流程。区域占图像比例过大时直接全图，ROI 质量检查失败时按开关重试全图。重试帧计时包含两次尝试，可能比原全图更慢。日志新增搜索区域尺寸和 `full_retry`，可视化标题标识 ROI。

`benchmark_stereo_frontend` 使用当前 debug 输入图像和 BC 参数比较两条前端路径，无需临时切换主程序配置。它保留原路径的左右 YOLO 并行，关闭两条路径的 PnP、可视化和图像写盘；预热后交替测量，记录 YOLO、灰度/裁剪/特征匹配与总耗时。基准不注册到自动单元测试中。

```bash
bash scripts/docker-toolchain.sh cmake -S . -B build-codex-bc-bench -DCMAKE_BUILD_TYPE=Release -DGPNP_BUILD_TESTS=ON
bash scripts/docker-toolchain.sh cmake --build build-codex-bc-bench --target benchmark_stereo_frontend -j2
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/benchmark_stereo_frontend config/tracker_config.json 100 output/bc_flow_benchmark/run
```

输出 `samples.csv` 和 `report.md`。仅适用于普通 class0 BC 输入，报告中的成功率为原路径提取足够的配对点、光流路径通过匹配检查的比例，不表示位姿成功率或两个算法已达到同样的精度。

对比方案 1 时，在上述基准命令末尾增加 `roi`，两侧均使用左 YOLO+BC，只改变全图/ROI 金字塔。独立的 `benchmark_stereo_roi` 以已知视差的合成场景检查精度和回退代价：

```bash
bash scripts/docker-toolchain.sh cmake --build build-codex-bc-bench --target benchmark_stereo_roi -j2
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/benchmark_stereo_roi 50 output/bc_roi_experiment/synthetic
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/benchmark_stereo_frontend config/tracker_config.json 100 output/bc_roi_experiment/front_run roi
```
