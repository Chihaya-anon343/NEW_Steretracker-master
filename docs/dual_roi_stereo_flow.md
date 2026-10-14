# Dual-ROI 双目光流路径

2026-10-13 更新：`disparity_full_chain=true` 时，BC 视差匹配失败后先执行左 class1 TT 光流，再按原开关决定是否补右 YOLO、执行双侧 TT。四点全部通过才接受，终止类错误不进入退化。详见 [完整仅视差策略链](disparity_full_chain.md)。

主配置和 BC 光流示例已开启 `stereo_frontend.dual_roi_flow=true`。省略时默认 false，保留旧 Dual-ROI 前端。普通 BC 的光流配置和质量阈值继续使用原设置。

```json
"stereo_frontend": {
  "method": "left_bc_lk",
  "dual_roi_flow": true,
  "dual_roi_legacy_fallback": true,
  "disparity_only": true,
  "roi_pyramid": true,
  "coarse_epipolar_band": true,
  "max_epipolar_error_px": 3.0,
  "on_failure": "reject"
}
```

## 数据流与模型

正常 Dual-ROI 帧只检测左图一次，获得 primary(class0)、secondary(class1)；右图仅参与匹配。class0 左 BC 提取后恢复全图坐标，使用板模板/人工标注的 3D 模型初始化及解算，右点由极线带 NCC＋正反向 LK 获得。

class0 位姿 P0 成功后，左 class1 BC 点与 P0 投影的 class1 标注按门限、一对一匹配建立模型对应。P0 的右相机投影仅作光流初值，右观测仍来自图像匹配。class0、class1 的有效点合并交给 GPnP，联合结果经过左右重投影与运动门控；class1 匹配或联合解算失败时保留已通过检查的 class0 位姿。

联合解算的两组点共用 `prepareDualBcTemplate()` 生成的板坐标系和尺寸，不使用普通 BC 的 `pixel_to_meter_scale_class0` 替代板模型。class0 原始角点数须与板模板数一致；检查发生在光流过滤之前。过滤后每个点保留区域、原左角点编号与模板编号，不按保留点数量重新配模板前缀。

class0 失败或 primary 几乎覆盖整个画面时，按原开关进入 class1 BC 退化，使用原 class1 局部模板尺寸。该位姿属于 class1 模板系，记录 `model_frame=class1`，并沿用现有缓存保护机制。`tier1_only=true` 在新光流路径中只执行 class0；关闭 `class1_fallback_enabled` 禁止 class1 单独接管帧。

## 单帧金字塔复用

`StereoFlowFrame` 为一对图像持有共同搜索裁剪及左右金字塔。class0 与 class1 顺序匹配时只构建一次左右金字塔；ROI 失败按原开关尝试全图，全图金字塔也可在该帧复用。primary 短路时仅准备 secondary 区域。一般外参或大区域仍采用全图。

该对象只服务当前输入对，不跨帧缓存视差或图像，避免使用上一帧的陈旧数据。日志 `pyramids` 为本帧新建的左右金字塔对数；全图回退或旧前端复核可能增加次数。

## 退化与右图检测

`dual_roi_legacy_fallback=true` 允许匹配失败后按需补右 YOLO。左侧原始检测结果直接复用，避免再检测左图。关闭该开关可测试纯光流路径。

位姿模式下可以恢复旧双侧前端及 BC→TT 退化。旧前端返回的双目位姿必须通过独立图像匹配、原右观测的一致性和左右重投影检查；被拒结果不更新缓存、不提交成功日志。若右图只检测到 class1，使用它实际的 class1 框执行退化，不伪造 class0 检测。

TT 的原位姿求解使用左图，标记为 `LeftPnpFallback`，不统计为双目位姿成功；仅在 `on_failure=left_pnp` 时接受。允许旧前端回退与允许左目位姿回退是两个不同开关。

接入过程中修正了旧 class1 退化链把 `PipelineResult.success` 当作提取成功的问题：提取器尚未求位姿时此字段为 false，应依据提取点判断是否进入解算。同时增加 class1 BC 原始点数与模型数量检查，防止四个矩形角点错误对应十角点模型的前四个点。

## 仅视差

`disparity_only=true` 支持 Dual-ROI，class0/class1 都可提取、匹配；不调用 InitialPnP、GPnP、运动位姿门控，也不写位姿缓存。class1 不借助 P0 建立标注 3D 身份，因此模板编号为 -1。帧是否通过看 `disparity_success`，位姿 `success/gpnp_success/stereo_success` 保持 false。

BC 匹配失败且允许旧前端退化时，可按需获取右 class1 ROI，调用原左右 TT 提取并校验匹配，输出 `DualRoi_C1TT` 视差；该分支同样完全不调用 PnP。关掉 `dual_roi_legacy_fallback` 可保证纯光流实验不调用右 YOLO。

`disparity_fN.csv` 保留原前六列：`left_index,left_x,left_y,right_x,right_y,disparity_px`，追加 `region,original_index,template_index,model_frame`。region 为 0/1；left_index 索引当前结果的原始左点数组，original_index 是所属区域内原编号。视差统一为 `uL-uR`，不取绝对值；失败帧的部分点不代表帧通过。

## 接口、日志与可视化

原五参数 `process()` 和路径接口保留。新增 `prefersLeftStereoFlow()` 包括普通 BC 和 Dual-ROI；旧 `prefersLeftBcFlow()` 的含义不变。位姿模式显式传入左右 ROI 的旧调用仍走旧前端；仅视差模式启用 Dual-ROI 光流时使用新路径，已有右 ROI 可直接供 TT 退化，不再检测。新增六参数重载接受按需双侧 ROI 提供函数，主程序负责右 YOLO 的延迟调用。

每个输入帧仅计一次帧号、写一条日志，包含所有尝试的耗时。日志追加 `c0/c1` 点数、金字塔构建次数、旧前端回退标记及路线。主程序将延迟右 YOLO 耗时计入 YOLO 阶段，避免重复计入跟踪阶段。

保留 `stereo_fN.png`、`dual_roi_left/right/binary/reproj/correspondence/bc_process_fN.png`，新增两组 `dual_roi_c0/c1_flow_status_fN.png`。class0 为黄色，class1 为绿色，点标签带区域与原编号。右图框表示搜索区域；位姿失败及仅视差模式不画位姿投影。仅 normal 输出主图，debug 输出完整面板。

## 验证与复现

```bash
bash scripts/docker-toolchain.sh cmake -S . -B build-codex-bc-bench -DCMAKE_BUILD_TYPE=Release -DGPNP_BUILD_TESTS=ON
bash scripts/docker-toolchain.sh cmake --build build-codex-bc-bench --target GPNP test_stereo_flow test_dual_roi_flow benchmark_dual_roi_frontend -j4
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/test_stereo_flow
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/test_dual_roi_flow
```

主配置保留此前的 `dual_trigger_area=2000000`，现有 `data/stereo_crop` 大部分图仍走普通 BC。Dual-ROI 专用实验使用独立配置降低触发阈值，不改变主配置或原数据。具体实验结果见 `docs/dual_roi_flow_validation.md`。
