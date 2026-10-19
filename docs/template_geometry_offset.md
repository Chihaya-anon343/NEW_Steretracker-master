# class 1 中心偏移与统一 PnP 原点

日期：2026-10-15。基于 HEAD `2f6c3df` 的工作区修改。

## 配置与约定

在 [tracker_config.json](C:/D/Codes/NEW_Steretracker-master_COPY/config/tracker_config.json) 顶层设置：

```json
"template_geometry": {
  "enabled": true,
  "class1_center_in_class0_mm": [0.0, 0.0]
}
```

主配置已启用，`[0,0]` 是待填写的初始值，未据此认定实物同心。用户应提供 class1 中心在 class0 坐标轴下的 `[x,y]`，单位 mm；按正向模板观察，向右为 +X、向下为 +Y。输入是固定物理偏移，不能使用运行时两个 YOLO 框的像素中心差。两图案须共面、同轴。

中心定义为每组完整规范模板角点的 XY 包围框中点。BC 的顶点均值与此定义可能不同，不能用可见角点、筛选后的子集或检测框临时重算中心。若实物图纸采用其他中心定义，须先换算到此中心约定。

旧配置省略该节或显式 `enabled=false` 时保留原模型坐标。启用却缺少偏移、数组不是两维、非数值或非有限值时拒绝。修改配置后重新创建跟踪器；没有运行时热更新接口。

## 实现

保留现有 BC 比例、TT 边长与 Dual 板尺寸。各组规范三维点初始化时，先减自身包围框中心，再给 class1 点加上偏移：

$$
P_{0,i}^{(0)}=P_i-C_0,\qquad
P_{0,i}^{(1)}=P_i-C_1+[\Delta x,\Delta y,0]^T.
$$

只平移 XY，不改变 Z、轴方向、点序、编号或物理尺度。双目标注仍用于定义轮廓与尺寸；人工偏移覆盖原标注中两组中心的相对位置，不把标注已有位移再叠加一次。

转换发生在求解之前，覆盖普通 BC／TT、独立 class1、Dual 整体与联合模型、class1 BC／TT 退化链。单目兼容路径也使用同一模型放置函数。TT 保留现有左目 PnP 求解方式，没有改为双目 GPnP。

统一模式下位姿满足 `P_camera = R * P_class0 + t`，`t` 是 class0 中心在左相机系的位置。初值、位姿缓存、重投影和坐标轴使用同一个模型；结果标记为 `model_frame=class0`，`is_class1` 仍表示使用了哪类目标。仅视差继续标记 `unresolved`，不产生位姿。

若从已有 class1 局部位姿换算，等价公式是 `t0=t1-R*Delta`。代码不对最终平移再加补偿，避免符号错误或重复转换。原有运动门控、来源与重投影保护保留。

主要入口：

- [TemplateGeometry.hpp](C:/D/Codes/NEW_Steretracker-master_COPY/include/common/TemplateGeometry.hpp)：配置检查、读取及共享模型放置。
- BC／TT 初始化和 `setUseClass1()`：从原始模板重建，不随反复切换累加偏移。
- StereoTracker／MonoTracker 的 `prepareDualBcTemplate()`：分别放置两组标注模型，联合与独立路径使用同一原点。
- TrackerBase：将统一配置传给提取器，并设置位姿模型坐标标记。

## Docker 验证

主程序与以下五组专项／回归目标均在项目 Docker 工具链下重新编译、运行：

| 目标 | 通过／总计 |
|---|---:|
| test_dual_roi_flow | 23／23 |
| test_disparity_chain | 11／11 |
| test_pose_solvers | 16／16 |
| test_config | 8／8 |
| test_extractors | 9／9 |
| 合计 | 67／67 |

这是相关专项总数，不是整个项目测试数量。新增六项注册测试覆盖配置检查、旧配置兼容、模板反复切换、非零偏移、倾斜位姿、初值切换、独立 class1 BC／TT、Dual 回退、图像输出及仅视差不变性。现有模板目录实际存在，提取器回归未因目录缺失而跳过。

物理一致的合成投影使用偏移 `[32,-19] mm`，真实平移 `[90,-45,1300] mm` 和非零倾角；依次验证 class0、class1、联合、TT 及恢复 class0。六次解的最大平移误差约 `0.000754 mm`。其中四点 TT 使用左目 PnP，其他测试使用 StereoGpnp；测试分别覆盖冷启动与上一策略初值。该数值来自理想合成投影，不代表实拍精度。

图像测试把中心图案移动到 `[5,-3] mm`，保留原标注中心位置，验证人工偏移只施加一次：

| 帧 | 策略 | tX / tY / tZ，mm | 输出系 |
|---|---|---|---|
| 21 | DualRoi_BCJ | 1.04388 / 0.60072 / 890.859 | class0 |
| 22 | DualRoi_C1BC | 0.664853 / 0.219704 / 898.995 | class0 |
| 23 | DualRoi_BCJ | 1.04388 / 0.60072 / 890.859 | class0 |

恢复帧回到相同结果；联合与 class1 回退仍有约 8.15 mm 差异。该差异说明图像提取噪声、不同点集及尺度误差仍会影响结果，不能以统一原点宣称零跳变。图中 class1 接管后坐标轴仍表示 class0 原点，重投影使用放置后的模型。

![联合解坐标轴](C:/D/Codes/NEW_Steretracker-master_COPY/output/template_geometry_optimization/visuals/stereo_f21.png)

![class1 接管后的坐标轴](C:/D/Codes/NEW_Steretracker-master_COPY/output/template_geometry_optimization/visuals/stereo_f22.png)

另对同一图像对比禁用配置与 `[130,-85] mm`：仅视差的点数、右坐标、视差和模型编号完全相同，PnP 时间为零。主程序以 `[12,-7] mm` 配置完成三帧冒烟，左 YOLO=3、右 YOLO=0、PnP=0；非法三维数组在启动时以返回码 1 拒绝。未进行新一轮性能基准或真实双目位姿实验。

完整日志、初始文件备份、CSV、配置与图像见 [验证目录](C:/D/Codes/NEW_Steretracker-master_COPY/output/template_geometry_optimization)。构建中还补齐测试公共源码遗漏的 RKNN 存根依赖；未修改检测算法或 IDE 设置。

## 使用边界

实际偏移尚待填写。两个平移分量只能修正原点关系；物理尺寸不一致、坐标轴旋转、非共面、角点错配、平面姿态多解及标定误差仍需分别处理。修改后需要以实测靶板验证尺度、原点及策略切换，不能由上述理想投影替代。
