# Steretracker 完整功能测试方案

> 版本: 1.0 | 日期: 2026-08-03 | 对应项目版本: c7b71ea

---

## 目录

1. [测试目标与范围](#1-测试目标与范围)
2. [测试策略与分层](#2-测试策略与分层)
3. [测试环境与依赖](#3-测试环境与依赖)
4. [T1 配置与工厂函数](#4-t1-配置与工厂函数)
5. [T2 五状态分级（RoiGenerator）](#5-t2-五状态分级roigenerator)
6. [T3 特征提取器](#6-t3-特征提取器)
7. [T4 位姿解算](#7-t4-位姿解算)
8. [T5 退化链（策略降级）](#8-t5-退化链策略降级)
9. [T6 退化全景（13类退化路径）](#9-t6-退化全景13类退化路径)
10. [T7 输入系统](#10-t7-输入系统)
11. [T8 端到端集成测试](#11-t8-端到端集成测试)
12. [T9 性能基线](#12-t9-性能基线)
13. [附录：合成测试图像规格](#13-附录合成测试图像规格)

---

## 1. 测试目标与范围

### 1.1 测试目标

验证 Steretracker 系统的以下核心能力：

| 目标 | 描述 |
|------|------|
| **正确性** | 五状态分级判定、特征提取、位姿解算结果符合预期数学模型 |
| **鲁棒性** | 退化链逐级降级、边界条件处理、非法输入拒绝 |
| **完整性** | 单目/双目、normal/debug、三种输入源所有模式组合可用 |
| **性能** | 单帧处理耗时在可接受范围内 |

### 1.2 不在范围内

- ONNX Runtime / YOLO 模型推理本身的正确性（属于模型验证范畴）
- GPU / CUDA 加速性能
- IMU / 高度计融合（Phase 2 占位，尚未实现）
- 实时摄像头源（Phase 3 未开始）
- 跨平台硬件兼容性

---

## 2. 测试策略与分层

```
                     ┌──────────────────────────┐
                     │     T8 端到端集成测试      │  ← 最高层
                     │  (主程序完整流水线)         │
                     └──────────┬───────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
  ┌────────▼────────┐  ┌───────▼───────┐  ┌─────────▼────────┐
  │  T5/T6 退化链测试 │  │ T7 输入系统测试│  │  T3 特征提取器测试 │
  │  (策略降级路径)    │  │ (图像源抽象层) │  │  (BC/TT/AKAZE)    │
  └────────┬────────┘  └───────┬───────┘  └─────────┬────────┘
           │                    │                    │
  ┌────────▼────────┐  ┌───────▼───────┐  ┌─────────▼────────┐
  │ T2 五状态分级测试 │  │ T1 配置工厂测试 │  │ T4 位姿解算测试   │
  │ (面积阈值决策树)  │  │ (参数校验)      │  │ (PnP求解器)       │
  └─────────────────┘  └───────────────┘  └──────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │   T9 性能基线测试    │  ← 基础层
                     │   (单帧耗时基准)     │
                     └─────────────────────┘
```

**分层原则**:
- 低层（T1/T2/T4）不依赖任何外部模型或文件 I/O
- 中层（T3/T5/T6/T7）依赖 OpenCV 图像数据，但不依赖 ONNX
- 高层（T8）依赖完整构建产物和 ONNX Runtime

---

## 3. 测试环境与依赖

### 3.1 单元测试 (T1-T7, T9)

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| C++ | 17 | 编译 |
| OpenCV | 4.x (`core, imgproc, features2d, calib3d, highgui`) | 图像处理、PnP |
| Eigen | 3.x | 线性代数 |
| CMake | 3.22+ | 构建 |

> **不依赖 ONNX Runtime**，可在 Windows / Linux / macOS 上编译运行。

### 3.2 集成测试 (T8)

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| ONNX Runtime | 1.20+ | YOLO 推理 |
| `best.onnx` | 项目提供的模型 | 目标检测 |
| `config/tracker_config.json` | 当前版本 | 运行参数 |

> **仅支持 Linux**（主项目 CMakeLists.txt 硬编码 Linux 路径）。

### 3.3 测试辅助工具

| 工具 | 用途 |
|------|------|
| Python 3.8+ + OpenCV | 合成测试图像生成 |
| `test_utils/test_assert.hpp` | 自定义断言宏（数值容差比较） |
| `test_utils/test_timer.hpp` | 计时器 |
| `test_utils/image_compare.hpp` | 图像相似度比较 |

---

## 4. T1 配置与工厂函数

### 4.1 测试目标

验证 `include/common/Config.hpp` 中的工厂函数：
- `makeTrackerConfig()` — 跟踪器配置构造
- `makeYoloConfig()` — YOLO 配置构造
- `makeStereoCameraParams()` — 相机参数构造

### 4.2 用例矩阵

| 用例ID | 函数 | 输入 | 预期 | 优先级 |
|--------|------|------|------|--------|
| T1-01 | `makeStereoCameraParams` | 合法 K, R=I, t=[-120,0,0] | 返回有效参数, `baseline=120`, `R_rl` 近似 I | P0 |
| T1-02 | `makeStereoCameraParams` | K 中 fx=0 | 抛出异常（内参非法） | P0 |
| T1-03 | `makeStereoCameraParams` | R 行列式 ≠ 1.0 | 抛出异常 | P0 |
| T1-04 | `makeTrackerConfig` | 所有合法值 | scale=0.5, gpnp_min_pts=4, 面积阈值正确 | P0 |
| T1-05 | `makeTrackerConfig` | `scale=0`（非法） | 抛出异常 | P0 |
| T1-06 | `makeTrackerConfig` | `scale=1.2`（>1 非法） | 抛出异常 | P0 |
| T1-07 | `makeTrackerConfig` | `gpnp_min_pts=2`（<3） | 抛出异常 | P0 |
| T1-08 | `makeTrackerConfig` | `tiny_max_area=0`（非法） | 抛出异常 | P0 |
| T1-09 | `makeYoloConfig` | 合法 model_path + conf=0.5 | 返回有效配置 | P0 |
| T1-10 | `makeYoloConfig` | `model_path=""`（空） | 抛出异常 | P0 |
| T1-11 | `makeYoloConfig` | `conf_threshold=0`（非法） | 抛出异常 | P0 |
| T1-12 | `makeYoloConfig` | `conf_threshold=1.2`（>1 非法） | 抛出异常 | P0 |

### 4.3 实现方式

纯 C++ 头文件级测试，构造结构体→检查成员值或异常：

```cpp
// 伪代码示意
TEST(T1, ValidTrackerConfig) {
    auto cfg = makeTrackerConfig(0.5, 4, true, 800, 40001, 490000, ...);
    ASSERT_EQ(cfg.akaze_scale, 0.5);
    ASSERT_EQ(cfg.gpnp_min_pts, 4);
    ASSERT_EQ(cfg.tiny_max_area, 800);
}
TEST(T1, InvalidScale) {
    ASSERT_THROW(makeTrackerConfig(0.0, ...), std::invalid_argument);
}
```

---

## 5. T2 五状态分级（RoiGenerator）

### 5.1 测试目标

验证 `RoiGenerator::generateGroup()` 的五状态判定决策树：

```
面积 ≤ tiny_max_area(400)              → State 1 远 (TinyTarget)
401 ≤ 面积 ≤ 40000                     → State 2 中 (BinaryCorner)
40001 ≤ 面积 < dual_trigger(490000)    → State 3 中近 (AKAZE)
面积 ≥ 490000 + 有 class1              → State 4 近 (Dual-ROI)
面积 ≥ 490000 + 无 class1              → State 3 中近 (AKAZE)
无 class0 + 有 class1 + ≥ class1_min   → State 5 极近 (Class1回退)
无 class0 + 无 class1                  → SKIP
```

### 5.2 用例矩阵

| 用例ID | 场景 | 输入 (class0面积, class1面积) | 预期状态 | 优先级 |
|--------|------|------------------------------|---------|--------|
| T2-01 | 远距离 | class0=200, class1=无 | State 1 (TinyTarget) | P0 |
| T2-02 | 远距离边界 | class0=400 (==tiny_max) | State 1 (TinyTarget) | P0 |
| T2-03 | 中距离下界 | class0=401 (>tiny_max) | State 2 (BinaryCorner) | P0 |
| T2-04 | 中距离典型 | class0=10000 | State 2 (BinaryCorner) | P0 |
| T2-05 | 中距离上界 | class0=40000 | State 2 (BinaryCorner) | P0 |
| T2-06 | 中近距离下界 | class0=40001 | State 3 (AKAZE) | P0 |
| T2-07 | 中近距离典型 | class0=200000 | State 3 (AKAZE) | P0 |
| T2-08 | 面积≥490000+无class1 | class0=500000, class1=无 | State 3 (AKAZE) | P0 |
| T2-09 | 面积≥490000+有class1 | class0=500000, class1=1000 | State 4 (Dual-ROI) | P0 |
| T2-10 | 仅class1回退 | class0=无, class1=5000 | State 5 (CloseRange) | P0 |
| T2-11 | class1面积不足 | class0=无, class1=50 (<class1_min) | SKIP | P0 |
| T2-12 | 完全无检测 | class0=无, class1=无 | SKIP | P0 |
| T2-13 | 边界 dual_trigger-1 | class0=489999, class1=有 | State 3 (AKAZE) | P1 |
| T2-14 | 边界 dual_trigger+1 | class0=490001, class1=有 | State 4 (Dual-ROI) | P1 |

### 5.3 实现方式

构造 `Detection` 向量 → 调用 `generateGroup()` → 验证 `RoiGroup.primary`/`secondary`/`is_dual` 标志和 ROI 尺寸。

```cpp
// 伪代码
std::vector<Detection> detections = {
    {0, 0.9f, cv::Rect2f(100, 100, sqrt(area), sqrt(area))}, // class0
    {1, 0.8f, cv::Rect2f(..., ..., ...)}                      // class1 (可选)
};
auto group = roi_gen.generateGroup(detections, img_size);
// 根据面积断言 group.is_dual, group.primary 等
```

---

## 6. T3 特征提取器

### 6.1 测试目标

验证三个特征提取器在合成图像上的正向提取正确性：
- **TinyTargetExtractor** — 4 角点矩形板
- **BinaryCornerExtractor** — 10 角点多边形板
- **AkazeGpnpExtractor** — AKAZE 纹理描述子（仅模板匹配部分，不含光流）

### 6.2 合成图像生成

见 [§13 附录：合成测试图像规格](#13-附录合成测试图像规格)。

### 6.3 用例矩阵

#### 6.3.1 TinyTargetExtractor

| 用例ID | 场景 | 合成图像 | 预期 | 优先级 |
|--------|------|---------|------|--------|
| T3-TT-01 | 正向提取—0° | 50×50 正方形板，无旋转 | 提取 4 角点，success=true，坐标在板四角 ±3px | P0 |
| T3-TT-02 | 正向提取—45° | 同上，旋转45° | 提取 4 角点，角度匹配 ±3° | P0 |
| T3-TT-03 | 正向提取—任意角度 | 随机旋转 0°~360° | 提取 4 角点，象限对齐正确 | P0 |
| T3-TT-04 | 低对比度 | 目标灰度值接近背景（差=30/255） | success=false 或角点数不足 4 | P1 |
| T3-TT-05 | 部分遮挡 | 右下 25% 区域覆盖为背景色 | success=false 或评分显著降低 | P1 |
| T3-TT-06 | 极小目标 | 板上 10×10 px（<50×50 标准化） | 触发面积过滤，success=false | P2 |
| T3-TT-07 | `extractMono` 与 `extract` 一致性 | 单目与双目（相同左图+右图） | 左图角点坐标一致 | P1 |

#### 6.3.2 BinaryCornerExtractor

| 用例ID | 场景 | 合成图像 | 预期 | 优先级 |
|--------|------|---------|------|--------|
| T3-BC-01 | 正向提取—0° | 100×100 10角点板，无旋转 | 提取 ≈10 个角点，success=true | P0 |
| T3-BC-02 | 正向提取—45° | 同上，旋转45° | 角点坐标旋转回正后与 0° 模板匹配 | P0 |
| T3-BC-03 | 旋转15°/30°/60° | 对应旋转 | 模板匹配选中正确角度模板 | P1 |
| T3-BC-04 | 最大连通域筛选 | 图像含噪声小区域 | 仅提取主目标区域 | P1 |
| T3-BC-05 | approxPolyDP 不精确命中 | 10角点变形 | 取 best_diff 最近似结果 | P1 |
| T3-BC-06 | 连通域仅1个 | 空白/单一区域 | 保留原二值图不崩溃 | P2 |
| T3-BC-07 | 模板匹配 IoU | 与模板完全一致的图像 | best_overlap ≈ 1.0 | P2 |

#### 6.3.3 AkazeGpnpExtractor

| 用例ID | 场景 | 合成图像 | 预期 | 优先级 |
|--------|------|---------|------|--------|
| T3-AK-01 | AKAZE 检测+描述子 | 纹理板 200×150mm | `n_kp_left ≥ min_pts` (≥4) | P0 |
| T3-AK-02 | 模板匹配三阶段 | 纹理板 + 模板 | 获得 good_matches，n_template_match ≥ 4 | P0 |
| T3-AK-03 | Ratio Test 过滤 | 弱纹理区域 | 低匹配数自动提前返回 | P1 |
| T3-AK-04 | Cross-Check 对称验证 | 左右纹理有歧义 | 双向匹配过滤假匹配 | P1 |
| T3-AK-05 | Homography RANSAC | 少量错匹配 | 内点筛选通过，H 非空 | P1 |
| T3-AK-06 | Homography 为空的回退 | 纯噪声图 | H=null → 回退到 Stage 2 结果 | P1 |
| T3-AK-07 | scale=0.5 降采样 | 大纹理板 | 坐标还原到原始 ROI 尺寸 ×2 | P2 |
| T3-AK-08 | 描述子 NORM_HAMMING | AKAZE 二值描述子 | 描述子为 61 字节 CV_8U | P2 |

### 6.4 实现方式

1. 用 Python+OpenCV 预生成合成图像（PNG，已知几何与 GT 角点）
2. C++ 单元测试加载图像，调用提取器
3. GT 坐标 (JSON) 伴随图像生成
4. 比较提取角点 ↔ GT 角点（配准后计算 RMS 误差）

---

## 7. T4 位姿解算

### 7.1 测试目标

验证三个 PnP 求解器在合成 2D-3D 对应点上的位姿反算正确性。

### 7.2 用例矩阵

#### 7.2.1 MonoPnPSolver

| 用例ID | 场景 | 输入 | 预期 | 优先级 |
|--------|------|------|------|--------|
| T4-MP-01 | 正向 EPnP | 合成 4 对 2D-3D + K | `t.z > 0`, `10<|t|<20000`, R 行列式≈1 | P0 |
| T4-MP-02 | 点数不足 | 仅 3 对 | `success=false` | P0 |
| T4-MP-03 | 2D 与 3D 数量不等 | 4对2D + 5对3D | `success=false`（或抛出） | P0 |
| T4-MP-04 | 深度为负（t.z<0） | 反转 Z 轴 | 位姿校验失败 `success=false` | P0 |
| T4-MP-05 | 深度过大 | t 的模 >20000mm | 位姿校验失败 | P0 |
| T4-MP-06 | 深度过小 | t 的模 <10mm | 位姿校验失败 | P0 |
| T4-MP-07 | 含 NaN 点 | 2D 坐标含 NaN | `success=false` | P1 |
| T4-MP-08 | RANSAC 后 ITERATIVE 精化 | 合成含噪声 | 精化后重投影误差 ≤ RANSAC 结果 | P2 |

#### 7.2.2 InitialPnPSolver

| 用例ID | 场景 | 输入 | 预期 | 优先级 |
|--------|------|------|------|--------|
| T4-IP-01 | 正向 RANSAC+ITERATIVE | 合成 8 对 2D-3D + K | 输出有效位姿 | P0 |
| T4-IP-02 | RANSAC 失败 | 仅 3 对（inliers<4） | `success=false` | P0 |
| T4-IP-03 | RANSAC 含离群值 | 10 对中 3 对错误 | 内点正确识别 | P1 |

#### 7.2.3 GPnPSolver

| 用例ID | 场景 | 输入 | 预期 | 优先级 |
|--------|------|------|------|--------|
| T4-GP-01 | 正向 LM 优化 | 合成 6 对 2D+视差+3D，warm-start | 最终 cost < 初始 cost | P0 |
| T4-GP-02 | 无 warm-start（首帧） | 同上，无缓存 | 仍输出有效位姿 | P0 |
| T4-GP-03 | warm-start 帧间一致性 | 连续 2 帧几乎相同 | 第 2 帧迭代次数 显著减少 | P1 |
| T4-GP-04 | TinyTarget solvePnP | 4 对 2D-3D (ITERATIVE) | 不经过 RANSAC，直接 ITERATIVE | P1 |
| T4-GP-05 | GPNPMonitor 信息 | 正常优化 | `initial_cost`, `final_cost`, `iterations` 合理 | P2 |

### 7.3 实现方式

1. 定义 GT 位姿 `(R_gt, t_gt)`
2. 定义已知 3D 模型点
3. 用 `K * (R_gt * P3d + t_gt)` 计算合成 2D 投影点
4. 调用求解器反算位姿
5. 比较 `||R_computed - R_gt||` 和 `||t_computed - t_gt||`（容差由重投影误差换算）

---

## 8. T5 退化链（策略降级）

### 8.1 测试目标

验证策略主退化链 `AKAZE → BinaryCorner → TinyTarget → 终止` 的逐级降级行为。

### 8.2 用例矩阵

| 用例ID | 场景 | 模拟条件 | 预期降级路径 | 优先级 |
|--------|------|---------|-------------|--------|
| T5-01 | AKAZE 正常 | State 3 有效特征 | 走 AKAZE，不触发退化 | P0 |
| T5-02 | AKAZE → BC | AKAZE 返回 success=false | 降级到 BC | P0 |
| T5-03 | AKAZE → BC → TT | AKAZE+BC 均失败 | 降级到 TT | P0 |
| T5-04 | AKAZE → BC → TT → 终止 | 三者均失败 | 输出空位姿 | P0 |
| T5-05 | BC → TT | State 2 BC 失败 | 降级到 TT | P1 |
| T5-06 | TT → 终止 | TT 失败 | 输出空位姿 | P1 |
| T5-07 | Dual-ROI 不退化 | Dual-ROI BC 或 AK 失败 | 整体失败，不降级到 BC/TT | P1 |
| T5-08 | YOLO 无检测 ≠ 退化 | RoiGroup 无效 | skip 帧，不触发策略链 | P1 |
| T5-09 | n_kp_left < 3 触发退化 | AKAZE 提取 2 个点 | success=false → 退化 | P2 |
| T5-10 | 退化不逆 | TT 失败 | 不回到 BC/AKAZE | P2 |

### 8.3 实现方式

**困难**: 退化链由 `StereoTracker::process()` / `MonoTracker::process()` 内部控制，难以从外部注入失败。

**方案**: 
- **Mock 提取器** — 创建继承 `FeatureExtractor` 的 mock 类，`extract()` 返回可控的 `PipelineResult`
- **或 子类化 Tracker** — 暴露 `fallback_extractors_` 供测试遍历
- **或 合成极端图像** — 对真实提取器输入使之失败（纯黑图 → 无 AKAZE 关键点；极小噪声图 → TT 失败），间接验证

推荐方案：创建 mock 提取器进行白盒测试验证退化遍历逻辑。

---

## 9. T6 退化全景（13类退化路径）

### 9.1 测试目标

覆盖 CLAUDE.md §9.12 中列出的全部 13 类退化机制，确保：
- 退化方向单向不可逆
- Dual-ROI 完全隔离
- 退化图是有向无环图 (DAG)

### 9.2 用例矩阵

| 用例ID | 退化类别 | 测试内容 | 优先级 |
|--------|---------|---------|--------|
| T6-01 | 策略主退化链 | 同 T5，此处置 [已通过 T5] | — |
| T6-02 | Dual-ROI 独立路径 | BC 失败→整体失败；AK 失败→整体失败；合并点数<4→失败 | P1 |
| T6-03 | State 5 class1 回退 | class0 丢失+class1 有效→按面积重分类→走标准策略链 | P1 |
| T6-04 | YOLO 层 skip（非退化） | 确认 YOLO 无检测不触发 fallback_extractors_ | P1 |
| T6-05 | AKAZE 模板匹配子退化 | RatioTest<4→失败; CrossCheck<4→失败; Homography 空→回退 Stage2 | P1 |
| T6-06 | BC 内部子退化 | 连通域≤1→保留原图；旋转回退→INTER_NEAREST；approxPolyDP→best_diff | P2 |
| T6-07 | TT 内部子退化 | 连通域<200→跳过；评分无合格→success=false | P2 |
| T6-08 | 光流追踪点级退化 | FB error≥1px→丢弃点；全部丢弃→num_valid=0 | P2 |
| T6-09 | MAD 视差滤波点级退化 | |disp-median|>3σ→剔除；大量剔除→degraded=true | P2 |
| T6-10 | 立体投影点级退化 | disparity≤0 或投影出边界→valid_mask=false | P2 |
| T6-11 | PnP 求解器退化 | RANSAC 失败/精化失败/校验失败 各类情况 | P2 |
| T6-12 | ROI 输入层退化 | RoiGroup 无效→全图回退 AKAZE 链 | P2 |
| T6-13 | Dual-ROI BC 模板退化 | 首次调用 prepareDualBcTemplate → 一次性生成 | P2 |

### 9.3 实现方式

以 **文档审查 + 手动验证** 为主，辅以 mock 方法。T6 的 P2 项在初次迭代中可通过代码审查确认逻辑正确性，后续迭代中逐步实现自动化。

---

## 10. T7 输入系统

### 10.1 测试目标

验证三种图像源 (`FileStereoSource` / `DirectoryStereoSource` / `SequenceSource`) 的加载与 `InputProvider` 统一协调。

### 10.2 用例矩阵

| 用例ID | 图像源类型 | 场景 | 预期 | 优先级 |
|--------|----------|------|------|--------|
| T7-01 | File | 合法双目文件对 | `SensorPacket.valid=true`, left/right 非空, 尺寸一致 | P1 |
| T7-02 | File | 左图路径不存在 | `valid=false` 或异常 | P1 |
| T7-03 | Directory | 双目编号序列 10 帧 | `getNextPacket()` 返回 10 帧，按序号递增 | P1 |
| T7-04 | Directory | 空目录 | 首帧 `valid=false` | P1 |
| T7-05 | Sequence | 单目序列 5 帧 | `getNextPacket()` 返回 5 帧，`right_image` 为空 | P1 |
| T7-06 | InputProvider | 正常协调 | 封装图像源的 `next()` 调用正确传递 | P1 |
| T7-07 | SensorPacket | timestamp 字段 | 递增或非零 | P2 |

### 10.3 实现方式

准备临时测试图像目录 → 调用 InputProvider → 遍历所有帧 → 验证 size/valid/timestamp。

```cpp
// 伪代码
InputSystemConfig cfg;
cfg.image.type = "file";
cfg.image.left_path = "test_data/file_left.png";
cfg.image.right_path = "test_data/file_right.png";
InputProvider provider(cfg);
auto packet = provider.getNextPacket();
ASSERT_TRUE(packet.valid);
ASSERT_FALSE(packet.left_image.empty());
```

---

## 11. T8 端到端集成测试

### 11.1 测试目标

验证完整流水线在真实图像上的端到端行为。

### 11.2 测试矩阵

| 用例ID | 模式 | 配置 | 输入 | 预期 | 优先级 |
|--------|------|------|------|------|--------|
| T8-01 | Debug 单目 | `mode=debug, mono_mode=true` | `data/大图/cj01_image_0032.jpg` | 输出位姿，终端行含 `[Frame N] Strategy n=...` | P0 |
| T8-02 | Debug 单目 手动ROI | T8-01 + 手动ROI | 同上 | 跳过 YOLO，使用手动 ROI | P0 |
| T8-03 | Debug 双目 | `mode=debug, mono_mode=false` | 大图左右对 | 输出包含立体约束信息 | P0 |
| T8-04 | Normal 双目 Directory | `mode=normal, source_type=directory` | `data/大图/` | InputProvider 迭代所有匹配帧 | P1 |
| T8-05 | Normal 单目 Sequence | `mode=normal, mono_mode=true, source_type=sequence` | 单目序列 | 仅左图，右图为空 | P1 |
| T8-06 | 配置文件缺失 | 不存在路径 | 报错退出 | 程序返回非零，错误信息非空 | P1 |
| T8-07 | 非法模式值 | `mode="invalid"` | — | 报错或默认 normal | P2 |

### 11.3 实现方式

Shell 脚本 (`tests/integration/run_e2e.sh`):
1. 确认 `build/Steretracker` 存在
2. 对每个配置 JSON 运行主程序
3. grep 终端输出验证关键字段
4. 检查 output/ 目录生成了预期文件

```bash
#!/bin/bash
echo "T8-01: Debug Mono..."
./build/Steretracker tests/integration/configs/t8_01_debug_mono.json 2>&1 | tee /tmp/t8_01.log
grep -q "Strategy" /tmp/t8_01.log && echo "PASS T8-01" || echo "FAIL T8-01"
```

---

## 12. T9 性能基线

### 12.1 测试目标

记录各策略单帧处理耗时基线，用于后续性能回归检测。

### 12.2 度量指标

| 指标 | 阶段 | 目标阈值（建议） | 优先级 |
|------|------|-----------------|--------|
| T9-01 | AKAZE 特征提取 + 匹配 | < 200 ms | P2 |
| T9-02 | BinaryCorner 特征提取 | < 100 ms | P2 |
| T9-03 | TinyTarget 特征提取 | < 50 ms | P2 |
| T9-04 | GPnP 优化 (6对) | < 5 ms | P2 |
| T9-05 | InitialPnP | < 10 ms | P2 |
| T9-06 | MonoPnP | < 5 ms | P2 |
| T9-07 | 完整帧 (最慢策略) | < 500 ms | P2 |

### 12.3 实现方式

`test_utils/test_timer.hpp` 提供 `AutoTimer` RAII 工具类，在关键函数入口/出口记录耗时。运行 50 次取 P50/P95。

---

## 13. 附录：合成测试图像规格

### 13.1 TinyTarget 测试图像

```
图像尺寸: 200×200 px (灰度)
目标: 50×50 px 正方形板
背景: 灰度 128, 目标: 灰度 200
3D 物理尺寸: 50×50 mm (Z=0)
GT 角点（全图坐标）: (75,75), (125,75), (125,125), (75,125)
```

生成脚本参数:
- `rotation`: [0, 15, 30, 45, 60, 75, 90, 随机]
- `contrast`: [high(差127), medium(差60), low(差30)]
- `occlusion`: [none, 25%右下]

### 13.2 BinaryCorner 测试图像

```
图像尺寸: 300×300 px (灰度)
目标: 10 角点多边形（基于 0_degrees.txt 的角点 × pixel_scale）
3D 物理尺寸: 等比例缩放至 100×100 mm (Z=0)
背景: 灰度 128, 目标: 灰度 220
```

### 13.3 AKAZE 测试图像

```
图像尺寸: 640×480 px (灰度)
目标: 纹理板（使用 OpenCV 随机噪声或真实纹理截图）
模板: 预先截取目标区域 200×150 px
3D 物理尺寸: 200×150 mm
```

### 13.4 生成脚本 (Python)

`tests/test_data/synthetic/generate.py` 使用 OpenCV Python 绑定量产上述三类图像，同时输出配套 JSON 格式的 GT 数据文件。

```python
# 伪代码
import cv2
import numpy as np
import json
import os

def generate_tinytarget(output_dir):
    """生成 TinyTarget 测试集"""
    for rotation in [0, 15, 30, 45, 60, 75, 90]:
        img = np.full((200, 200), 128, dtype=np.uint8)
        # 绘制 50×50 白色正方形
        rect = ((100, 100), (50, 50), rotation)
        box = cv2.boxPoints(rect)
        cv2.drawContours(img, [np.int0(box)], -1, 200, -1)
        # 保存图像 + GT JSON
        cv2.imwrite(f"{output_dir}/tt_{rotation:03d}.png", img)
        with open(f"{output_dir}/tt_{rotation:03d}.json", 'w') as f:
            json.dump({"corners": box.tolist(), "rotation": rotation}, f)

def generate_binarycorner(output_dir):
    """生成 10 角点 BinaryCorner 测试集"""
    # 读取 0_degrees.txt 角点坐标
    # 绘制填充多边形
    pass

def generate_akaze(output_dir):
    """生成 AKAZE 纹理测试集"""
    # 生成随机纹理或使用 Perlin 噪声
    pass
```

---

## 修订历史

| 版本 | 日期 | 修订内容 |
|------|------|---------|
| 1.0 | 2026-08-03 | 初版 — 覆盖 9 个测试域，共 85+ 个用例 |