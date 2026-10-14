# 当前双目处理流程与分支图

依据 2026-10-13 工作区代码整理。当前实现不包含 AKAZE；特征策略为 BinaryCorner（BC）和 TinyTarget（TT），Dual-ROI 使用 class0/class1 两组 BC 及退化路径。

2026-10-13 后续更新：主配置已启用完整仅视差链。TT、独立 class1 和 BC→TT 不再因策略类型被跳过；Dual 退化先尝试左 TT 光流。新增分支及当前默认流程见 [完整仅视差策略链](disparity_full_chain.md)，下列兼容分支图适用于 `disparity_full_chain=false` 或位姿模式。

## 1. 整体入口

```mermaid
flowchart TD
    A[读取配置、相机标定、模板和模型] --> B[获取同步左右图<br/>normal 批量输入 / debug 固定图像]
    B --> C{光流前端开启?}
    C -->|是| D[仅左图 YOLO<br/>生成 class0 / class1 ROI]
    C -->|否| E[左右 YOLO<br/>原双侧前端]
    D --> F{左侧策略支持光流?}
    F -->|普通 class0 BC| G[普通 BC 光流路径]
    F -->|Dual-ROI 且 dual_roi_flow 开启| H[Dual-ROI 光流路径]
    F -->|TT / 独立 class1 / 未开启的 Dual| I{仅视差模式?}
    I -->|是| S[跳过此帧<br/>记录原因并使缓存老化]
    I -->|否| J[复用左检测，补右 YOLO]
    J --> E2{双侧 ROI 有效?}
    E --> E2
    E2 -->|是| K[原双侧策略<br/>普通 BC→TT / Dual-ROI / class1 BC→TT]
    E2 -->|否| M{有可用单侧 ROI<br/>且允许单目降级?}
    M -->|是| N[单目处理并标记来源]
    M -->|否| S
    G --> O[匹配或位姿结果]
    H --> O
    K --> O
    N --> O
    O --> P{仅视差模式?}
    P -->|是| Q[输出匹配、视差 CSV、日志和可视化<br/>不调用 PnP，不提交视觉位姿，不更新位姿缓存]
    P -->|否| R{位姿被接受?}
    R -->|是| T[更新位姿缓存<br/>输出位姿、来源、日志和可视化]
    R -->|否| U[输出失败与诊断<br/>不更新成功位姿缓存]
    T --> V[normal 模式按融合开关提交有效视觉观测<br/>与 IMU / 雷达进入 ESKF]
    U --> W[融合开启时仍可惯性传播<br/>失败视觉观测不作有效更新]
    Q --> X[下一帧]
    V --> X
    W --> X
    S --> X
    X --> B
```

debug 的手动 ROI 可绕过 YOLO。无有效目标时直接跳过。旧 ROI 生成器可能把单侧检测失败归并为空组，单目降级还要求入口确实保留了可用单侧 ROI。

## 2. 普通 BC 光流与共用匹配器

```mermaid
flowchart TD
    A[左 ROI BC 单目提取<br/>二值化、连通域清理、模板角度匹配、角点排序] --> B[恢复全图坐标<br/>保留原角点与模板编号]
    B --> C{仅视差模式?}
    C -->|是| D[使用图像坐标初值<br/>不求左目初始位姿]
    C -->|否| E[左目 InitialPnP 候选与检查<br/>有效时投影至右图作为匹配初值]
    D --> F[准备搜索 ROI 与左右图像金字塔]
    E --> F
    F --> G[初值 LK；无可靠初值或失败点<br/>使用极线带 NCC 粗搜索后 LK 重试]
    G --> H[左→右与右→左 LK<br/>FB、边界、极线误差、NCC 与正视差检查]
    H --> I[右点去重<br/>有效点数、有效比例与空间覆盖检查]
    I --> J{ROI 匹配不通过<br/>且允许全图重试?}
    J -->|是| K[全图金字塔重试匹配]
    K --> L[保留有效点原编号<br/>计算 d = uL - uR]
    J -->|否| L
    L --> M{仅视差模式?}
    M -->|是| N[以 disparity_success 判定帧通过<br/>输出匹配点与视差]
    M -->|否且匹配通过| O[GPnP：模板 3D 点与左右观测<br/>正深度、左右重投影等检查]
    M -->|否且匹配失败| P{允许左目 PnP 回退?}
    O -->|通过且来源为 StereoGpnp| Q[接受双目位姿]
    O -->|不通过| P
    P -->|是| R[接受经过检查的左目初始位姿<br/>提取失败时可尝试左侧 TT]
    R --> T{左目位姿通过?}
    T -->|是| U[标记 LeftPnpFallback]
    T -->|否| S[此帧失败]
    P -->|否| S
```

匹配器不靠右图 BC 角点编号配对，而是直接在右图寻找左角点对应观测。极线带允许一定法向偏差，当前容差为 3 px；它不代替相机标定或双目校正。视差保留符号，可靠负视差会提示核对左右输入顺序，不通过取绝对值掩盖问题。

普通 BC 使用自身模板物理尺度。GPnP 使用模板 3D 点与左右图观测；仅视差模式不会把人工图像的视差换成具有物理意义的位姿。

## 3. Dual-ROI 光流分支

```mermaid
flowchart TD
    A[左图一次 YOLO<br/>class0 primary + class1 secondary] --> B{primary 几乎覆盖画面<br/>且允许 class1 回退、非 tier1_only?}
    B -->|是| C[跳过 class0，直接 class1 BC 光流]
    B -->|否| D[左 class0 BC 提取<br/>匹配右观测，保留角点身份]
    D --> E{仅视差模式?}
    E -->|是| F[非 tier1_only 时再做左 class1 BC 光流<br/>两区域共用单帧金字塔]
    F --> G[汇总通过质量检查的区域<br/>class1 单独接管须允许回退]
    E -->|否| H[class0 板模型 GPnP 与独立检查]
    H --> I{class0 位姿通过?}
    I -->|否且允许回退| C
    I -->|是| J{启用增强<br/>且存在 class1 标注?}
    J -->|否| K[保留 class0 位姿]
    J -->|是| L[用 class0 位姿关联左 class1 点与板标注<br/>投影右初值，再执行光流匹配]
    L --> M[合并两组有效观测<br/>同一板坐标系联合 GPnP]
    M -->|通过重投影与运动检查| N[接受联合位姿]
    M -->|增强失败| K
    C --> O{仅视差模式?}
    O -->|是| G
    O -->|否| P[class1 局部模板 GPnP 与检查<br/>标记 model_frame=class1]
    G --> Q{视差匹配通过?}
    Q -->|是| R[输出两区域或有效单区域视差]
    Q -->|否| S{允许旧前端退化?}
    P -->|位姿通过| T[接受 class1 位姿]
    P -->|位姿失败| U[按 on_failure 尝试受控左目回退<br/>仍失败才进入旧前端退化]
    U --> S
    I -->|否且不允许回退| S
    S -->|否| V[此帧失败]
    S -->|是| W[按需补右 YOLO<br/>复用原左图检测结果]
    W --> X{仅视差模式?}
    X -->|是| Y[满足退化开关时用真实左右 class1 ROI<br/>TT 提取与匹配检查，不进行 PnP]
    Y -->|匹配通过| R
    Y -->|失败| V
    X -->|否| Z[恢复原双侧 Dual / class1 BC→TT 链]
    Z --> AA[独立复核位姿来源、左右观测与重投影<br/>左目结果还须显式允许 left_pnp]
    AA -->|通过| AB[接受并标记旧前端退化]
    AA -->|失败| V
```

`tier1_only=true` 只保留 class0，不执行 class1 增强、primary 短路和 class1 退化。`dual_roi_legacy_fallback` 与 `on_failure=left_pnp` 是两个独立权限：允许补右检测，并不等于允许接受左目位姿。

class0 与增强 class1 采用同一板模型坐标系；单独 class1 回退采用 class1 局部模板坐标系。仅视差时没有 class0 初始位姿，不能借投影建立 class1 的板标注身份，因此相关模板编号可为 -1。TT 原位姿路径使用左目 PnP，不应统计为双目 GPnP 成功。

## 4. 当前配置与实际执行路线

| 项目 | 当前值及含义 |
|---|---|
| 输入 | normal，`data/stereo_crop` 目录批量处理，`max_frames=0` |
| 前端 | `method=left_bc_lk`，Dual-ROI 光流开启 |
| 输出模式 | `disparity_only=true`，只匹配与输出视差 |
| 普通策略提名 | class0 ROI 面积 ≤900 px² 为 TT，否则 BC；实际切换还受 EMA、迟滞及连续确认影响 |
| Dual 触发 | class0 ROI 面积严格 **大于** 2,000,000 px²，且存在 class1 |
| 独立 class1 | class0 缺失时，class1 面积至少 100 px² 才允许生成 ROI；BC/TT 阈值为 400 px² |
| 旧前端退化 | 开启；Dual 匹配失败时可能按需调用右 YOLO |
| 左目位姿回退 | `on_failure=reject`，不允许 |
| 入口单目降级 | `stereo_mono_fallback=false` |

当前数据大部分走：**批量左右图 → 左 YOLO → 普通 BC → 极线带 NCC / LK → 质量过滤 → 视差 CSV、日志和可视化**。完整视差链还支持 TT 和独立 class1；关闭 `disparity_full_chain` 时，原非支持策略仍按兼容行为跳过。

匹配成功看 `disparity_success`；位姿成功看 `success/gpnp_success`；是否为真正双目位姿还要看来源及 `stereo_success`。失败帧保留的部分视差点不代表整帧通过。

normal 输出主可视化，debug 输出中间面板。Dual 保留区域、原角点和模板编号；右图 ROI 框表示搜索范围。仅视差及位姿失败时不画位姿轴或重投影。每帧只计一次帧号、写一条跟踪日志，包含回退尝试耗时。

代码依据：`main.cpp`、`src/detection/RoiGenerator.cpp`、`src/tracker/TrackerBase.cpp`、`src/tracker/StereoTracker.cpp`、`src/tracker/StereoTrackerFlow.cpp`、`src/tracker/StereoTrackerDualFlow.cpp`、`src/matching/StereoFlowMatcher.cpp`。
