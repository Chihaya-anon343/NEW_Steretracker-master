# CLAUDE.md — Steretracker 项目 AI 上下文

> 优化目标：AI 快速定位代码、理解流程、追踪数据流。不追求人类可读性。

## 项目概要

C++17 双目视觉定位系统。YOLO ONNX 检测 → ROI → 面积策略分发 → 特征提取 → 视差 → 位姿解算。
技术栈: OpenCV 4.x, Eigen 3.x, ONNX Runtime.
面向无人机视觉定位。

## 入口: main.cpp 整体流程

```
main.cpp:22-272  main()
  ├─ 26-127  读取 tracker_config.json (cv::FileStorage)
  │   ├─ camera: fx,fy,cx,cy,baseline_mm
  │   ├─ strategies/akaze_gpnp: template_path,real_w/h,scale,min_pts,use_init_pnp
  │   ├─ strategies/binary_corner: corners,kernel_size,scale,target_size,pixel_to_meter_scale,roi_pad_pixels,otsu_ratio
  │   ├─ strategies/tiny_target: target_size,scale_factor,square_size_m,roi_pad_pixels
  │   ├─ yolo: model_path,conf_threshold,target_class_id,roi_expand_ratio,roi_min_size
  │   ├─ strategies/akaze_min_area (40001), tiny_max_area (800)
  │   ├─ strategies/dual_roi: secondary_expand_pixels (10), akaze.scale (0.5)
  │   ├─ input: left/right image paths
  │   ├─ manual_roi: enabled, left{x,y,w,h}, right{x,y,w,h}
  │   └─ mono_mode: bool
  ├─ 132-140  构造输出目录 output/{img_name}
  ├─ 145-155  初始化相机参数 K,R_rl,t_rl, TrackerConfig
  ├─ 157-176  加载图像, 初始化 YoloRoiProvider + StereoTracker
  ├─ 177-187  若 mono_mode: 设置 MonoConfig
  │
  ├─ 192-217  MONO PATH (mono_mode==true):
  │   for frame 1..2:
  │     if use_manual_roi: left_group = RoiGroup{manual_rl, {}, false}  // is_dual=false
  │     elif yolo_ok: left_group = yolo.detect(left_img, right_img).first  // BUG: 应调用detectMono
  │     plg = left_group.valid() ? &left_group : nullptr
  │     result = tracker.processMono(left_img, visualize, plg)
  │
  └─ 218-261  STEREO PATH (默认):
      for frame 1..2:
        if use_manual_roi: lg=RoiGroup{l,{},false}, rg=RoiGroup{r,{},false}
        elif yolo_ok: tie(lg,rg) = yolo.detect(left, right)
        result = tracker.process(left, right, visualize, plg, prg)
```

## 核心类: StereoTracker

**文件**: `include/tracker/StereoTracker.hpp` | `src/tracker/StereoTracker.cpp`

### 构造与初始化

```
ctor(line ~30-90):
  ├─ 预创建 3 个提取器: akaze_extractor_, binary_extractor_, tiny_extractor_
  ├─ 预创建 dual_akaze_extractor_ (双ROI专用, 不同scale参数, lines 55-63)
  ├─ 加载 AKAZE 模板 → akaze_extractor_->setTemplateData()
  ├─ 加载 BC 模板 → binary_extractor_->loadTemplates()
  ├─ 加载 TT 模板 → tiny_extractor_->loadTemplates()
  ├─ 存储 akaze_min_area_, tiny_max_area_, dual_roi_secondary_expand_, dual_roi_akaze_scale_
  ├─ 初始化 mono_pnp_ (MonoPnPSolver)
  └─ 初始化 gpnp_solver_, initial_pnp_, stereo_projector_
```

### 关键方法

| 方法 | 文件:行 | 用途 |
|------|---------|------|
| `process(left,right,vis,lg,rg)` | cpp:~480 | 双目主入口 |
| `processMono(left,vis,lg)` | cpp:1779-1954 | 单目主入口 |
| `processDualRoi(left,right,lg,rg,vis)` | cpp:~1300 | 双目双ROI |
| `processDualRoiMono(left,lg,vis)` | cpp:1513-1771 | 单目双ROI |
| `configureStrategyChain(roi_area)` | cpp:94-122 | 按面积选策略链(仅指针切换) |
| `applyRoiPadding(roi,pad)` | cpp:~125 | ROI外扩 |
| `validateRoi(roi,img_size,tag)` | cpp:~1960 | ROI有效性校验+裁剪 |
| `prepareDualBcTemplate()` | cpp:954-1038 | 一次性从AKAZE模板提取BC角点 |
| `dispatchPnP(result,extractor)` | cpp:~200 | PnP路由(Akaze→runAkazePnP, BC→runBinaryCornerPnP, TT→runTinyTargetPnP) |
| `runAkazePnP(result)` | cpp:~230 | AKAZE PnP + MAD滤波器 |
| `runBinaryCornerPnP(result)` | cpp:~350 | BC PnP (InitialPnP→GPNP) |
| `runTinyTargetPnP(result)` | cpp:~430 | TT PnP (solvePnP ITERATIVE) |
| `finalizePose(result,pose)` | cpp:~470 | 填充R,t, success标志 |

### 策略链配置: configureStrategyChain() cpp:94-122

```
roi_area >= akaze_min_area_ || roi_area == 0:
  primary = akaze_extractor_
  fallback = [binary_extractor_, tiny_extractor_]
  → AKAZE → BC → TT

roi_area > tiny_max_area_:
  primary = binary_extractor_
  fallback = [tiny_extractor_]
  → BC → TT

else (roi_area <= tiny_max_area_):
  primary = tiny_extractor_
  fallback = []
  → TT (no degradation)
```

## 单目模式四种输入场景

### 场景1: 手动ROI (main.cpp:201-202)
```
left_group = RoiGroup{manual_rl, {}, false}
→ processMono() → is_dual? no → configureStrategyChain(area) → extractMono链 → EPnP
```

### 场景2: YOLO单ROI (main.cpp:203-206)
```
RoiGenerator::generateGroup() (RoiGenerator.cpp:73-96):
  1. generate(class=0) → 取最高置信度检测框 → detectionToRoi(expand+clamp)
  2. 若 primary.area > 490000 && class=1存在 → is_dual=true (→场景3)
  3. 否则 → RoiGroup{primary, {}, false}
→ processMono() → is_dual? no → 同场景1
```

### 场景3: YOLO双ROI (main.cpp:203-206, RoiGenerator.cpp:73-96)
```
条件: class0面积>490000 && class1检测存在
RoiGroup{primary(class0), secondary(class1), is_dual=true}
→ processMono() cpp:1796-1800:
    if(left_group && left_group->is_dual) {
        return processDualRoiMono(left_img, *left_group, visualize);  // EARLY RETURN
    }
→ 完全跳过策略链!
```

### 场景4: 无ROI/全图回退 (main.cpp:208)
```
plg = nullptr
→ processMono() cpp:1811-1815:
    roi = validateRoi(nullptr) → 无效 → RoiRect{0,0,cols,rows}
    roi_area = cols*rows (极大)
→ configureStrategyChain → AKAZE链 (area远超40001)
```

## processMono() 详细流程 cpp:1779-1954

```
1779  processMono(left_img, visualize, left_group)
1785    if !mono_cfg_.enabled → return empty
1790    if left_img.empty() → return empty
1796    if left_group && left_group->is_dual:
1798      result = processDualRoiMono(left_img, *left_group, visualize)
1799      state_.frame_count++
1800      return result                          // EARLY RETURN — 不走后续!
1803    loadImage(left_img) → gray, color
1811    left_roi = left_group ? &left_group->primary : nullptr
1812    roi = validateRoi(left_roi, left_img.size(), "mono_left")
1814    if !roi.valid(): roi = RoiRect{0, 0, left_img.cols, left_img.rows}
1823    裁剪: left_gray_roi, left_color_roi = gray/color(roi)
1825    left_offset = (roi.x, roi.y)
1828    configureStrategyChain(roi.width * roi.height)
1830    is_first = (state_.frame_count == 0)
1834-1880  退化链循环:
    for ext in [primary, fallback0, fallback1]:
1847      local = ext->extractMono(left_gray_roi, left_color_roi)
1850-1865 offset pts_left_match, pts_left_good, kp_left 回到全图坐标
1867      if local.success && local.n_kp_left >= 3:
            result = local; break
          else: log degradation, continue
1882    if all failed: log + return empty
1888-1895  构建3D点:
    for m in result.good_matches:
      if m.trainIdx in range(template_.pts_3d):
        matched_pts_3d.push_back(template_.pts_3d[m.trainIdx])
1898    pose = mono_pnp_.solve(result.pts_left_match, matched_pts_3d, camera_.K)
1901-1953 finalizePose + visualize + log
1952    state_.frame_count++
```

## processDualRoiMono() 详细流程 cpp:1513-1771

```
1517  prepareDualBcTemplate()  // 一次性
1520-1551  加载左图, expand secondary ROI
1543  sec_to_pri_offset = (sec.x-pri.x, sec.y-pri.y)
1555-1558  裁剪4个子图:
    left_c0_gray/color = left[primary], left_c1_gray/color = left[secondary]
1561  result_bc = binary_extractor_->extractMono(left_c0_gray, left_c0_color)
1566  result_ak = dual_akaze_extractor_->extractMono(left_c1_gray, left_c1_color)
1573-1582  AK所有点 += sec_to_pri_offset  // class-1-local → class-0-local
1584-1644  MERGE:
    1589-1611: 若|matched_angle|>0.5°, 用reorderByGeometry重排bc_pts3d副本
    1613-1620: BC贡献: merged += result_bc.pts_left_match[i] ↔ bc_pts3d[i]
    1622-1631: AK贡献: for each good_match, idx=trainIdx
              merged += result_ak.pts_left_match[i] ↔ ak_pts3d[idx]
    1638: if total_use < 4 → return失败
1646-1652  恢复全图坐标: merged_pts_2d += (primary.x, primary.y)
1655  pose = mono_pnp_.solve(merged_pts_2d, merged_pts_3d, camera_.K)
1657-1771  finalizePose + log + visualize(4面板)
```

## prepareDualBcTemplate() cpp:954-1038

```
958  tmpl_gray = akaze_extractor_->templateData().gray_image
969  Otsu二值化
972-983  findContours(RETR_EXTERNAL) → 取最大面积
986-1006  approxPolyDP二分搜索: lo=0.001, hi=0.05(×perimeter), 8次迭代
          目标n = binary_extractor_->lastCornersBeforeReorder().size() 或 10
1015-1020  reorderByGeometry(corners, center, 0.0°, -1.0) → dual_bc_tmpl_corners_
1022-1031  3D点: X=(x/tw)*real_w_mm, Y=(y/th)*real_h_mm, Z=0 → dual_bc_tmpl_pts3d_
```

## 四种策略方法的 extractMono() 算法

### 策略1: AKAZE (大图 ≥40001px²)

**文件**: `src/feature/AkazeGpnpExtractor.cpp:115-145`

```
extractMono(gray, color):
  120-127  akaze_extractor_.extract(gray):
    若scale<1.0: resize(INTER_LINEAR)降采样
    detectAndCompute(默认AKAZE, NORM_HAMMING) → 61字节二值描述子 N×61
    kp.pt *= (1.0/scale) 坐标还原
    → FeatureSet{keypoints(N), descriptors(N×61), points(N×2)}
  
  130-133  template_matcher_.match(desc_left, kp_left, template_desc, template_kp):
    Stage1 (TemplateMatcher.cpp:48-76):
      knnMatch(L→T, k=2, NORM_HAMMING)
      Lowe's ratio: dist[0] < 0.75 × dist[1]
      <4 matches → 提前返回
    Stage2 (cpp:78-112):
      knnMatch(T→L, k=2) + ratio test
      对称验证: T→L映射回L→T同一索引
      <4 → 提前返回
    Stage3 (cpp:114-148):
      findHomography(L_pts, T_pts, RANSAC, 5.0px)
      H为空 → 返回Stage2结果
      仅保留inlier
    → MatchResult{good_matches, pts_left_match, pts_template_match, num_matches}
  
  137-142  填充PipelineResult:
    good_matches, pts_left_match, pts_template_match
    success = (num_matches >= 4)
```

**与双目extract()的区别**: 跳过光流追踪(flow_tracker_.track)、立体投影(projector_->project)、MAD视差滤波、pts_right填充。

**模板预计算** `setTemplateData()` cpp:33-45:
```
AkazeExtractor(scale=1.0)  // 全分辨率
→ detectAndCompute → keypoints + descriptors
→ 3D: X=(kp.x/tw)*real_w_mm, Y=(kp.y/th)*real_h_mm, Z=0
→ template_.pts_3d[i] ↔ template_.keypoints[i]
```

**PnP 3D点查找**(processMono中 cpp:1893-1895):
```
matched_pts_3d.push_back(template_.pts_3d[m.trainIdx])
// trainIdx 索引到 AKAZE 模板的关键点列表
```

### 策略2: BinaryCorner (中图 801~40000px²)

**文件**: `src/feature/BinaryCornerExtractor.cpp:1024-1098 (extractMono), 285-470 (extractFromBinary)`

**模板**: 24个角度模板(0°,15°,...,345°) = `.txt`(角点坐标) + `.png`(二值mask `image>127`)

```
extractMono(gray, color):
  1036  Otsu: threshold(gray, binary, 0, 255, THRESH_BINARY|THRESH_OTSU)
  1041  extractFromBinary(binary, gray, corners):

    322  keepLargestRegion():
      connectedComponentsWithStats(8) → 取最大面积(排除背景)
      num_labels<=1 → 返回原图
    
    325  fillHoles():
      findContours(RETR_EXTERNAL) → drawContours(FILLED)
    
    328  smoothBoundary():
      MORPH_CLOSE → MORPH_OPEN, kernel=config_.kernel_size(默认3)
    
    331-348  模板匹配(IoU):
      resize二值图到target_size(100×100)
      findBestMatch():
        extractAndNormalizeRoi(binary, 50) → 裁剪包围盒→正方形→resize 50×50
        遍历24个模板: calculateOverlap(norm_binary, tmpl.image_bool)
        IoU = countNonZero(A&B) / countNonZero(A|B)
      取最高IoU角度 → last_matched_template_
    
    354-402  旋转回正:
      优选: warpAffine(INTER_CUBIC)旋转灰度图 -matched_angle
        → Otsu × otsu_ratio(1.3)
        → keepRegionFromCenter(): 从中心螺旋搜索白色像素→floodFill
        → fillHoles + smoothBoundary
      回退: warpAffine(INTER_NEAREST)旋转二值图 + 形态学
    
    404  extractLargestContour():
      findContours(RETR_EXTERNAL) → 取最大area
    
    411-417  extractCornersFromContour():
      可选scale up(config_.scale, 默认1.0)
      perimeter = arcLength(contour, closed=true)
      二分搜索epsilon: lo=0.001, hi=0.05(×perimeter), 8次迭代
        目标 n == config_.corners(默认10)
        n<target → hi=mid; n>target → lo=mid
      无精确命中 → 取best_diff最近似
      scale down回原始
    
    421-433  逆旋转:
      dx=x-cx_rot, dy=y-cy_rot
      x_out = dx*cos(θ) + dy*sin(θ) + cx_orig
      y_out = -dx*sin(θ) + dy*cos(θ) + cy_orig
    
    435-465  matchCorners + reorderByGeometry:
      模板角点缩放到smoothed尺寸
      matchCorners(tmpl_scaled, corners, angle+20°(CPSAGL)):
        reorderByGeometry(tmpl, center, ref_angle) → tmpl_order
        reorderByGeometry(bin, center, ref_angle) → bin_order
        1:1索引配对: ordered[tmpl_idx] = corners[bin_idx]
  
  1049-1053  corners → KeyPoint(kp_left)
  1056  pts_left_match = corners
  1059-1093  查找0°模板 → 角点×pixel_to_meter_scale×1000 → pts_3d
    good_matches = [DMatch(i,i,0) for i in range(n)]
    template_data_.pts_3d[i] = (tmpl_corners[i].x*s_mm, tmpl_corners[i].y*s_mm, 0)
  1097  success = !corners.empty()
```

**reorderByGeometry()** cpp:735-791:
```
零轴=正上方(-y), CCW=正
每个角点: a = atan2(-dx, -dy), 归一化[0,2π)
找最接近ref_rad的角点为起点
若ref_dist>0: 距离中心最近的优先
从起点CCW遍历输出索引[start, start+1, ..., n-1, 0, ..., start-1]
```

### 策略3: TinyTarget (小图 ≤800px²)

**文件**: `src/feature/TinyTargetExtractor.cpp:403-477 (extractMono), 163-244 (extract4Corners)`

**模板**: 同BC, 24个角度模板(0°~345°), target_size=50×50

```
extractMono(gray, color):
  extract4Corners(gray, corners, best_angle, best_overlap):

    173-181  Otsu + matchTemplate():
      threshold(gray, binary, 0, 255, THRESH_BINARY|THRESH_OTSU)
      matchTemplate():
        connectedComponentsWithStats(8) → 取最大连通域
        裁剪包围盒→扩展正方形→resize 50×50(INTER_NEAREST)
        遍历24个模板: calculateOverlap(IoU)
        → best_angle, best_overlap
    
    183-199  超分辨率放大(×4):
      sf = scale_factor(默认4)
      resize(INTER_CUBIC) ×sf
      GaussianBlur(3×3, σ=0.3)
      Otsu → MORPH_OPEN(3×3) → MORPH_CLOSE(5×5)
    
    201-217  连通域评分 selectBestComponent():
      最小面积过滤: area<200(×4空间)跳过
      4维评分:
        矩形度     = area/(w*h)                         ×0.25
        面积       = area_ratio∈[0.15,0.6]?1.0:clip    ×0.30
        中心距离   = 1 - dist/roi_radius               ×0.30
        长宽比     = 1 / (max(w,h)/min(w,h))            ×0.15
      总分 = Σ(score×weight), 取最高分组件
    
    220-224  minAreaRect(best_contour) → 4角点
      orderPoints(): TL=min(sum), BR=max(sum), 
                    余下: 大diff=TR, 小diff=BL
    
    227  亚像素精化:
      copyMakeBorder(6) → cornerSubPix(winSize=5×5, 
        zeroZone=-1×-1, TermCriteria(EPS+MAX_ITER,50,0.001))
    
    229-233  象限旋转对齐:
      quadrant = round((360-best_angle)/90.0) % 4
      std::rotate(corners, corners+quadrant, corners+4)
      0°→不转, 90°→转3, 180°→转2, 270°→转1
    
    236-241  缩放回原始: x_out = x/sf, y_out = y/sf

  填充PipelineResult:
    4个KeyPoint, 1:1 DMatch(i,i,0)
    3D点: half_mm = square_size_m*1000/2
      TL=(-h,-h,0), TR=(+h,-h,0), BR=(+h,+h,0), BL=(-h,+h,0)
```

### 策略4: Dual-ROI混合 (见上方processDualRoiMono)

```
BC(class0边缘) + AK(class1中心) 并行提取 → 合并 → EPnP
无退化链, 无策略链, 直接processDualRoiMono()处理
```

## MonoPnPSolver 统一求解

**文件**: `src/pose/MonoPnPSolver.cpp:18-131`

```
solve(pts_2d, pts_3d, K):
  18-23  校验: size>=4 && 2d.size()==3d.size()
  26-38  Eigen::Vector3d→cv::Point3f, K→cv::Mat
  45-65  cv::solvePnPRansac(EPNP, 300iter, 8.0px, 0.99, noExtrinsicGuess)
    失败/inlier<4 → return PoseEstimate{fail}
  67-88  cv::solvePnP(ITERATIVE, useExtrinsicGuess=true) on inliers
    抛出异常 → 非致命, 继续用RANSAC结果
  90-131  有效性校验:
    cv::Rodrigues(rvec→R), 转Eigen
    t.z>0, 10<|t|<20000mm
    R,t各分量isfinite
    → PoseEstimate{R,t,success=true,num_points=inlier_count}
```

## 关键数据结构 (include/common/Types.hpp)

| 结构体 | 行 | 关键字段 |
|--------|-----|---------|
| `FeatureSet` | 66-71 | keypoints, descriptors, points, num_keypoints |
| `TemplateData` | 74-88 | keypoints, descriptors, pts_3d, gray_image, corners, angle, image_bool |
| `PipelineResult` | 91-160 | kp_left/right, pts_left_match/template_match, good_matches, disparity, R, t, success, gpnp_success, total_time_ms, n_kp_left, n_matched, n_projected, n_template_match |
| `TrackResult` | 163-175 | pts_left_good, pts_right_good, disparity, fb_error, num_valid |
| `ProjectionResult` | 178-188 | pts_right_projected, valid_mask |
| `MatchResult` | 191-200 | good_matches, pts_left_match, pts_template_match, num_matches, ratio_test_count, cross_check_count, homography_count |
| `PoseEstimate` | 203-210 | R(Matrix3d), t(Vector3d), success, num_points |
| `GPNPMonitor` | 213-225 | initial_cost, final_cost, iterations, failure_reason |
| `LogEntry` | 228-242 | frame_id, n_features, n_matches, median_disparity, t_ms |
| `TrackingState` | 245-258 | R_prev, t_prev, has_cache, frame_count, logs |
| `RoiRect` | 291-302 | x,y,width,height, valid()=width>0&&height>0 |
| `RoiGroup` | 305-315 | primary, secondary, is_dual |
| `MonoConfig` | StereoTracker.hpp:72-76 | enabled, akaze_min_area, tiny_max_area |
| `MadFilterResult` | 318-324 | filtered_points, filter_mask, degraded |
| `YoloConfig` | 327-342 | model_path, device_type, conf_threshold, target_class_id, roi_expand_ratio |
| `Detection` | 374-378 | class_id, confidence, bbox |
| `StereoCameraParams` | 28-47 | K, K_inv, R_rl, t_rl, focal_length, baseline |
| `TrackerConfig` | 50-68 | scale, gpnp_min_pts, use_initial_pnp, template_real_w/h, akaze_min_area, tiny_max_area, dual_roi_secondary_expand, dual_roi_akaze_scale |

## ROI生成链

```
YoloRoiProvider::detect(left,right) (YoloRoiProvider.cpp:37-68):
  YoloDetector::detect(left) + detect(right)  // ONNX推理
  若任一侧失败→返回{RoiGroup{}, RoiGroup{}}
  → roi_gen_->generateStereoGroup()

YoloRoiProvider::detectMono(left) (cpp:70-98):  // 单目专用
  YoloDetector::detect(left)
  → roi_gen_->generateGroup()

RoiGenerator::generateGroup() (RoiGenerator.cpp:73-96):
  1. generate(class=0) → 最高置信度检测框→detectionToRoi(expand+clamp)
  2. if primary.area > 490000(700×700) && generate(class=1).valid:
       return RoiGroup{pri, sec, true}  // DUAL
  3. return RoiGroup{pri, {}, false}     // SINGLE

RoiGenerator::generateStereoGroup() (cpp:184-244):
  左右各generateGroup→normalizeStereoPair(尺寸对齐)
  仅当两侧都dual时才保留dual

detectionToRoi() (cpp:250-294):
  expand_ratio扩展→clamp边界→强制min_size→re-clamp
```

## FeatureExtractor基类 (include/feature/FeatureExtractor.hpp)

```cpp
enum class StrategyType { Akaze, BinaryCorner, TinyTarget };
class FeatureExtractor {
  virtual std::string name() const = 0;          // PnP路由 + 退化去重
  virtual PipelineResult extract(left_gray, right_gray, left_color, right_color) = 0;
  virtual PipelineResult extractMono(gray, color);  // 默认: extract(gray,empty,color,empty)
};
```

## 文件索引

| 目录/文件 | 内容 |
|-----------|------|
| `main.cpp` | 程序入口, 配置解析, 帧循环 |
| `include/common/Types.hpp` | 所有强类型结构体 |
| `include/common/Config.hpp` | makeTrackerConfig, makeYoloConfig 工厂函数 |
| `include/common/GeometryUtils.hpp` | 内联几何工具 |
| `include/tracker/StereoTracker.hpp` | StereoTracker类声明, MonoConfig |
| `src/tracker/StereoTracker.cpp` | 全部核心逻辑: process, processMono, processDualRoi, processDualRoiMono, configureStrategyChain, dispatchPnP, prepareDualBcTemplate |
| `include/feature/FeatureExtractor.hpp` | 基类, StrategyType枚举 |
| `include/feature/AkazeExtractor.hpp` | 纯AKAZE提取(无光流/投影) |
| `src/feature/AkazeExtractor.cpp` | extract(), extractTemplate() |
| `include/feature/AkazeGpnpExtractor.hpp` | AKAZE+光流+投影+匹配 |
| `src/feature/AkazeGpnpExtractor.cpp` | extract(), extractMono(), setTemplateData() |
| `include/feature/BinaryCornerExtractor.hpp` | BC提取器, Config, TemplateData |
| `src/feature/BinaryCornerExtractor.cpp` | extract(), extractMono(), extractFromBinary(), extractCornersFromContour(), matchCorners(), reorderByGeometry(), keepLargestRegion(), fillHoles(), smoothBoundary(), findBestMatch() |
| `include/feature/TinyTargetExtractor.hpp` | TT提取器, Config |
| `src/feature/TinyTargetExtractor.cpp` | extract(), extractMono(), extract4Corners(), matchTemplate(), selectBestComponent(), refineCorners() |
| `include/feature/OpticalFlowTracker.hpp` | LK光流+FB校验 |
| `src/feature/OpticalFlowTracker.cpp` | track() |
| `include/feature/MadDisparityFilter.hpp` | MAD视差滤波 |
| `src/feature/MadDisparityFilter.cpp` | filter() |
| `include/matching/TemplateMatcher.hpp` | 三阶段模板匹配 |
| `src/matching/TemplateMatcher.cpp` | match(): RatioTest→CrossCheck→HomographyRANSAC |
| `include/pose/InitialPnPSolver.hpp` | RANSAC+ITERATIVE初始PnP |
| `src/pose/InitialPnPSolver.cpp` | solve() |
| `include/pose/GPnPSolver.hpp` | Eigen LM GPNP优化 |
| `src/pose/GPnPSolver.cpp` | solve(): 7维[q,t], 交叉射线残差 |
| `include/pose/MonoPnPSolver.hpp` | 单目EPnP |
| `src/pose/MonoPnPSolver.cpp` | solve(): EPnP RANSAC→ITERATIVE精化→有效性校验 |
| `include/stereo/StereoProjector.hpp` | 视差→深度→投影 |
| `src/stereo/StereoProjector.cpp` | project() |
| `include/detection/YoloDetector.hpp` | ONNX YOLO推理(header-only) |
| `include/detection/YoloRoiProvider.hpp` | 检测→RoiGroup facade |
| `src/detection/YoloRoiProvider.cpp` | detect(), detectMono() |
| `include/detection/RoiGenerator.hpp` | ROI生成器 |
| `src/detection/RoiGenerator.cpp` | generate(), generateGroup(), generateStereoGroup(), detectionToRoi(), normalizeStereoPair() |
| `include/visualization/Visualizer.hpp` | 多面板调试图 |
| `src/visualization/Visualizer.cpp` | 渲染逻辑 |
| `include/utils/PoseUtils.hpp` | 位姿工具 |
| `src/utils/PoseUtils.cpp` | loadTemplates(), readCorners(), calculateOverlap(), extractAndNormalizeRoi(), orderPoints() |
| `config/tracker_config.json` | 默认配置 |
