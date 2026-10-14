scripts/ 目录 Python 脚本说明
================================
最后更新: 2026-10-12

1. camera_capture.py
   摄像头预览 / 抓拍 / 连拍 / 流式导出工具。用于采集实测图像素材。

2. calibrate_camera.py
   相机内参标定 (棋盘格)。读取 data/calib/ 下多张棋盘图, cv2.calibrateCamera
   求解, 输出 scripts/camera_intrinsics.json (K / 畸变系数 / 规格校验字段)。
   硬编码棋盘 7x10 格 -> 内角点 9x6, 单格 18mm。
   注意: objp 点序必须 np.mgrid[0:cols, 0:rows].T (cols 在前)。

3. solve_chessboard_pose.py
   棋盘格位姿解算 (PnP, 批量)。遍历 IMAGE_DIR 下所有图片, 逐张
   findChessboardCorners -> solvePnP(IPPE), 平移向量 T(mm) 写在
   *_pose.png 左上角并叠加角点/坐标轴。内参优先读 camera_intrinsics.json,
   缺失时用硬编码兜底值。注意 T 原点 = 第一个检测到的内角点 (非棋盘外角)。

4. annotate_points.py
   交互式特征点标注。在目标图上依次点 2xN 个点 (前 N=class0 整体轮廓,
   后 N=class1 中心), 输出 "Corner_N: X, Y" 格式 txt, 与 C++ 侧
   PoseUtils::readCorners() 正则兼容。产物: class0_points.txt /
   class1_points.txt (在 data/big/img_1.png 上手工标注)。

5. generate_synthetic_dataset.py
   合成训练数据集生成。复用平面单应透视投影把目标贴图投影到背景任意
   位置 (不截断), yaw 0-360 全覆盖 + pitch/roll ±10°, 右图经 T_disp
   单应产生视差; 每图输出 class0/class1 特征点 txt + manifest.json。

6. generate_approach_sequence.py
   连续接近 + 旋转单目序列合成。复用 generate_synthetic_dataset.py 的
   单应几何, 把随机采样换成确定性轨迹 (几何增长的尺寸 / 匀速 yaw /
   慢摆 pitch/roll, 第 0 帧严格正位)。输出 frame_XXXXXX.png 单目序列
   (匹配 SequenceSource), 终止条件 = class1 特征点越出画布。

7. generate_stereo_crop_dataset.py
   单目序列 -> 双目数据集 (裁剪法)。左图裁右侧 d 列、右图裁左侧 d 列
   (两图分辨率一致 W-d, u_L - u_R = d > 0 标正视差); 右图额外 warpAffine
   亚像素垂直平移 1~2px 模拟装配失配。d 支持 ramp (线性逼近) / random
   (每帧随机), dy 支持 fixed (全序列一次采样) / random; 可选左右独立
   光度增强。输出 left_XXXXXX.png / right_XXXXXX.png + manifest.json,
   直接被 DirectoryStereoSource 消费。参数在脚本顶部 CONFIG 硬编码。

8. apriltag_worker.py
   AprilTag Python sidecar worker (pose_compare2 专用)。pupil-apriltags
   检测 + 位姿解算, 与 C++ 端 AprilTagExtractor 通过 stdin/stdout JSON
   管道协议通信。容器内需 pip install pupil-apriltags。
   注意: 回包不可含 JSON null (C++ FileStorage 会崩溃), 无值字段省略键。

9. extract_log_to_xlsx.py
   从 tracking_log.txt 提取每帧位姿写入 xlsx (帧序 / rvec / tvec(mm) /
   择优重投影误差 / 状态)。PnP 失败帧保留行、数值留空。

依赖关系:
- 5/6 共享平面单应几何代码 (6 复用 5 的函数)
- 4 的输出格式被 5/6 和 C++ readCorners() 共同消费
- 7 独立 (纯裁剪, 无单应投影)
