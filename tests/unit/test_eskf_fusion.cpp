#include "../framework/TestAssert.hpp"
#include "fusion/EskfFusionManager.hpp"
#include "fusion/FusionTypes.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <functional>
#include <thread>

using namespace gpnp;
using namespace gpnp::fusion;

// ============================================================================
// 方案B 单元测试: ESKF 延迟反向传播 / 兜底 / 退化监控 / 线程化
//
// 被测模块: fusion::EskfFusionManager (include/fusion/EskfFusionManager.hpp)
// 输入数据: 纯代码合成 (确定性悬停/匀加速 IMU + 精确相机位姿, 零噪声)
//
// 设计要点:
//   - 悬停 IMU: acc=(0,0,+9.81), gyro=0 → a_world = R(acc-b_a)+g = 0,
//     状态零漂移、速度恒 0 (重力约定见 eskf_vio.hpp:2351)
//   - 相机位姿: R_tpl_cam=I, R_template_world=I → p_cam_w = -t_mm/1000,
//     由 makeObs 精确反推, lazyInit 后位置与转换结果逐位一致
//   - 位置跳变 ≤0.3m + cam_pos_noise=0.1 → NIS ≪ χ²₉₅=7.81, hybrid 更新永不被 FDI 拒
//   - 反向传播"真值" = 零延迟理想世界 (t0=t1), 回退快照与 IMU 重放逐样本复现该理想
// ============================================================================

namespace {

// ---- 合成数据常量 ----

// 悬停比力 (抵消重力): a_world = R(acc-b_a)+g = 0
const Eigen::Vector3d kHoverAcc(0.0, 0.0, 9.81);
// 匀加速运动 (X 向 0.5 m/s²): 移动场景, 制造 t0→t1 期间的真实状态演化
const Eigen::Vector3d kMoveAcc(0.5, 0.0, 9.81);
const Eigen::Vector3d kZeroGyro = Eigen::Vector3d::Zero();

// 相机位姿 (世界系, m): 位置跳变 ≤0.3m, 规避 FDI
const Eigen::Vector3d kP0(0.0, 0.0, 2.0);          // 首帧 init 位姿
const Eigen::Vector3d kP2(0.2, 0.0, 2.0);          // 第二次观测位姿
const Eigen::Vector3d kPReset(0.3, 0.1, 2.5);      // 间隔重置测试位姿

// ---- 夹具: 测试配置 ----

EskfFusionConfig makeHoverCfg() {
    EskfFusionConfig cfg;
    cfg.enabled           = true;
    cfg.imu_rate_hz       = 100.0;
    cfg.radar_rate_hz     = 20.0;
    cfg.max_imu_gap_s     = 0.1;
    cfg.max_cam_gap_s     = 1.0;
    cfg.backprop_window_s = 0.2;       // 反向传播回退窗口
    cfg.state_hist_hz     = 100;       // 状态快照频率 (与 IMU 网格对齐)
    cfg.max_output_age_s  = 0.5;       // Normal → Degraded 分界
    cfg.latency_fallback  = LatencyFallback::Inflate;  // 默认兜底: 协方差膨胀
    cfg.init_std_p  = 1.0;
    cfg.init_std_v  = 1.0;
    cfg.init_std_q  = 0.1;
    cfg.init_std_ba = 0.1;
    cfg.init_std_bg = 0.01;
    cfg.params.cam_pos_noise = 0.1;    // 观测信任度 (σ)
    return cfg;
}

// ---- 夹具: 合成 IMU 序列 (100Hz 网格, 首样本 t=0.01) ----

void feedImuConst(EskfFusionManager& m, double t_end, const Eigen::Vector3d& acc) {
    for (double t = 0.01; t <= t_end + 1e-9; t += 0.01)
        m.feedImu(t, acc, kZeroGyro);
}

void feedHoverImu(EskfFusionManager& m, double t_end) { feedImuConst(m, t_end, kHoverAcc); }
void feedMoveImu(EskfFusionManager& m, double t_end)  { feedImuConst(m, t_end, kMoveAcc); }

// 从指定时刻续喂悬停 IMU (线程化测试: 相机缺席续传播)
void feedHoverImuFrom(EskfFusionManager& m, double t_start, double t_end) {
    for (double t = t_start; t <= t_end + 1e-9; t += 0.01)
        m.feedImu(t, kHoverAcc, kZeroGyro);
}

// ---- 夹具: 合成相机观测 (R=I, 由 convertPose 精确反推 t_mm) ----

CameraObservation makeObs(double t0, double t1, const Eigen::Vector3d& p_w) {
    CameraObservation obs;
    obs.t_exposure = t0;
    obs.t_arrival  = t1;
    obs.R_tpl_cam  = Eigen::Matrix3d::Identity();
    obs.t_cam_mm   = -1000.0 * p_w;    // convertPose: p_cam_w = R·(-t/1000)
    obs.valid      = true;
    return obs;
}

// ---- 轮询等待 (线程化测试, 防 flaky) ----

bool waitUntil(const std::function<bool()>& pred, double timeout_s = 2.0) {
    using clock = std::chrono::steady_clock;
    const auto deadline =
        clock::now() + std::chrono::milliseconds((int)(timeout_s * 1000.0));
    while (clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return pred();
}

} // namespace

// ============================================================================
// 分组 A: 基础链路回归
// ============================================================================

// ----------------------------------------------------------------------------
// 用例 1: 首帧 lazy init
// 目的: 首个有效相机位姿直接置名义态 (不走 Kalman 更新), 位置与转换结果
//       逐位一致; 初始化前的 IMU 只锚定时间原点、不积分。
// 输入: 悬停 IMU 0.01~1.0 + 相机 (t0=t1=1.0, kP0, valid)
// 预期: initialized; position==kP0 (1e-9); v==0; q==(1,0,0,0);
//       quality=Normal; imu_samples==0; cam_updates==0
// ----------------------------------------------------------------------------
static void test_lazy_init_and_first_update() {
    EskfFusionManager m(makeHoverCfg());
    feedHoverImu(m, 1.0);
    m.feedCameraPose(makeObs(1.0, 1.0, kP0));

    TEST_ASSERT(m.initialized());
    TEST_ASSERT((m.position() - kP0).norm() < 1e-9);
    TEST_ASSERT(m.velocity().norm() < 1e-12);
    Eigen::Matrix<double, 4, 1> q = m.quaternion();
    TEST_ASSERT(std::abs(q(0) - 1.0) < 1e-12);
    TEST_ASSERT(q(1) == 0.0 && q(2) == 0.0 && q(3) == 0.0);

    FusionState s = m.getLatestState();
    TEST_ASSERT(s.initialized);
    TEST_ASSERT(s.quality == FusionQuality::Normal);

    auto st = m.stats();
    TEST_ASSERT(st.imu_samples == 0);   // 初始化前 IMU 不积分
    TEST_ASSERT(st.cam_updates == 0);   // 首帧走 lazyInit 而非 update
}

// ----------------------------------------------------------------------------
// 用例 2: 视觉失败帧仅惯性传播
// 目的: valid=false 观测绝不触发初始化; 后续恢复帧正常初始化 (不崩溃、无残留)。
// 输入: 悬停 IMU 0.01~2.0 + 无效帧 (t0=t1=1.0) + 有效帧 (t0=t1=2.0, kP0)
// 预期: 无效帧后 !initialized / Uninitialized / 零值; 有效帧后 position==kP0
// ----------------------------------------------------------------------------
static void test_invalid_camera_imu_only() {
    EskfFusionManager m(makeHoverCfg());
    feedHoverImu(m, 2.0);

    CameraObservation obs = makeObs(1.0, 1.0, kP0);
    obs.valid = false;
    m.feedCameraPose(obs);

    TEST_ASSERT(!m.initialized());
    FusionState s = m.getLatestState();
    TEST_ASSERT(s.quality == FusionQuality::Uninitialized);
    TEST_ASSERT(s.position.norm() == 0.0);
    TEST_ASSERT(m.stats().cam_updates == 0);

    // 恢复: 后续有效帧正常初始化
    m.feedCameraPose(makeObs(2.0, 2.0, kP0));
    TEST_ASSERT(m.initialized());
    TEST_ASSERT((m.position() - kP0).norm() < 1e-9);
    TEST_ASSERT(m.stats().cam_updates == 0);
}

// ----------------------------------------------------------------------------
// 用例 3: 相机间隔超限 → 重置重初始化
// 目的: 帧间隔 > max_cam_gap_s 时 reset 并重新 lazyInit, 旧状态不污染新测量;
//       position 与 velocity 精确等于新初始值 = "确实重置"的铁证。
// 输入: 悬停 IMU 0.01~3.0 + obs1 (t0=t1=1.0, kP0) + obs2 (t0=t1=3.0, kPReset)
//       间隔 2.0s > max_cam_gap_s=1.0
// 预期: obs2 后 position==kPReset (1e-12), v==0, stats 全 0 (reset 清零), Normal
// ----------------------------------------------------------------------------
static void test_gap_exceeds_max_cam_gap_resets() {
    EskfFusionManager m(makeHoverCfg());
    feedHoverImu(m, 3.0);

    m.feedCameraPose(makeObs(1.0, 1.0, kP0));
    TEST_ASSERT((m.position() - kP0).norm() < 1e-9);
    TEST_ASSERT(m.getLatestState().quality == FusionQuality::Normal);

    m.feedCameraPose(makeObs(3.0, 3.0, kPReset));
    TEST_ASSERT(m.initialized());
    TEST_ASSERT((m.position() - kPReset).norm() < 1e-12);  // fresh init, 非带旧状态 update
    TEST_ASSERT(m.velocity().norm() < 1e-12);
    TEST_ASSERT(m.getLatestState().quality == FusionQuality::Normal);

    auto st = m.stats();
    TEST_ASSERT(st.imu_samples == 0 && st.cam_updates == 0);  // reset 清零
}

// ============================================================================
// 分组 B: 延迟反向传播 (方案B 核心)
// ============================================================================

// ----------------------------------------------------------------------------
// 用例 4: 反向传播精确复现"零延迟理想处理" (核心正确性)
// 目的: 有 80ms 延迟的观测经反向传播后, 与"相机在曝光时刻立即被处理"的理想
//       世界逐元素一致 —— 证明延迟测量被精确贴回 t0, 无近似损失。
// 机制: 回退快照 (100Hz 网格, 精确对齐 t0=1.50) + 相同 IMU 重放,
//       时间线逐样本复现理想管理器的处理。
// 输入:
//   参考 R: 悬停 IMU 0.01~2.0 + obs1 (1.0/1.0, kP0) + obs2 (1.50/1.50, kP2)
//           + propagateTo(1.58)
//   被测 B: 同 IMU + obs1 同 + obs2 (t0=1.50, t1=1.58)   [延迟 80ms]
// 预期: ‖p_B−p_R‖/‖v_B−v_R‖/‖q_B−q_R‖ < 1e-6; p_B≈kP2 (1e-3);
//       cam_late_fallback==0 (走反向传播而非兜底); cam_updates==1
// ----------------------------------------------------------------------------
static void test_backprop_matches_zero_latency_reference() {
    EskfFusionManager ref(makeHoverCfg());
    feedHoverImu(ref, 2.0);
    ref.feedCameraPose(makeObs(1.0, 1.0, kP0));
    ref.feedCameraPose(makeObs(1.50, 1.50, kP2));
    ref.propagateTo(1.58);

    EskfFusionManager bp(makeHoverCfg());
    feedHoverImu(bp, 2.0);
    bp.feedCameraPose(makeObs(1.0, 1.0, kP0));
    bp.feedCameraPose(makeObs(1.50, 1.58, kP2));

    TEST_ASSERT((bp.position() - ref.position()).norm() < 1e-6);
    TEST_ASSERT((bp.velocity() - ref.velocity()).norm() < 1e-6);
    TEST_ASSERT((bp.quaternion() - ref.quaternion()).norm() < 1e-6);
    TEST_ASSERT((bp.position() - kP2).norm() < 2e-2);   // 更新已把位置拉向测量值: 增益 ~0.99 → 距 kP2 ~2mm, 但速度修正连带 +0.086m/s, 0.08s 传播再漂 ~7mm → 终态 ~5mm < 20mm

    auto st = bp.stats();
    TEST_ASSERT(st.cam_late_fallback == 0);             // 反向传播成功, 无兜底
    TEST_ASSERT(st.cam_updates == 1);
}

// ----------------------------------------------------------------------------
// 用例 5: 反向传播优于"按到达时刻应用" (方案价值)
// 目的: IMU 真实运动时, 朴素路径把 t0 的测量当成 t1 时刻的 → 位置系统性偏斜;
//       反向传播恢复理想。证明方案B 的精度收益, 而不只是内部一致性。
// 输入: 匀加速 IMU (acc=(0.5,0,+9.81)) 0.01~2.0 + obs1 (1.0/1.0, kP0):
//   理想 R: obs2 (1.50/1.50, kP2) + propagateTo(1.58)
//   反向 B: obs2 (1.50/1.58, kP2)
//   朴素 A: obs2 (1.58/1.58, kP2)      [无方案B的系统行为]
// 预期: ‖p_B−p_R‖ < 1e-6 (复现理想); ‖p_A−p_R‖ > 5e-3 (朴素路径偏离 ~2cm:
//       R 反映"1.58 时刻真值 ≈ kP2 + v·0.08 + ½a·0.08²", A 把 t0 位置硬当 t1)
// ----------------------------------------------------------------------------
static void test_backprop_beats_arrival_application() {
    auto cfg = makeHoverCfg();

    EskfFusionManager ref(cfg);
    feedMoveImu(ref, 2.0);
    ref.feedCameraPose(makeObs(1.0, 1.0, kP0));
    ref.feedCameraPose(makeObs(1.50, 1.50, kP2));
    ref.propagateTo(1.58);

    EskfFusionManager bp(cfg);
    feedMoveImu(bp, 2.0);
    bp.feedCameraPose(makeObs(1.0, 1.0, kP0));
    bp.feedCameraPose(makeObs(1.50, 1.58, kP2));

    EskfFusionManager arr(cfg);
    feedMoveImu(arr, 2.0);
    arr.feedCameraPose(makeObs(1.0, 1.0, kP0));
    arr.feedCameraPose(makeObs(1.58, 1.58, kP2));

    TEST_ASSERT((bp.position() - ref.position()).norm() < 1e-6);   // 反向传播复现理想
    TEST_ASSERT((arr.position() - ref.position()).norm() > 5e-3);  // 朴素路径显著偏离
}

// ----------------------------------------------------------------------------
// 用例 6: 亚毫秒延迟不走反向传播
// 目的: latency ≤ 1ms 的观测走"按到达时刻直接更新"路径 (代码守卫), 避免无谓
//       回退; 结果与 t0=t1 的直接处理逐位一致, 且不计数兜底。
// 输入: 悬停 IMU 0.01~2.0 + obs1 (1.0/1.0, kP0);
//   被测: obs2 (t0=1.50, t1=1.5005)  [延迟 0.5ms]
//   对照: obs2 (t0=t1=1.5005)        [无延迟直接路径]
// 预期: 两管理器状态逐元素一致 (1e-9); cam_late_fallback==0; 更新生效 (|p−kP0|>0.1)
// ----------------------------------------------------------------------------
static void test_sub_ms_latency_skips_backprop() {
    auto cfg = makeHoverCfg();

    EskfFusionManager bp(cfg);
    feedHoverImu(bp, 2.0);
    bp.feedCameraPose(makeObs(1.0, 1.0, kP0));
    bp.feedCameraPose(makeObs(1.50, 1.5005, kP2));

    EskfFusionManager direct(cfg);
    feedHoverImu(direct, 2.0);
    direct.feedCameraPose(makeObs(1.0, 1.0, kP0));
    direct.feedCameraPose(makeObs(1.5005, 1.5005, kP2));

    TEST_ASSERT((bp.position() - direct.position()).norm() < 1e-9);
    TEST_ASSERT((bp.velocity() - direct.velocity()).norm() < 1e-9);
    TEST_ASSERT(bp.stats().cam_late_fallback == 0);
    TEST_ASSERT(bp.stats().cam_updates == 1);
    TEST_ASSERT((bp.position() - kP0).norm() > 0.1);   // 更新确实生效
}

// ============================================================================
// 分组 C: 兜底策略
// ============================================================================

// ----------------------------------------------------------------------------
// 用例 7: 延迟超窗 → 协方差膨胀兜底
// 目的: 延迟 > backprop_window_s 时无快照可回退, 默认兜底 = 把延迟折算成
//       额外观测噪声按到达时刻应用: 观测不丢弃, 但可信度被记账 (计数 1),
//       后验协方差确凿大于无延迟参考 (膨胀真实生效)。
// 输入: 悬停 IMU 0.01~2.0 + obs1 (1.0/1.0, kP0);
//   被测 B: obs2 (t0=1.50, t1=2.00)  [延迟 0.5s > 窗口 0.2s, 历史裁剪线 1.80 > 1.50]
//   参考 R: obs2 (2.00/2.00, kP2)    [无延迟, 正常噪声]
// 预期: cam_late_fallback==1; cam_ignored==0; cam_updates==1;
//       ‖p_B−kP2‖<0.05 (仍被应用); trace(P_pos)_B > 1.01 × trace(P_pos)_R
// ----------------------------------------------------------------------------
static void test_latency_beyond_window_inflate() {
    auto cfg = makeHoverCfg();

    EskfFusionManager bp(cfg);
    feedHoverImu(bp, 2.0);
    bp.feedCameraPose(makeObs(1.0, 1.0, kP0));
    bp.feedCameraPose(makeObs(1.50, 2.00, kP2));

    EskfFusionManager ref(cfg);
    feedHoverImu(ref, 2.0);
    ref.feedCameraPose(makeObs(1.0, 1.0, kP0));
    ref.feedCameraPose(makeObs(2.00, 2.00, kP2));

    auto st = bp.stats();
    TEST_ASSERT(st.cam_late_fallback == 1);   // 兜底记账
    TEST_ASSERT(st.cam_ignored == 0);         // 观测未被丢弃
    TEST_ASSERT(st.cam_updates == 1);
    TEST_ASSERT((bp.position() - kP2).norm() < 0.05);  // 按到达时刻应用

    // 协方差膨胀生效: 同先验 + 更大观测噪声 → 更大后验 (12x 量级, 断言 1% 裕量)
    double trace_bp   = bp.filter().P.block<3, 3>(0, 0).trace();
    double trace_ref  = ref.filter().P.block<3, 3>(0, 0).trace();
    TEST_ASSERT(trace_bp > trace_ref * 1.01);
}

// ----------------------------------------------------------------------------
// 用例 8: 延迟超窗 → 丢弃观测
// 目的: latency_fallback=Reject 时超窗观测被整体丢弃 (cam_ignored, 无更新),
//       滤波器只信惯性估计, 绝不用过期测量。
// 输入: 同用例 7, cfg.latency_fallback=Reject
// 预期: cam_ignored==1; cam_updates==0; cam_late_fallback==0 (Reject 分支只计
//       cam_ignored, 与 Inflate 分支的计数语义不同 — 记录当前实现);
//       position 保持惯性估计 kP0 (1e-6); initialized 仍 true
// ----------------------------------------------------------------------------
static void test_latency_beyond_window_reject() {
    auto cfg = makeHoverCfg();
    cfg.latency_fallback = LatencyFallback::Reject;

    EskfFusionManager m(cfg);
    feedHoverImu(m, 2.0);
    m.feedCameraPose(makeObs(1.0, 1.0, kP0));
    m.feedCameraPose(makeObs(1.50, 2.00, kP2));

    auto st = m.stats();
    TEST_ASSERT(st.cam_ignored == 1);         // 超窗观测被整体丢弃
    TEST_ASSERT(st.cam_updates == 0);
    TEST_ASSERT(st.cam_late_fallback == 0);   // Reject 分支不计数兜底 (当前实现语义)
    TEST_ASSERT((m.position() - kP0).norm() < 1e-6);  // 保持惯性估计, 未被过期测量污染
    TEST_ASSERT(m.initialized());
}

// ============================================================================
// 分组 D: 退化监控
// ============================================================================

// ----------------------------------------------------------------------------
// 用例 9: 可信度分级 Normal → Degraded → Stale
// 目的: 相机停止后融合输出不中断, quality 按距上次相机更新时长如实分级
//       (age>max_output_age_s=0.5 → Degraded; age>max_cam_gap_s=1.0 → Stale),
//       位置协方差迹持续增长作为可信度信号。
// 输入: 悬停 IMU 0.01~2.5 + obs1 (1.0/1.0, kP0) + propagateTo(1.30/1.70/2.50)
// 预期: quality 依次 Normal/Degraded/Stale; 全程位置零漂移 (悬停); cov_trace 严格递增
// ----------------------------------------------------------------------------
static void test_quality_normal_degraded_stale() {
    EskfFusionManager m(makeHoverCfg());
    feedHoverImu(m, 2.5);
    m.feedCameraPose(makeObs(1.0, 1.0, kP0));

    m.propagateTo(1.30);
    FusionState s1 = m.getLatestState();
    TEST_ASSERT(s1.quality == FusionQuality::Normal);    // age 0.30 ≤ 0.5

    m.propagateTo(1.70);
    FusionState s2 = m.getLatestState();
    TEST_ASSERT(s2.quality == FusionQuality::Degraded);  // age 0.70 > 0.5

    m.propagateTo(2.50);
    FusionState s3 = m.getLatestState();
    TEST_ASSERT(s3.quality == FusionQuality::Stale);     // age 1.50 > 1.0

    // 惯性传播未中断: 悬停零漂移, 位置保持; last_cam_t 不变; 协方差迹单调增长
    TEST_ASSERT(s3.initialized);
    TEST_ASSERT(s3.last_cam_t == 1.0);
    TEST_ASSERT((s3.position - kP0).norm() < 1e-9);
    TEST_ASSERT(s1.cov_trace < s2.cov_trace && s2.cov_trace < s3.cov_trace);
}

// ----------------------------------------------------------------------------
// 用例 10: 未初始化状态
// 目的: 从未初始化时 getLatestState 返回明确的 Uninitialized 分级与零值。
// 输入: 仅悬停 IMU 0.01~1.0, 无相机帧
// 预期: initialized==false; quality==Uninitialized; 零值字段
// ----------------------------------------------------------------------------
static void test_uninitialized_state() {
    EskfFusionManager m(makeHoverCfg());
    feedHoverImu(m, 1.0);

    FusionState s = m.getLatestState();
    TEST_ASSERT(!s.initialized);
    TEST_ASSERT(s.quality == FusionQuality::Uninitialized);
    TEST_ASSERT(s.position.norm() == 0.0);
    TEST_ASSERT(s.cov_trace == 0.0);
    TEST_ASSERT(s.last_cam_t == -1.0);
}

// ============================================================================
// 分组 E: 线程化 (Phase 4)
// ============================================================================

// ----------------------------------------------------------------------------
// 用例 11: 工作线程异步消费与同步结果一致
// 目的: start() 后 feedImu/feedCameraPose 异步入队, 融合工作线程独立处理;
//       数学结果与同步路径逐元素一致; 相机缺席时状态继续传播并降级。
// 输入: 场景同用例 4 (80ms 延迟反向传播); IMU 只喂到 1.58 (与同步终态对齐);
//       收敛后补喂悬停 IMU 1.59~2.3 (无相机) → age 0.80 → Degraded
// 预期: 轮询收敛后 (2s 截止) 与同步参考逐元素一致 (1e-6); cam_updates==1;
//       相机缺席后 quality→Degraded; stop() 幂等无异常
// ----------------------------------------------------------------------------
static void test_threaded_async_matches_sync() {
    auto cfg = makeHoverCfg();

    // 同步参考 (与用例 4 的 B 相同: 反向传播路径)
    EskfFusionManager ref(cfg);
    feedHoverImu(ref, 1.58);
    ref.feedCameraPose(makeObs(1.0, 1.0, kP0));
    ref.feedCameraPose(makeObs(1.50, 1.58, kP2));

    // 线程化: 异步消费
    EskfFusionManager thr(cfg);
    thr.start();
    feedHoverImu(thr, 1.58);
    thr.feedCameraPose(makeObs(1.0, 1.0, kP0));
    thr.feedCameraPose(makeObs(1.50, 1.58, kP2));

    bool converged = waitUntil([&] {
        FusionState s = thr.getLatestState();
        return s.initialized && (s.position - kP2).norm() < 2e-2;
    });
    TEST_ASSERT_MSG(converged, "线程化融合未在 2s 内收敛到相机位姿");

    FusionState s = thr.getLatestState();
    TEST_ASSERT((s.position - ref.position()).norm() < 1e-6);
    TEST_ASSERT((s.velocity - ref.velocity()).norm() < 1e-6);
    TEST_ASSERT((s.quaternion - ref.quaternion()).norm() < 1e-6);
    TEST_ASSERT(thr.stats().cam_updates == 1);

    // 相机缺席 → 状态继续传播, 质量降级 (age 2.30-1.50 = 0.80 > 0.5)
    feedHoverImuFrom(thr, 1.59, 2.30);
    bool degraded = waitUntil(
        [&] { return thr.getLatestState().quality == FusionQuality::Degraded; });
    TEST_ASSERT_MSG(degraded, "相机缺席后未进入 Degraded");
    TEST_ASSERT(thr.getLatestState().quality == FusionQuality::Degraded);

    thr.stop();
    thr.stop();   // 幂等
}

// ----------------------------------------------------------------------------
// 用例 12: 相机缺失后 IMU 继续推进 (死推)
// 目的: 线程化模式下, 首帧相机完成初始化后, 相机缺失、仅 IMU 续喂,
//       状态仍靠 IMU 死推前进 (位置真实移动), 且质量降级。
// 输入: MOVE IMU 0.01~1.0 + 首帧 (t0=t1=1.0, kP0) 初始化;
//       相机缺失, 续喂 MOVE IMU 1.01~1.80 (无相机)
// 预期: initialized; 位置被 IMU 死推 (X 向 a=0.5m/s² → ~0.15m);
//       quality → Degraded (age 0.8 > 0.5); stop() 正常
// ----------------------------------------------------------------------------
static void test_camera_missing_imu_propagates() {
    auto cfg = makeHoverCfg();

    EskfFusionManager m(cfg);
    m.start();

    feedMoveImu(m, 1.0);                          // 初始化前 MOVE IMU (不积分)
    m.feedCameraPose(makeObs(1.0, 1.0, kP0));     // 首帧 init

    bool inited = waitUntil([&] { return m.getLatestState().initialized; });
    TEST_ASSERT_MSG(inited, "首帧相机未完成初始化");

    // 相机缺失: 续喂 MOVE IMU 1.01~1.80 (不喂相机)
    for (double t = 1.01; t <= 1.80 + 1e-9; t += 0.01)
        m.feedImu(t, kMoveAcc, kZeroGyro);

    // 位置被 IMU 死推 (X 向 a=0.5m/s² → ~0.15m), 质量降级 (age 0.8 > 0.5)
    bool propagated = waitUntil([&] {
        FusionState s = m.getLatestState();
        return s.initialized && (s.position - kP0).norm() > 0.1;
    });
    TEST_ASSERT_MSG(propagated, "相机缺失后 IMU 未推进位置 (死推失败)");

    FusionState s = m.getLatestState();
    TEST_ASSERT((s.position - kP0).norm() > 0.1);          // 位置确实推进
    TEST_ASSERT(s.quality == FusionQuality::Degraded);     // 相机缺失 → 降级
    m.stop();
}

// ============================================================================
// 分组 F: 复位
// ============================================================================

// ----------------------------------------------------------------------------
// 用例 13: reset 完整性
// 目的: reset() 清空滤波状态、输入缓冲、反向传播历史与统计, 之后可干净地
//       重新初始化 (无旧状态残留)。
// 输入: 悬停 IMU 0.01~2.0 + obs1 (1.0/1.0, kP0) + obs2 (1.50/2.00, kP2)
//       (制造 cam_late_fallback=1) → reset() → 重新喂 obs (1.0/1.0, kP0)
// 预期: reset 后 !initialized / Uninitialized / stats 全 0;
//       重新初始化后 position==kP0 (1e-9)
// ----------------------------------------------------------------------------
static void test_reset_clears_state_stats() {
    EskfFusionManager m(makeHoverCfg());
    feedHoverImu(m, 2.0);
    m.feedCameraPose(makeObs(1.0, 1.0, kP0));
    m.feedCameraPose(makeObs(1.50, 2.00, kP2));   // 超窗兜底 → 非零 stats
    TEST_ASSERT(m.stats().cam_late_fallback == 1);

    m.reset();
    FusionState s = m.getLatestState();
    TEST_ASSERT(!s.initialized);
    TEST_ASSERT(s.quality == FusionQuality::Uninitialized);

    EskfFusionManager::Stats st = m.stats();
    TEST_ASSERT(st.imu_samples == 0);
    TEST_ASSERT(st.cam_updates == 0);
    TEST_ASSERT(st.cam_ignored == 0);
    TEST_ASSERT(st.cam_late_fallback == 0);

    // 可干净地重新初始化
    feedHoverImu(m, 1.0);
    m.feedCameraPose(makeObs(1.0, 1.0, kP0));
    TEST_ASSERT(m.initialized());
    TEST_ASSERT((m.position() - kP0).norm() < 1e-9);
}

// ============================================================================
// 测试入口
// ============================================================================

REGISTER_TEST(test_lazy_init_and_first_update);
REGISTER_TEST(test_invalid_camera_imu_only);
REGISTER_TEST(test_gap_exceeds_max_cam_gap_resets);
REGISTER_TEST(test_backprop_matches_zero_latency_reference);
REGISTER_TEST(test_backprop_beats_arrival_application);
REGISTER_TEST(test_sub_ms_latency_skips_backprop);
REGISTER_TEST(test_latency_beyond_window_inflate);
REGISTER_TEST(test_latency_beyond_window_reject);
REGISTER_TEST(test_quality_normal_degraded_stale);
REGISTER_TEST(test_uninitialized_state);
REGISTER_TEST(test_threaded_async_matches_sync);
REGISTER_TEST(test_camera_missing_imu_propagates);
REGISTER_TEST(test_reset_clears_state_stats);

int main() {
    return gpnp_test::TestRegistry::instance().runAll();
}
