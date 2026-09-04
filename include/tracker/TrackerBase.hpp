#pragma once

/**
 * @file TrackerBase.hpp
 * @brief MonoTracker / StereoTracker 共享基础设施。
 */

#include "common/Config.hpp"
#include "common/LogConfig.hpp"
#include "common/Types.hpp"
#include "feature/FeatureExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "visualization/Visualizer.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace gpnp {

class AkazeGpnpExtractor;

class TrackerBase {
public:
    virtual ~TrackerBase() = default;

    // ---- Logging & Output ----
    const std::vector<LogEntry>& getLogs() const { return state_.logs; }
    void printLogs() const;
    void setOutputDir(const std::string& dir) { output_dir_ = dir; }
    void setVerboseConsole(bool v) { verbose_console_ = v; g_verbose_console = v; }
    void setVisualizeDetailed(bool v) { visualize_detailed_ = v; }
    void setFrameNumber(int n) { current_frame_ = n; }
    bool verboseConsole() const { return verbose_console_; }
    const StereoCameraParams& cameraParams() const { return camera_; }
    const TrackerConfig& config() const { return config_; }
    int frameCount() const { return state_.frame_count; }
    void clearCache() {
        state_ = TrackingState{};
        cache_age_ = 0;
        has_ema_ = false;
        resetStickiness();
        has_prev_c1_ = false;
    }
    /// YOLO 无检测的 skip 帧通知 —— 让位姿 seed 缓存老化（不进 process 的帧也会失效）
    void onFrameSkipped() { if (state_.has_cache) ++cache_age_; }

protected:
    TrackerBase() = default;

    // ---- 提取器初始化（子类构造中调用）----
    void initExtractors(const std::string& template_path,
                        const BinaryCornerExtractor::Config& binary_cfg,
                        const std::string& binary_template_dir,
                        const TinyTargetExtractor::Config& tiny_cfg,
                        const std::string& tiny_template_dir);

    // ---- 共享方法 ----
    void configureStrategyChain(int roi_area, bool is_class1 = false);
    static RoiRect validateRoi(const RoiRect* roi, const cv::Size& img, const std::string& name);
    static std::pair<cv::Mat, cv::Mat> loadImage(const cv::Mat& img);
    void finalizePose(PipelineResult& result, const PoseEstimate& pose);
    void addLogEntry(const PipelineResult& result, bool is_first, bool fallback_used);

    // ---- 时序连贯性 (temporal): 策略粘滞 + 位姿链 + 运动门控 ----
    // 策略档位: 1=TinyTarget(远) 2=BinaryCorner(中) 3=AkazeGpnp(中近/大)
    static int bandFromStrategyName(const std::string& name);
    static const char* strategyNameFromBand(int band);
    void effectiveThresholds(bool is_class1, int& akaze_t, int& tiny_t) const;
    int nominateBand(int roi_area, bool is_class1) const;      ///< 纯面积分类（现有三分段语义）
    void applyBandChain(int band, bool is_class1);             ///< 档位 → 提取器 + 单向退化链
    int resolveLockedStrategy(int band_raw, bool is_class1);   ///< 粘滞状态机: 迟滞+去抖+越级硬切
    void resetStickiness();
    /// 帧末反馈: 锁定档位跟随实际胜者（面积说该切且实际切成了 → 立即对齐；
    /// 锁定策略被 fallback 顶掉 → 计数，连续 locked_fail_limit 帧后跟随胜者）
    void updateStickinessFromWinner(const std::string& winner_name);
    /// 位姿 seed 是否可用 (temporal.enabled && 有缓存 && 未超龄)
    bool seedActive() const;
    /// 运动一致性门控: Δt ≤ max_trans_ratio×|t_seed| 且 Δθ ≤ max_rot_deg；
    /// ext 档位 ≠ 锁定档位（策略切换帧）或 widened=true（冷重解二道门）时阈值 ×switch_margin
    bool motionGatePass(const PoseEstimate& pose, const PoseSeed& seed,
                        const FeatureExtractor* ext, bool widened) const;
    /// 将时序观测字段填入 PipelineResult（日志用）
    void fillTemporalMeta(PipelineResult& result) const;

    // ---- 时序连贯性状态 ----
    bool has_prev_c1_{false};
    bool prev_use_c1_{false};
    bool has_ema_{false};
    double ema_area_{0.0};
    int cache_age_{0};                  ///< 距上次成功位姿的帧数（finalizePose 成功清零）
    bool has_sticky_state_{false};
    int sticky_locked_{0};              ///< 当前锁定档位
    int sticky_pending_{0};             ///< 待确认的提名档位
    int pending_count_{0};              ///< 提名连续确认计数
    int locked_fail_count_{0};          ///< 锁定策略被 fallback 顶掉的连续帧数
    int nominated_band_{0};             ///< 本帧面积提名档位（raw）
    bool sticky_hold_{false};           ///< 本帧粘滞抑制了切换
    std::string sticky_switch_reason_{"init"};

    // ---- 共享数据 ----
    StereoCameraParams camera_;
    TrackerConfig config_;
    TemplateData template_;
    std::unique_ptr<AkazeGpnpExtractor> akaze_extractor_;
    std::unique_ptr<BinaryCornerExtractor> binary_extractor_;
    std::unique_ptr<TinyTargetExtractor> tiny_extractor_;
    FeatureExtractor* extractor_ = nullptr;
    std::vector<FeatureExtractor*> fallback_extractors_;
    int akaze_min_area_ = 40000, tiny_max_area_ = 800;
    int akaze_min_area_class1_ = 0, tiny_max_area_class1_ = 0;
    int binary_roi_pad_ = 0, tiny_roi_pad_ = 0;
    std::string output_dir_;
    bool verbose_console_ = true;
    bool visualize_detailed_ = true;  ///< true=Debug 模式生成 per-strategy 面板; false=仅三维轴叠加图
    int current_frame_ = 0;           ///< 当前输入帧号（用于可视化文件名）
    std::unique_ptr<Visualizer> visualizer_;
    TrackingState state_;
};

} // namespace gpnp
