#pragma once

#include "common/Config.hpp"
#include "common/Types.hpp"
#include "feature/FeatureExtractor.hpp"
#include "feature/MadDisparityFilter.hpp"
#include "pose/GPnPSolver.hpp"
#include "pose/InitialPnPSolver.hpp"
#include "visualization/Visualizer.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"

namespace gpnp {

// 预初始化提取器的前置声明
class AkazeGpnpExtractor;
class BinaryCornerExtractor;
class TinyTargetExtractor;

class StereoTracker {
public:
    /// 扩展构造函数：预初始化全部三种特征提取器。
    /// 每帧调用 configureStrategyChain() 来选择当前活跃的策略链。
    StereoTracker(const Eigen::Matrix3d& K,
                  const Eigen::Matrix3d& R_rl,
                  const Eigen::Vector3d& t_rl,
                  const std::string& template_path,
                  const TrackerConfig& config,
                  const BinaryCornerExtractor::Config& binary_cfg,
                  const std::string& binary_template_dir,
                  const TinyTargetExtractor::Config& tiny_cfg,
                  const std::string& tiny_template_dir);

    ~StereoTracker();

    StereoTracker(const StereoTracker&) = delete;
    StereoTracker& operator=(const StereoTracker&) = delete;

    // ========================================================================
    // Public API
    // ========================================================================

    /// Process one stereo frame with optional ROI group.
    /// 根据 ROI 面积自动选择提取策略，并在使用非 AKAZE 提取器时对 ROI 进行填充扩展。
    /// 当 RoiGroup::is_dual 为 true 时，委托给 processDualRoi()。
    PipelineResult process(const cv::Mat& left_img,
                           const cv::Mat& right_img,
                           bool visualize = false,
                           const RoiGroup* left_group = nullptr,
                           const RoiGroup* right_group = nullptr);

    /// 从文件路径处理，可选 ROI 组。
    PipelineResult process(const std::string& left_path,
                           const std::string& right_path,
                           bool visualize = false,
                           const RoiGroup* left_group = nullptr,
                           const RoiGroup* right_group = nullptr);

    // ========================================================================
    // Mono API — 单目模式（仅左图，PnP 解算）
    // ========================================================================

    /// 单目模式配置
    struct MonoConfig {
        bool enabled = false;             ///< true 时 processMono() 生效
        int akaze_min_area = 40001;
        int tiny_max_area = 800;
    };

    /// 设置单目模式配置
    void setMonoConfig(const MonoConfig& cfg) { mono_cfg_ = cfg; }
    const MonoConfig& monoConfig() const { return mono_cfg_; }

    /// 单目帧处理（仅左图）：ROI → 策略分发 → 单图特征提取 → PnP 解算
    PipelineResult processMono(const cv::Mat& left_img,
                               bool visualize = false,
                               const RoiRect* left_roi = nullptr);

    void clearCache();

    // ========================================================================
    // Logging
    // ========================================================================

    const std::vector<LogEntry>& getLogs() const;
    void printLogs() const;

    /// 设置可视化图像的输出目录。
    void setOutputDir(const std::string& dir) { output_dir_ = dir; }

    const StereoCameraParams& cameraParams() const { return camera_; }
    const TemplateData& templateData() const { return template_; }
    const TrackerConfig& config() const { return config_; }
    int frameCount() const { return state_.frame_count; }

private:
    StereoCameraParams camera_;
    TrackerConfig config_;
    TemplateData template_;

    // ========================================================================
    // 预初始化提取器（在构造函数中创建一次，每帧复用）
    // ========================================================================
    std::unique_ptr<AkazeGpnpExtractor> akaze_extractor_;
    std::unique_ptr<AkazeGpnpExtractor> dual_akaze_extractor_;  ///< 双 ROI class 1 专用的 AKAZE 提取器
    std::unique_ptr<BinaryCornerExtractor> binary_extractor_;
    std::unique_ptr<TinyTargetExtractor> tiny_extractor_;

    /// 当前活跃的主提取器（指向上述三个提取器之一）。
    FeatureExtractor* extractor_ = nullptr;

    /// 当前活跃的回退链（按顺序指向上述三个提取器之一）。
    std::vector<FeatureExtractor*> fallback_extractors_;

    // 策略选择阈值（从配置读取）
    int akaze_min_area_ = 40000;
    int tiny_max_area_  = 800;

    // 单目模式配置
    MonoConfig mono_cfg_;

    // 非 AKAZE 提取器的 ROI 填充
    int binary_roi_pad_ = 0;
    int tiny_roi_pad_   = 0;

    // 双 ROI：在 AKAZE 模板上预计算的 BinaryCorner 角点
    bool dual_bc_template_ready_{false};
    std::vector<cv::Point2f> dual_bc_tmpl_corners_;  ///< N=10 个角点，模板像素坐标
    std::vector<Eigen::Vector3d> dual_bc_tmpl_pts3d_; ///< 每个角点的 3D 坐标

    std::string output_dir_;

    // 子系统模块
    InitialPnPSolver initial_pnp_;
    GPnPSolver gpnp_solver_;
    MadDisparityFilter mad_filter_;
    std::unique_ptr<Visualizer> visualizer_;

    TrackingState state_;

    // ========================================================================
    // ROI 辅助函数
    // ========================================================================

    static RoiRect validateRoi(const RoiRect* roi, const cv::Size& img_size,
                               const std::string& name);

    void offsetResultToOriginal(PipelineResult& result,
                                const cv::Point2d& left_offset,
                                const cv::Point2d& right_offset,
                                const cv::Mat& left_color_orig,
                                const cv::Mat& right_color_orig);

    // ========================================================================
    // 图像加载
    // ========================================================================

    static std::pair<cv::Mat, cv::Mat> loadImage(const cv::Mat& img);

    // ========================================================================
    // 降级辅助函数
    // ========================================================================

    /// 根据 ROI 面积配置活跃的提取策略链（由 process() 自动调用）。
    void configureStrategyChain(int roi_area);

    /// 使用非 AKAZE 提取器时，为角点上下文扩展 ROI 填充。
    void applyRoiPadding(RoiRect& rl, RoiRect& rr, int roi_area,
                         int left_cols, int left_rows,
                         int right_cols, int right_rows) const;

    /// 双 ROI 策略：class 0（边缘）使用 BinaryCorner + class 1（中心）使用 AKAZE。
    /// 合并两个提取器的角点，输入 GPNP 优化。
    PipelineResult processDualRoi(const cv::Mat& left_img,
                                   const cv::Mat& right_img,
                                   const RoiGroup& left_group,
                                   const RoiGroup& right_group,
                                   bool visualize);

    /// 一次性初始化：从 AKAZE 模板中提取 BinaryCorner 风格的角点。
    void prepareDualBcTemplate();

    /// 对预裁剪的 ROI 运行一次提取 + 坐标恢复。
    bool runExtraction(FeatureExtractor& ext,
                       const cv::Mat& left_gray, const cv::Mat& right_gray,
                       const cv::Mat& left_color, const cv::Mat& right_color,
                       const cv::Point2d& left_offset, const cv::Point2d& right_offset,
                       const cv::Mat& left_color_orig, const cv::Mat& right_color_orig,
                       PipelineResult& result);

    /// ROI 裁剪辅助函数：从双目图像对中提取 ROI。
    struct StereoRoi {
        cv::Mat left_gray, right_gray;
        cv::Mat left_color, right_color;
        cv::Point2d left_offset, right_offset;
    };
    StereoRoi cropStereoRoi(const cv::Mat& left_img, const cv::Mat& right_img,
                            const RoiRect* left_roi, const RoiRect* right_roi);

    // ========================================================================
    // PnP 分发 — 策略特定的位姿估计
    // ========================================================================

    std::pair<bool, PoseEstimate> dispatchPnP(FeatureExtractor* ext,
                                               PipelineResult& result, bool is_first);

    std::pair<bool, PoseEstimate> runAkazePnP(PipelineResult& result, bool is_first);
    std::pair<bool, PoseEstimate> runBinaryCornerPnP(PipelineResult& result, bool is_first);
    std::pair<bool, PoseEstimate> runTinyTargetPnP(PipelineResult& result);

    void finalizePose(PipelineResult& result, const PoseEstimate& pose);

    // ========================================================================
    // Logging
    // ========================================================================

    void addLogEntry(const PipelineResult& result, bool is_first, bool fallback_used);
};

} // namespace gpnp