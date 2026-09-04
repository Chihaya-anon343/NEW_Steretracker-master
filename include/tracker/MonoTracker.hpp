#pragma once

/**
 * @file MonoTracker.hpp
 * @brief 单目视觉追踪器 —— 仅左图，EPnP 位姿解算。
 *
 * 继承 TrackerBase，不包含任何双目模块
 * （无 GPnPSolver / InitialPnPSolver / MadDisparityFilter / StereoProjector）。
 */

#include "tracker/TrackerBase.hpp"
#include "pose/MonoPnPSolver.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace gpnp {

class MonoTracker : public TrackerBase {
public:
    MonoTracker(const Eigen::Matrix3d& K,
                const std::string& template_path,
                const TrackerConfig& config,
                const BinaryCornerExtractor::Config& binary_cfg,
                const std::string& binary_template_dir,
                const TinyTargetExtractor::Config& tiny_cfg,
                const std::string& tiny_template_dir);

    /// 单目帧处理（仅左图）：ROI → 策略分发 → 单图特征提取 → EPnP 解算。
    PipelineResult process(const cv::Mat& left_img,
                           bool visualize = false,
                           const RoiGroup* left_group = nullptr);

private:
    PipelineResult processDualRoi(const cv::Mat& left_img,
                                   const RoiGroup& left_group,
                                   bool visualize);
    void prepareDualBcTemplate();
    /// Dual-ROI 第 3 级退化: 合并与 BC-only 均失败后, 在 secondary ROI (class1) 上
    /// 依次尝试 BC → TT (class1 3D 尺寸, State 5 同机制; MonoPnP 冷启动)。
    /// 成功时 out_result 已截断为 2D/3D 严格 1:1 的配对子集 (全图坐标)，
    /// out_strategy 为 "DualRoi_C1BC"/"DualRoi_C1TT"。
    std::pair<bool, PoseEstimate> runDualRoiClass1Chain(
        const cv::Mat& left_gray, const cv::Mat& left_color,
        const RoiRect& left_sec,
        bool is_first,
        PipelineResult& out_result,
        std::vector<Eigen::Vector3d>& out_pts3d,
        std::string& out_strategy);

    MonoPnPSolver mono_pnp_;

    std::unique_ptr<AkazeGpnpExtractor> dual_akaze_extractor_;
    bool dual_bc_template_ready_{false};
    std::vector<cv::Point2f> dual_bc_tmpl_corners_;
    std::vector<Eigen::Vector3d> dual_bc_tmpl_pts3d_;
};

} // namespace gpnp
