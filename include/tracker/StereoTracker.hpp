#pragma once

#include "tracker/TrackerBase.hpp"
#include "feature/MadDisparityFilter.hpp"
#include "pose/GPnPSolver.hpp"
#include "pose/InitialPnPSolver.hpp"
#include "pose/MonoPnPSolver.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace gpnp {

class StereoTracker : public TrackerBase {
public:
    StereoTracker(const Eigen::Matrix3d& K,
                  const Eigen::Matrix3d& R_rl,
                  const Eigen::Vector3d& t_rl,
                  const std::string& template_path,
                  const TrackerConfig& config,
                  const BinaryCornerExtractor::Config& binary_cfg,
                  const std::string& binary_template_dir,
                  const TinyTargetExtractor::Config& tiny_cfg,
                  const std::string& tiny_template_dir);

    ~StereoTracker() override;

    StereoTracker(const StereoTracker&) = delete;
    StereoTracker& operator=(const StereoTracker&) = delete;

    // ---- Stereo API ----

    PipelineResult process(const cv::Mat& left_img,
                           const cv::Mat& right_img,
                           bool visualize = false,
                           const RoiGroup* left_group = nullptr,
                           const RoiGroup* right_group = nullptr);

    PipelineResult process(const std::string& left_path,
                           const std::string& right_path,
                           bool visualize = false,
                           const RoiGroup* left_group = nullptr,
                           const RoiGroup* right_group = nullptr);

    /// 单目降级处理：左右图仅一侧检测到目标时使用（EPnP，无立体）。
    PipelineResult processMono(const cv::Mat& img,
                               bool visualize = false,
                               const RoiGroup* roi_group = nullptr);

private:
    // ---- Stereo-specific helpers ----
    PipelineResult processDualRoi(const cv::Mat& left_img,
                                   const cv::Mat& right_img,
                                   const RoiGroup& left_group,
                                   const RoiGroup& right_group,
                                   bool visualize);
    void prepareDualBcTemplate();
    /// Dual-ROI 第 3 级退化: 合并与 BC-only 均失败后, 在 secondary ROI (class1) 上
    /// 依次尝试 BC → TT (class1 3D 尺寸, State 5 同机制; 全程冷启动)。
    /// 成功时 out_result 为链胜出结果 (全图坐标), out_pts3d 与
    /// out_result.pts_left_match 前缀 1:1 对应, out_strategy 为 "DualRoi_C1BC"/"DualRoi_C1TT"。
    std::pair<bool, PoseEstimate> runDualRoiClass1Chain(
        const cv::Mat& left_gray, const cv::Mat& right_gray,
        const cv::Mat& left_color, const cv::Mat& right_color,
        const RoiRect& left_sec, const RoiRect& right_sec,
        bool is_first,
        PipelineResult& out_result,
        std::vector<Eigen::Vector3d>& out_pts3d,
        std::string& out_strategy);
    void applyRoiPadding(RoiRect& rl, RoiRect& rr, int roi_area,
                         int left_cols, int left_rows,
                         int right_cols, int right_rows) const;
    bool runExtraction(FeatureExtractor& ext,
                       const cv::Mat& left_gray, const cv::Mat& right_gray,
                       const cv::Mat& left_color, const cv::Mat& right_color,
                       const cv::Point2d& left_offset, const cv::Point2d& right_offset,
                       const cv::Mat& left_color_orig, const cv::Mat& right_color_orig,
                       PipelineResult& result);
    void offsetResultToOriginal(PipelineResult& result,
                                const cv::Point2d& left_offset,
                                const cv::Point2d& right_offset,
                                const cv::Mat& left_color_orig,
                                const cv::Mat& right_color_orig);
    struct StereoRoi {
        cv::Mat left_gray, right_gray;
        cv::Mat left_color, right_color;
        cv::Point2d left_offset, right_offset;
    };
    StereoRoi cropStereoRoi(const cv::Mat& left_img, const cv::Mat& right_img,
                            const RoiRect* left_roi, const RoiRect* right_roi);

    // ---- PnP ----
    std::pair<bool, PoseEstimate> dispatchPnP(FeatureExtractor* ext,
                                               PipelineResult& result);
    // PnP 入口显式接收当前策略的 3D 模板 (ext->templateData().pts_3d)，
    // 不读成员 extractor_ —— 退化链中 ext 可不同于 extractor_，混用会错配 3D 索引
    std::pair<bool, PoseEstimate> runAkazePnP(
        PipelineResult& result,
        const std::vector<Eigen::Vector3d>& pnp_pts_3d);
    std::pair<bool, PoseEstimate> runBinaryCornerPnP(
        PipelineResult& result,
        const std::vector<Eigen::Vector3d>& pnp_pts_3d);
    std::pair<bool, PoseEstimate> solveBcPnpChain(
        PipelineResult& result,
        const std::vector<Eigen::Vector3d>& pnp_pts_3d);
    std::pair<bool, PoseEstimate> runTinyTargetPnP(
        PipelineResult& result,
        const std::vector<Eigen::Vector3d>& pnp_pts_3d);

    // ---- Stereo-specific members ----
    std::unique_ptr<AkazeGpnpExtractor> dual_akaze_extractor_;
    bool dual_bc_template_ready_{false};
    std::vector<cv::Point2f> dual_bc_tmpl_corners_;
    std::vector<Eigen::Vector3d> dual_bc_tmpl_pts3d_;
    InitialPnPSolver initial_pnp_;
    GPnPSolver gpnp_solver_;
    MadDisparityFilter mad_filter_;
    MonoPnPSolver mono_pnp_;  ///< 单侧检测降级时的单目 PnP
};

} // namespace gpnp
