#include "feature/AkazeGpnpExtractor.hpp"
#include "common/LogConfig.hpp"

#include <chrono>
#include <iostream>

namespace gpnp {

// ============================================================================
// 构造
// ============================================================================

AkazeGpnpExtractor::AkazeGpnpExtractor(double scale, const LKParams& lk_params)
    : scale_(scale)
    , akaze_extractor_(scale)
    , flow_tracker_(lk_params)
    , template_matcher_()
    , projector_(nullptr)
{
}

// ============================================================================
// 相机初始化
// ============================================================================

void AkazeGpnpExtractor::initCamera(const StereoCameraParams& camera_params) {
    projector_ = std::make_unique<StereoProjector>(camera_params);
}

// ============================================================================
// 模板数据
// ============================================================================

void AkazeGpnpExtractor::setTemplateData(const std::string& template_dir,
                                          double real_width_mm,
                                          double real_height_mm) {
    auto t0 = std::chrono::high_resolution_clock::now();
    // 使用独立的原始比例 AKAZE 实例进行模板提取
    AkazeExtractor full_akaze(1.0);
    template_ = full_akaze.extractTemplate(template_dir, real_width_mm, real_height_mm);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (g_verbose_console)
        std::cout << "[AkazeGpnp] Template features: "
                  << template_.keypoints.size() << " kp, "
                  << t_ms << " ms" << std::endl;
}

// ============================================================================
// 特征提取（从 StereoTracker::runPipeline 迁移而来）
// ============================================================================

PipelineResult AkazeGpnpExtractor::extract(const cv::Mat& left_gray,
                                            const cv::Mat& right_gray,
                                            const cv::Mat& left_color,
                                            const cv::Mat& right_color) {
    PipelineResult result;
    auto& timing = result.timing;

    // 步骤1: 对左图进行 AKAZE
    auto t0 = std::chrono::high_resolution_clock::now();
    FeatureSet features = akaze_extractor_.extract(left_gray);
    auto t1 = std::chrono::high_resolution_clock::now();
    timing["akaze"] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.kp_left = std::move(features.keypoints);
    result.desc_left = features.descriptors;
    result.n_kp_left = features.num_keypoints;

    // 步骤2: LK + FB（不含 MAD — 已移至 process() 层级）
    t0 = std::chrono::high_resolution_clock::now();
    TrackResult track = flow_tracker_.track(left_gray, right_gray, features.points);
    t1 = std::chrono::high_resolution_clock::now();
    timing["flow"] = std::chrono::duration<double, std::milli>(t1 - t0).count();

    result.pts_left_good = std::move(track.pts_left_good);
    result.pts_right_good = std::move(track.pts_right_good);
    result.dx_filtered = std::move(track.dx_filtered);
    result.idx_from_filtered = std::move(track.idx_from_filtered);
    result.disparity = std::move(track.disparity);

    // 步骤3: 立体投影（需要先调用 initCamera）
    t0 = std::chrono::high_resolution_clock::now();
    ProjectionResult proj;
    if (projector_) {
        proj = projector_->project(result.pts_left_good, result.pts_right_good);
    }
    t1 = std::chrono::high_resolution_clock::now();
    timing["proj"] = std::chrono::duration<double, std::milli>(t1 - t0).count();

    result.pts_right_projected = std::move(proj.pts_right_projected);
    result.pts_left_used = std::move(proj.pts_left_used);
    result.pts_right_used = std::move(proj.pts_right_used);
    result.valid_mask = std::move(proj.valid_mask);

    // 步骤4: 模板匹配
    t0 = std::chrono::high_resolution_clock::now();
    MatchResult match = template_matcher_.match(
        result.desc_left, result.kp_left,
        template_.descriptors, template_.keypoints);
    t1 = std::chrono::high_resolution_clock::now();
    timing["match_template"] = std::chrono::duration<double, std::milli>(t1 - t0).count();

    result.good_matches = std::move(match.good_matches);
    result.pts_left_match = std::move(match.pts_left_match);
    result.pts_template_match = std::move(match.pts_template_match);
    result.n_template_match = match.num_matches;
    result.left_color = left_color;
    result.right_color = right_color;

    return result;
}

// ============================================================================
// 单目特征提取 —— 仅 AKAZE + 模板匹配，无光流/立体投影
// ============================================================================

PipelineResult AkazeGpnpExtractor::extractMono(const cv::Mat& gray,
                                                const cv::Mat& color) {
    PipelineResult result;
    auto& timing = result.timing;

    // 步骤1: AKAZE 提取
    auto t0 = std::chrono::high_resolution_clock::now();
    FeatureSet features = akaze_extractor_.extract(gray);
    auto t1 = std::chrono::high_resolution_clock::now();
    timing["akaze"] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.kp_left = std::move(features.keypoints);
    result.desc_left = features.descriptors;
    result.n_kp_left = features.num_keypoints;

    // 步骤2: 模板匹配（单目模式仅做描述子匹配，无立体验证）
    t0 = std::chrono::high_resolution_clock::now();
    MatchResult match = template_matcher_.match(
        result.desc_left, result.kp_left,
        template_.descriptors, template_.keypoints);
    t1 = std::chrono::high_resolution_clock::now();
    timing["match_template"] = std::chrono::duration<double, std::milli>(t1 - t0).count();

    result.good_matches = std::move(match.good_matches);
    result.pts_left_match = std::move(match.pts_left_match);
    result.pts_template_match = std::move(match.pts_template_match);
    result.n_template_match = match.num_matches;
    result.left_color = color;
    result.n_matched = match.num_matches;
    result.success = (match.num_matches >= 4);

    return result;
}

} // namespace gpnp
