#pragma once

/**
 * @file TrackerBase.hpp
 * @brief MonoTracker / StereoTracker 共享基础设施。
 */

#include "common/Config.hpp"
#include "common/LogConfig.hpp"
#include "common/Types.hpp"
#include "feature/FeatureExtractor.hpp"
#include "visualization/Visualizer.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace gpnp {

class AkazeGpnpExtractor;
class BinaryCornerExtractor;
class TinyTargetExtractor;

class TrackerBase {
public:
    virtual ~TrackerBase() = default;

    // ---- Logging & Output ----
    const std::vector<LogEntry>& getLogs() const { return state_.logs; }
    void printLogs() const;
    void setOutputDir(const std::string& dir) { output_dir_ = dir; }
    void setVerboseConsole(bool v) { verbose_console_ = v; g_verbose_console = v; }
    bool verboseConsole() const { return verbose_console_; }
    const StereoCameraParams& cameraParams() const { return camera_; }
    const TrackerConfig& config() const { return config_; }
    int frameCount() const { return state_.frame_count; }
    void clearCache() { state_ = TrackingState{}; }

protected:
    TrackerBase() = default;

    // ---- 提取器初始化（子类构造中调用）----
    void initExtractors(const std::string& template_path,
                        const BinaryCornerExtractor::Config& binary_cfg,
                        const std::string& binary_template_dir,
                        const TinyTargetExtractor::Config& tiny_cfg,
                        const std::string& tiny_template_dir);

    // ---- 共享方法 ----
    void configureStrategyChain(int roi_area);
    static RoiRect validateRoi(const RoiRect* roi, const cv::Size& img, const std::string& name);
    static std::pair<cv::Mat, cv::Mat> loadImage(const cv::Mat& img);
    void finalizePose(PipelineResult& result, const PoseEstimate& pose);
    void addLogEntry(const PipelineResult& result, bool is_first, bool fallback_used);

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
    int binary_roi_pad_ = 0, tiny_roi_pad_ = 0;
    std::string output_dir_;
    bool verbose_console_ = true;
    std::unique_ptr<Visualizer> visualizer_;
    TrackingState state_;
};

} // namespace gpnp
