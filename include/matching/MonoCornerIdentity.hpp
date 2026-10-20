#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace gpnp {
struct MonoCornerIdentityConfig {
    bool enabled{false};
    bool class1_enabled{false}; // opt-in: class1 contour quality/scale switching needs separate validation
    std::string method{"hybrid"}; // tt_fourway / projective / hybrid
    std::string frontend{"legacy"}; // legacy extraction + identity / normalize before extraction
    bool refine_original{true};
    int normalized_size{128};
    double min_span_px{16.0};
    double min_roi_span_ratio{0.25};
    double min_iou{0.75};
    double min_score_gap{0.05};
    double max_fit_error_px{4.0};
    double max_fit_error_ratio{0.03};
    bool quality_filter{false};
    bool partial_matching{false};
    int min_reliable_points{8};
    int max_partial_hypotheses{16};
    double min_model_coverage{0.60};
};
MonoCornerIdentityConfig readMonoCornerIdentityConfig(const cv::FileNode& node);
void validateMonoCornerIdentityConfig(const MonoCornerIdentityConfig& config);

struct MonoCornerIdentityResult {
    bool accepted{false};
    std::string status{"disabled"}, path;
    std::vector<int> indices; // canonical ID -> original observed-point index
    std::vector<int> template_ids; // parallel to indices; may be sparse
    std::vector<cv::Point2f> rejected_points;
    std::vector<int> rejected_ids;
    std::vector<double> residuals_px;
    int detected_count{0};
    double fit_error_px{0.0}, iou{0.0}, score_gap{0.0}, coarse_iou{0.0};
    double elapsed_ms{0.0};
    int hypotheses{0}, normalized_tests{0};
    int refined_points{0};
    double normalization_ms{0}, corners_ms{0}, refinement_ms{0}, validation_ms{0};
    std::vector<cv::Point2f> points, before_refinement;
    // Capture-only external contour, reused by the extractor's debug display.
    std::vector<cv::Point> source_contour;
    cv::Mat foreground;
    cv::Mat image_to_template, normalized, coarse_normalized, reference;
};

// Per-extractor immutable model cache; no pose seed or dataset ground truth.
class MonoCornerIdentity {
public:
    explicit MonoCornerIdentity(const MonoCornerIdentityConfig& config = {});
    void setModel(const std::vector<cv::Point2f>& canonical);
    MonoCornerIdentityResult match(const std::vector<cv::Point2f>& observed,
                                  const cv::Mat& foreground, bool capture = false) const;
    // Cleaned foreground in original ROI coordinates. Never reads pose/GT or
    // invokes the old 24-angle extractor. Returned points retain original units.
    MonoCornerIdentityResult extractNormalized(const cv::Mat& foreground,
                                               bool capture = false) const;
private:
    MonoCornerIdentityResult matchImpl(const std::vector<cv::Point2f>& observed,
        const cv::Mat& foreground, bool capture, const cv::Mat& coarse_transform,
        const cv::Mat& coarse_image, double coarse_iou, double coarse_gap) const;
    MonoCornerIdentityConfig config_;
    std::vector<cv::Point2f> model_;
    cv::Mat reference_;
};
} // namespace gpnp
