#include "feature/TinyTargetExtractor.hpp"
#include "common/TemplateGeometry.hpp"
#include "utils/PoseUtils.hpp"
#include "common/GeometryUtils.hpp"
#include "common/LogConfig.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace gpnp {

namespace {
using Clock = std::chrono::steady_clock;
double elapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// atan2 winding gives four unique corners even at exact diamond orientations,
// unlike independently selecting extrema of x+y and x-y.
std::vector<cv::Point2f> uniqueWinding(std::vector<cv::Point2f> corners) {
    if (corners.size() != 4) return {};
    cv::Point2f center;
    for (const auto& p : corners) center += p * .25f;
    std::sort(corners.begin(), corners.end(), [&](const auto& a, const auto& b) {
        return std::atan2(a.y - center.y, a.x - center.x) <
               std::atan2(b.y - center.y, b.x - center.x);
    });
    auto first = std::min_element(corners.begin(), corners.end(), [](const auto& a, const auto& b) {
        const double da = a.x + a.y, db = b.x + b.y;
        if (std::abs(da - db) > 1e-4) return da < db;
        return a.y != b.y ? a.y < b.y : a.x < b.x;
    });
    std::rotate(corners.begin(), first, corners.end());
    return corners;
}

bool validQuad(const std::vector<cv::Point2f>& corners, double min_edge) {
    if (corners.size() != 4) return false;
    for (const auto& p : corners)
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
    double sign = 0.0;
    for (int i = 0; i < 4; ++i) {
        const auto a = corners[(i + 1) % 4] - corners[i];
        const auto b = corners[(i + 2) % 4] - corners[(i + 1) % 4];
        if (cv::norm(a) < min_edge) return false;
        const double cross = a.x * b.y - a.y * b.x;
        if (std::abs(cross) < 1e-6) return false;
        if (i == 0) sign = cross;
        else if (sign * cross <= 0.0) return false;
    }
    return std::abs(cv::contourArea(corners)) > min_edge * min_edge * .25;
}

cv::Mat normalizeQuad(const cv::Mat& mask, const std::vector<cv::Point2f>& corners,
                      int side) {
    if (!validQuad(corners, 1.0) || side < 8) return {};
    const float hi = static_cast<float>(side - 2);
    const std::vector<cv::Point2f> target{{1.f, 1.f}, {hi, 1.f}, {hi, hi}, {1.f, hi}};
    cv::Mat normalized;
    const cv::Mat H = cv::getPerspectiveTransform(corners, target);
    cv::warpPerspective(mask, normalized, H, cv::Size(side, side),
                        cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    return normalized;
}
} // namespace

// ============================================================================
// 构造 —— 从 NewMuBan 加载24个角点模板
// ============================================================================

TinyTargetExtractor::TinyTargetExtractor(const Config& config,
                                           const std::string& template_dir)
    : config_(config)
{
    validateTinyTargetRobustnessConfig(config_.robustness);
    if (config_.scale_factor < 1 || config_.scale_factor > 8 ||
        config_.target_size.width < 8 || config_.target_size.width > 512 ||
        config_.target_size.height < 8 || config_.target_size.height > 512)
        throw std::invalid_argument("Invalid TinyTarget scale_factor or target_size");
    templates_ = loadTemplates(template_dir, false);
    if (!templates_.empty()) {
        if (g_verbose_console)
            std::cout << "[TinyTarget] Loaded " << templates_.size()
                      << " templates from " << template_dir << std::endl;
    } else {
        std::cerr << "[TinyTarget] WARNING: No templates loaded from "
                  << template_dir << std::endl;
    }

    prepareCanonicalReference();

    // 一次性初始化 3D 模板点（基于 square_size_m）
    initPts3d();
}

// ============================================================================
// initPts3d — 一次性计算矩形目标的 4 个 3D 角点
// ============================================================================

void TinyTargetExtractor::initPts3d() {
    double sz = use_class1_ ? config_.square_size_m_class1
                            : config_.square_size_m_class0;
    // class1 fallback: 0 → use class0 value
    if (use_class1_ && sz <= 0.0)
        sz = config_.square_size_m_class0;
    if (sz <= 0.0) return;
    double half_mm = sz * 1000.0 / 2.0;  // m → mm
    template_data_.pts_3d = {
        {-half_mm, -half_mm, 0.0},  // 左上
        { half_mm, -half_mm, 0.0},  // 右上
        { half_mm,  half_mm, 0.0},  // 右下
        {-half_mm,  half_mm, 0.0},  // 左下
    };
    placeTemplateInClass0(template_data_.pts_3d, use_class1_, config_.template_geometry);

    if (g_verbose_console)
        std::cout << "[TinyTarget] 3D pts (mm): half=" << half_mm
                  << "  square_size_m=" << sz
                  << "  use_class1=" << (use_class1_ ? "true" : "false") << std::endl;
}

void TinyTargetExtractor::setUseClass1(bool v) {
    if (use_class1_ == v) return;
    use_class1_ = v;
    initPts3d();
}

// ============================================================================
// 模板数据
// ============================================================================

void TinyTargetExtractor::setTemplateData(const std::string& template_dir,
                                           double /*real_width_mm*/,
                                           double /*real_height_mm*/) {
    templates_ = loadTemplates(template_dir, false);
    prepareCanonicalReference();
    resetLastCall();
    if (!templates_.empty()) {
        if (g_verbose_console)
            std::cout << "[TinyTarget] Reloaded " << templates_.size()
                      << " templates" << std::endl;
    }
}

// ============================================================================
// 公共 extract() —— FeatureExtractor 接口
// ============================================================================

void TinyTargetExtractor::resetLastCall() {
    last_best_angle_ = -1;
    last_best_overlap_ = 0.0;
    last_quality_ = TinyTargetQuality{};
    last_call_debug_ = DebugImages{};
    last_left_debug_ = DebugImages{};
    last_right_debug_ = DebugImages{};
}

void TinyTargetExtractor::attachQuality(PipelineResult& result) const {
    result.tt_status = last_quality_.status;
    result.tt_legacy_iou = last_quality_.legacy_iou;
    result.tt_direction_iou = last_quality_.direction_iou;
    result.tt_direction_gap = last_quality_.direction_gap;
    result.tt_quadrant = last_quality_.quadrant;
    result.tt_refined_count = last_quality_.refined_count;
    result.tt_max_refine_shift_px = last_quality_.max_refine_shift_px;
    for (const auto& item : last_quality_.stage_ms)
        result.timing["tt_" + item.first] = item.second;
}

void TinyTargetExtractor::prepareCanonicalReference() {
    canonical_reference_.release();
    const auto it = std::find_if(templates_.begin(), templates_.end(),
                                [](const auto& t) { return t.angle == 0; });
    if (it == templates_.end() || it->image_bool.empty()) return;
    // Apply the same foreground semantics as the observation: retain the largest
    // white component, fill holes and close/open before coarse normalization.
    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(it->image_bool, labels, stats, centroids);
    if (count < 2) return;
    int best = 1;
    for (int i = 2; i < count; ++i)
        if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(best, cv::CC_STAT_AREA)) best = i;
    cv::Mat mask = labels == best;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return;
    mask.setTo(0);
    cv::drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);
    const auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return;
    const auto best_contour = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
    cv::Point2f box[4];
    cv::minAreaRect(*best_contour).points(box);
    const auto ordered = uniqueWinding(std::vector<cv::Point2f>(box, box + 4));
    canonical_reference_ = normalizeQuad(mask, ordered,
        std::max(config_.target_size.width, config_.target_size.height));
}

void TinyTargetExtractor::evaluateDirection(const cv::Mat& cleaned,
                                           const std::vector<cv::Point2f>& initial) {
    const auto start = Clock::now();
    if (!canonical_reference_.empty()) {
        const cv::Mat normalized = normalizeQuad(cleaned, initial, canonical_reference_.cols);
        if (!normalized.empty()) {
            int best = -1;
            double first = -1.0, second = -1.0;
            cv::Mat candidate = normalized;
            for (int q = 0; q < 4; ++q) {
                if (q > 0) cv::rotate(candidate, candidate, cv::ROTATE_90_COUNTERCLOCKWISE);
                const double score = calculateOverlap(candidate, canonical_reference_);
                last_quality_.direction_scores[q] = score;
                if (score > first) { second = first; first = score; best = q; }
                else if (score > second) second = score;
            }
            last_quality_.quadrant = best;
            last_quality_.direction_iou = std::max(0.0, first);
            last_quality_.direction_gap = std::max(0.0, first - second);
            last_quality_.direction_reliable = best >= 0 &&
                first >= config_.robustness.min_direction_iou &&
                first - second >= config_.robustness.min_direction_gap;
            if (debug_capture_) {
                last_quality_.normalized = normalized.clone();
                last_quality_.reference = canonical_reference_.clone();
            }
        }
    }
    last_quality_.stage_ms["direction"] = elapsedMs(start);
}

std::vector<cv::Point2f> TinyTargetExtractor::guardedRefine(
    const cv::Mat& large, const std::vector<cv::Point2f>& initial) {
    const auto start = Clock::now();
    const double sf = config_.scale_factor;
    double edge = std::numeric_limits<double>::max();
    for (int i = 0; i < 4; ++i)
        edge = std::min(edge, cv::norm(initial[(i + 1) % 4] - initial[i]) / sf);
    // Radius is expressed in original pixels. Do not span most of a tiny edge.
    const double radius = std::min(1.75, std::max(.75, .06 * edge));
    const int win = std::max(2, static_cast<int>(std::lround(radius * sf)));
    const int border = win + 2;
    cv::Mat padded;
    cv::copyMakeBorder(large, padded, border, border, border, border, cv::BORDER_REPLICATE);
    std::vector<cv::Point2f> result = initial;
    last_quality_.accepted_refinement.assign(4, 0);
    last_quality_.refined_count = 0;
    for (int i = 0; i < 4; ++i) {
        const auto p = initial[i];
        // Replicated padding is available to the solver but is not independent
        // evidence for a corner. A virtual or clipped corner can remain raw.
        if (p.x < win || p.y < win || p.x >= large.cols - win || p.y >= large.rows - win)
            continue;
        cv::Mat patch;
        cv::getRectSubPix(large, cv::Size(2 * win + 1, 2 * win + 1), p, patch);
        cv::Mat dx, dy;
        cv::Sobel(patch, dx, CV_32F, 1, 0, 3, 1.0 / 255.0);
        cv::Sobel(patch, dy, CV_32F, 0, 1, 3, 1.0 / 255.0);
        const double xx = cv::sum(dx.mul(dx))[0], yy = cv::sum(dy.mul(dy))[0];
        const double xy = cv::sum(dx.mul(dy))[0];
        const double trace = xx + yy;
        const double delta = std::sqrt(std::max(0.0, (xx - yy) * (xx - yy) + 4.0 * xy * xy));
        const double ratio = (trace - delta) / std::max(1e-12, trace + delta);
        if (trace / patch.total() < 1e-4 || ratio < config_.robustness.min_gradient_ratio)
            continue;
        std::vector<cv::Point2f> proposed{p + cv::Point2f(static_cast<float>(border), static_cast<float>(border))};
        cv::cornerSubPix(padded, proposed, cv::Size(win, win), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, .001));
        const cv::Point2f refined = proposed[0] - cv::Point2f(static_cast<float>(border), static_cast<float>(border));
        const double shift = cv::norm(refined - p) / sf;
        if (std::isfinite(shift)) last_quality_.max_proposed_refine_shift_px = std::max(last_quality_.max_proposed_refine_shift_px, shift);
        if (!std::isfinite(refined.x) || !std::isfinite(refined.y) ||
            shift > config_.robustness.max_refine_shift_px ||
            refined.x < 0 || refined.y < 0 || refined.x >= large.cols || refined.y >= large.rows)
            continue;
        result[i] = refined;
        last_quality_.max_refine_shift_px = std::max(last_quality_.max_refine_shift_px, shift);
        last_quality_.accepted_refinement[i] = 1;
        ++last_quality_.refined_count;
    }
    bool consistent = validQuad(result, sf);
    const double original_area = std::abs(cv::contourArea(initial));
    const double ratio = std::abs(cv::contourArea(result)) / std::max(1.0, original_area);
    consistent = consistent && ratio >= .65 && ratio <= 1.35;
    for (int i = 0; consistent && i < 4; ++i) {
        const double initial_edge = cv::norm(initial[(i + 1) % 4] - initial[i]);
        const double refined_edge = cv::norm(result[(i + 1) % 4] - result[i]);
        consistent = refined_edge >= .65 * initial_edge && refined_edge <= 1.35 * initial_edge;
    }
    if (!consistent) {
        result = initial;
        last_quality_.accepted_refinement.assign(4, 0);
        last_quality_.refined_count = 0;
        last_quality_.max_refine_shift_px = 0.0;
    }
    last_quality_.stage_ms["refinement"] = elapsedMs(start);
    return result;
}

PipelineResult TinyTargetExtractor::extract(const cv::Mat& left_gray,
                                             const cv::Mat& right_gray,
                                             const cv::Mat& left_color,
                                             const cv::Mat& right_color) {
    resetLastCall();
    // extract() is the legacy stereo entry even if an instance was previously
    // configured for monocular work. Restore the owner context on every return.
    const bool saved_context = mono_context_;
    struct RestoreContext { bool& target; bool value; ~RestoreContext() { target = value; } } restore{mono_context_, saved_context};
    mono_context_ = false;
    PipelineResult result;
    result.left_color = left_color;
    result.right_color = right_color;

    if (left_gray.empty()) {
        last_quality_.status = "empty_input";
        attachQuality(result);
        std::cerr << "[TinyTarget] extract() called with empty left image." << std::endl;
        return result;
    }

    auto t_extract_start = std::chrono::steady_clock::now();

    bool has_right = !right_gray.empty();
    if (g_verbose_console) {
        std::cout << "[TinyTarget] Left ROI=" << left_gray.cols << "x" << left_gray.rows;
        if (has_right) std::cout << " | Right ROI=" << right_gray.cols << "x" << right_gray.rows;
        std::cout << std::endl;
    }

    // ---- 步骤1: 从左图提取4个角点 ----
    std::vector<cv::Point2f> left_corners;
    int best_angle = -1;
    double best_overlap = 0.0;
    Status s_left = extract4Corners(left_gray, left_corners, best_angle, best_overlap);
    last_left_debug_ = last_call_debug_;
    attachQuality(result);

    if (s_left != Status::Success || left_corners.size() != 4) {
        std::cerr << "[TinyTarget] Left extraction failed (status="
                  << static_cast<int>(s_left) << ", n=" << left_corners.size() << ")"
                  << std::endl;
        return result;
    }

    last_best_angle_ = best_angle;
    last_best_overlap_ = best_overlap;
    const TinyTargetQuality left_quality = last_quality_;

    // ---- 步骤2: 从右图提取4个角点 ----
    std::vector<cv::Point2f> right_corners;
    if (has_right) {
        int ra = -1; double ro = 0.0;
        Status s_right = extract4Corners(right_gray, right_corners, ra, ro);
        last_right_debug_ = last_call_debug_;
        if (s_right != Status::Success || right_corners.size() != 4) {
            std::cerr << "[TinyTarget] Right extraction failed, stereo disabled." << std::endl;
            right_corners.clear();
        }
    }
    last_quality_ = left_quality;

    // ---- 步骤3: 按索引匹配左右（均为 TL→TR→BR→BL，各4个角点）----
    int n_stereo = (has_right && right_corners.size() == 4) ? 4 : 0;

    // ---- 步骤4: 填充 PipelineResult ----

    // 4a. kp_left
    result.kp_left.reserve(4);
    for (const auto& pt : left_corners)
        result.kp_left.emplace_back(pt, 1.0f);
    result.n_kp_left = 4;

    // 4b. 立体数据（视差占位 — 在 process() 中偏移后计算）
    if (n_stereo > 0) {
        result.pts_left_good = left_corners;
        result.pts_right_good = right_corners;
        result.disparity.resize(4, 0.0);
        result.dx_filtered.resize(4, 0.0);
        result.idx_from_filtered = {0, 1, 2, 3};
    }

    // 4c. 模板匹配数据（按索引一一对应）
    result.pts_left_match = left_corners;
    if (best_angle >= 0) {
        // 查找匹配模板用于可视化
        for (const auto& tmpl : templates_) {
            if (tmpl.angle == best_angle) {
                // 将模板角点缩放至 ROI 尺寸用于 pts_template_match
                double sx = static_cast<double>(left_gray.cols) / tmpl.image.cols;
                double sy = static_cast<double>(left_gray.rows) / tmpl.image.rows;
                result.pts_template_match = tmpl.corners;
                for (auto& pt : result.pts_template_match) {
                    pt.x *= static_cast<float>(sx);
                    pt.y *= static_cast<float>(sy);
                }
                result.n_template_match = 4;
                break;
            }
        }
    }

    // 为 GPNP 合成 DMatch：4个角点 ↔ 4个模板3D点（1:1）
    for (int i = 0; i < 4; ++i)
        result.good_matches.emplace_back(i, i, 0.0f);

    // 3D 物方点已在构造时由 initPts3d() 初始化。

    result.timing["tiny_target"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_extract_start).count();

    if (g_verbose_console)
        std::cout << "[TinyTarget] Extracted 4 corners (L), " << right_corners.size()
                  << " corners (R), stereo=" << n_stereo
                  << ", angle=" << best_angle << ", overlap=" << best_overlap << std::endl;

    return result;
}

// ============================================================================
// 核心：extract4Corners（从旧版 extractCorners + estimatePose 迁移而来）
// ============================================================================

Status TinyTargetExtractor::extract4Corners(const cv::Mat& roi_gray,
                                              std::vector<cv::Point2f>& out_corners,
                                              int& best_angle,
                                              double& best_overlap) {
    out_corners.clear();
    best_angle = -1;
    best_overlap = 0.0;
    last_call_debug_ = DebugImages{};
    last_quality_ = TinyTargetQuality{};
    last_quality_.status = "no_component";

    if (roi_gray.empty()) { last_quality_.status = "empty_input"; return Status::EmptyInput; }
    if (roi_gray.type() != CV_8UC1 || roi_gray.cols < 2 || roi_gray.rows < 2 ||
        roi_gray.cols > std::numeric_limits<int>::max() / config_.scale_factor ||
        roi_gray.rows > std::numeric_limits<int>::max() / config_.scale_factor) {
        last_quality_.status = "invalid_size";
        return Status::InvalidSize;
    }

    const bool robust = mono_context_ && config_.robustness.enabled;
    const bool diagnostics = mono_context_ && (config_.robustness.diagnostics || robust);
    const auto preprocessing_start = Clock::now();

    if (debug_capture_) last_call_debug_.roi_gray = roi_gray.clone();

    // ---- 1. 超分辨率（×scale_factor）+ Otsu ----
    int sf = config_.scale_factor;
    int new_w = roi_gray.cols * sf;
    int new_h = roi_gray.rows * sf;

    cv::Mat large;
    cv::resize(roi_gray, large, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);
    cv::GaussianBlur(large, large, cv::Size(3, 3), 0.3);

    cv::Mat binary;
    cv::threshold(large, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    if (debug_capture_) last_call_debug_.otsu_binary = binary.clone();  // 清理前快照

    // ---- 2. BC 品质清理链（超分空间，3×3 核等效原尺度 <1px）----
    // 最大连通域（BC 同款触边排除，全贴边时回退全局最大）→ 填洞 → CLOSE→OPEN
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8);
    if (num_labels <= 1) return Status::NoSuitableComponent;

    // ROI 边界处为背景残片，真实目标通常位于 ROI 内部（BC keepLargestRegion 同款）
    const int img_w = binary.cols;
    const int img_h = binary.rows;
    auto touchesBorder = [&](int i) {
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top  = stats.at<int>(i, cv::CC_STAT_TOP);
        int w    = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h    = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        return left <= 0 || top <= 0 || left + w >= img_w || top + h >= img_h;
    };

    int best_label = -1;
    int best_area = 0;
    int touch_label = -1;
    int touch_area = 0;
    for (int i = 1; i < num_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (touchesBorder(i)) {
            if (area > touch_area) { touch_area = area; touch_label = i; }
            continue;
        }
        if (area > best_area) { best_area = area; best_label = i; }
    }

    // 贴边覆盖率 (BC keepLargestRegion 同款): 真目标紧框下仅在外扩画幅的
    // 极值点少量贴边; 沿整条边界延伸的高覆盖域是背景, 不参与翻案
    auto borderCoverage = [&](int label) {
        long cnt = 0;
        for (int x = 0; x < img_w; ++x) {
            if (labels.at<int>(0, x) == label) ++cnt;
            if (labels.at<int>(img_h - 1, x) == label) ++cnt;
        }
        for (int y = 0; y < img_h; ++y) {
            if (labels.at<int>(y, 0) == label) ++cnt;
            if (labels.at<int>(y, img_w - 1) == label) ++cnt;
        }
        const long total = 2L * (img_w + img_h) - 4;
        return total > 0 ? static_cast<double>(cnt) / static_cast<double>(total) : 0.0;
    };

    // 面积占优规则 (BC keepLargestRegion 同款): 触边区域面积 ≥ 4× 最大内部区域
    // 且贴边覆盖率低 (非背景) 时选触边区域 —— YOLO 紧框下目标本体 bbox 顶到
    // ROI 边界即被判触边, 不该因此让内部小图案夺走选择权; 高贴边覆盖 (≥5%)
    // 的触边域是背景, 不参与翻案
    constexpr int kTouchDominanceRatio = 4;
    constexpr double kBgBorderCover = 0.05;
    if (best_label >= 0 && touch_label >= 0 &&
        touch_area >= kTouchDominanceRatio * best_area &&
        borderCoverage(touch_label) < kBgBorderCover) {
        best_label = touch_label;
        best_area = touch_area;
    }

    // 全部白域都贴边（小目标被 ROI 截断的常态）→ 回退全局最大，不劣于旧行为
    if (best_label < 0) {
        best_label = 1;
        best_area = stats.at<int>(1, cv::CC_STAT_AREA);
        for (int i = 2; i < num_labels; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > best_area) { best_area = area; best_label = i; }
        }
    }

    cv::Mat region = cv::Mat::zeros(binary.size(), CV_8UC1);
    region.setTo(255, labels == best_label);

    cv::Mat filled = cv::Mat::zeros(binary.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> ext_contours;
    cv::findContours(region, ext_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (ext_contours.empty()) return Status::NoSuitableComponent;
    cv::drawContours(filled, ext_contours, -1, cv::Scalar(255), cv::FILLED);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat cleaned;
    cv::morphologyEx(filled, cleaned, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);
    if (debug_capture_) last_call_debug_.super_binary = cleaned.clone();
    last_quality_.area_px = static_cast<double>(cv::countNonZero(cleaned)) / (sf * sf);
    last_quality_.stage_ms["preprocess"] = elapsedMs(preprocessing_start);

    // ---- 3. 角度匹配：清理后的二值图（与角点提取共用同一张图）----
    if (!templates_.empty()) {
        auto match_res = matchTemplate(cleaned);
        best_angle = match_res.best_angle;
        best_overlap = match_res.best_overlap;
    }
    last_quality_.legacy_iou = best_overlap;
    if (best_angle >= 0)
        last_quality_.legacy_quadrant = static_cast<int>(std::round((360 - best_angle) / 90.0)) % 4;

    // 角点提取输入：清理后的单域实心二值图
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cleaned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return Status::NoSuitableComponent;

    auto best_contour = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });

    // ---- 4. minAreaRect → 4个角点 → orderPoints ----
    cv::RotatedRect rect = cv::minAreaRect(*best_contour);
    cv::Point2f box_pts[4];
    rect.points(box_pts);
    std::vector<cv::Point2f> corners_large(box_pts, box_pts + 4);
    // Diagnostics use the unique winding, but must not alter legacy coordinates.
    const auto unique_corners = uniqueWinding(corners_large);
    corners_large = robust ? unique_corners : orderPoints(corners_large);

    const auto bbox = cv::boundingRect(*best_contour);
    last_quality_.span_x_px = (bbox.width - 1.0) / sf;
    last_quality_.span_y_px = (bbox.height - 1.0) / sf;
    last_quality_.span_px = std::min(last_quality_.span_x_px, last_quality_.span_y_px);
    last_quality_.geometry_valid = validQuad(unique_corners, sf) &&
        last_quality_.span_px >= config_.robustness.min_span_px;
    last_quality_.initial_corners.reserve(4);
    for (const auto& p : unique_corners)
        last_quality_.initial_corners.emplace_back(p.x / sf, p.y / sf);
    if (diagnostics) evaluateDirection(cleaned, unique_corners);
    if (robust && !last_quality_.geometry_valid) {
        last_quality_.status = "invalid_geometry";
        return Status::InsufficientFeatures;
    }
    if (robust && config_.robustness.direction_match) {
        if (canonical_reference_.empty()) {
            last_quality_.status = "no_templates";
            return Status::InsufficientFeatures;
        }
        if (last_quality_.direction_iou < config_.robustness.min_direction_iou) {
            last_quality_.status = "direction_low_iou";
            return Status::InsufficientFeatures;
        }
        if (!last_quality_.direction_reliable) {
            last_quality_.status = "direction_ambiguous";
            return Status::InsufficientFeatures;
        }
        // Template angle compatibility for existing visualization only. Physical
        // IDs are determined directly by the cyclic source index below.
        best_angle = (360 - 90 * last_quality_.quadrant) % 360;
        best_overlap = last_quality_.direction_iou;
    }

    // ---- 5. 亚像素细化 ----
    std::vector<cv::Point2f> refined;
    if (robust && config_.robustness.guarded_refinement) {
        refined = guardedRefine(large, corners_large);
    } else {
        const auto refinement_start = Clock::now();
        refined = refineCorners(large, corners_large, 5);
        last_quality_.accepted_refinement.assign(4, 1);
        last_quality_.refined_count = 4;
        for (int i = 0; i < 4; ++i)
            last_quality_.max_refine_shift_px = std::max(last_quality_.max_refine_shift_px,
                cv::norm(refined[i] - corners_large[i]) / sf);
        last_quality_.stage_ms["refinement"] = elapsedMs(refinement_start);
    }

    // ---- 6. 角度对齐：旋至规范方向 ----
    if (best_angle >= 0) {
        int quadrant = static_cast<int>(std::round((360 - best_angle) / 90.0)) % 4;
        std::rotate(refined.begin(), refined.begin() + quadrant, refined.end());
        std::rotate(corners_large.begin(), corners_large.begin() + quadrant, corners_large.end());
        if (last_quality_.accepted_refinement.size() == 4)
            std::rotate(last_quality_.accepted_refinement.begin(),
                        last_quality_.accepted_refinement.begin() + quadrant,
                        last_quality_.accepted_refinement.end());
    }

    // ---- 7. 缩放回原始 ROI 坐标 ----
    double inv_sf = 1.0 / sf;
    // cv::resize maps source pixel centers: dst=(src+.5)*sf-.5.
    // Keep the original direct division unless explicitly enabled for ablation.
    const double center_correction = robust && config_.robustness.pixel_center_mapping
        ? .5 * inv_sf - .5 : 0.0;
    out_corners.resize(4);
    last_quality_.initial_corners.resize(4);
    for (int i = 0; i < 4; ++i) {
        out_corners[i].x = static_cast<float>(refined[i].x * inv_sf + center_correction);
        out_corners[i].y = static_cast<float>(refined[i].y * inv_sf + center_correction);
        last_quality_.initial_corners[i] = cv::Point2f(
            static_cast<float>(corners_large[i].x * inv_sf + center_correction),
            static_cast<float>(corners_large[i].y * inv_sf + center_correction));
    }

    last_quality_.final_corners = out_corners;
    last_quality_.geometry_valid = last_quality_.geometry_valid && validQuad(out_corners, .5);
    last_quality_.status = "success";

    return Status::Success;
}

// ============================================================================
// 模板匹配（基于 IoU，与旧版相同）
// ============================================================================

TinyTargetExtractor::TemplateMatchResult
TinyTargetExtractor::matchTemplate(const cv::Mat& roi_binary) {
    TemplateMatchResult result;

    if (templates_.empty() || roi_binary.empty()) return result;

    // 提取最大连通分量
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(roi_binary, labels, stats, centroids, 8);
    if (num_labels <= 1) return result;

    int best_label = 1, best_area = stats.at<int>(1, cv::CC_STAT_AREA);
    for (int i = 2; i < num_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > best_area) { best_area = area; best_label = i; }
    }

    // The same selected component's exact pixel bounds are already available.
    // Keep inclusive maxima and the original crop/rounding behavior.
    const int x_min = stats.at<int>(best_label, cv::CC_STAT_LEFT);
    const int y_min = stats.at<int>(best_label, cv::CC_STAT_TOP);
    const int x_max = x_min + stats.at<int>(best_label, cv::CC_STAT_WIDTH) - 1;
    const int y_max = y_min + stats.at<int>(best_label, cv::CC_STAT_HEIGHT) - 1;

    int side = std::max(x_max - x_min + 1, y_max - y_min + 1);
    int cx = (x_min + x_max) / 2, cy = (y_min + y_max) / 2;
    int x1 = std::max(0, cx - side / 2);
    int y1 = std::max(0, cy - side / 2);
    int x2 = std::min(roi_binary.cols, x1 + side);
    int y2 = std::min(roi_binary.rows, y1 + side);

    cv::Mat square = roi_binary(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    cv::Mat roi_norm;
    cv::resize(square, roi_norm, config_.target_size, 0, 0, cv::INTER_NEAREST);

    // 计算与所有模板的重叠度，跟踪最佳
    for (size_t i = 0; i < templates_.size(); ++i) {
        double overlap = calculateOverlap(roi_norm, templates_[i].image_bool);
        result.all_overlaps.emplace_back(templates_[i].angle, overlap);
        if (overlap > result.best_overlap) {
            result.best_overlap = overlap;
            result.best_angle = templates_[i].angle;
        }
    }

    std::sort(result.all_overlaps.begin(), result.all_overlaps.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    return result;
}

// ============================================================================
// 亚像素角点细化（与旧版相同）
// ============================================================================

std::vector<cv::Point2f> TinyTargetExtractor::refineCorners(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& corners,
    int win_size) {

    if (corners.size() < 4) return corners;

    int border = win_size + 1;
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, border, border, border, border,
                        cv::BORDER_REPLICATE);

    std::vector<cv::Point2f> shifted(corners.size());
    for (size_t i = 0; i < corners.size(); ++i) {
        shifted[i].x = corners[i].x + border;
        shifted[i].y = corners[i].y + border;
    }

    cv::Size win(win_size, win_size);
    cv::Size zero_zone(-1, -1);
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                               50, 0.001);

    cv::cornerSubPix(padded, shifted, win, zero_zone, criteria);

    std::vector<cv::Point2f> result(corners.size());
    for (size_t i = 0; i < corners.size(); ++i) {
        result[i].x = shifted[i].x - border;
        result[i].y = shifted[i].y - border;
    }
    return result;
}

// ============================================================================
// 单目特征提取 —— 仅左图 4 角点提取（无立体匹配）
// ============================================================================

PipelineResult TinyTargetExtractor::extractMono(const cv::Mat& gray,
                                                 const cv::Mat& color) {
    resetLastCall();
    PipelineResult result;
    result.left_color = color;

    if (gray.empty()) {
        last_quality_.status = "empty_input";
        attachQuality(result);
        std::cerr << "[TinyTarget::extractMono] empty image" << std::endl;
        return result;
    }

    auto t_extract_start = std::chrono::steady_clock::now();

    if (g_verbose_console) std::cout << "[TinyTarget] Mono ROI=" << gray.cols << "x" << gray.rows << std::endl;

    // 单图提取4个角点
    std::vector<cv::Point2f> corners;
    int best_angle = -1;
    double best_overlap = 0.0;
    Status s = extract4Corners(gray, corners, best_angle, best_overlap);
    last_left_debug_ = last_call_debug_;
    attachQuality(result);

    if (s != Status::Success || corners.size() != 4) {
        if (mono_context_ && config_.robustness.enabled)
            result.failure_reason = "tt_" + last_quality_.status;
        std::cerr << "[TinyTarget::extractMono] extraction failed (status="
                  << static_cast<int>(s) << ", n=" << corners.size() << ")" << std::endl;
        return result;
    }

    last_best_angle_ = best_angle;
    last_best_overlap_ = best_overlap;

    // kp_left
    result.kp_left.reserve(4);
    for (const auto& pt : corners)
        result.kp_left.emplace_back(pt, 1.0f);
    result.n_kp_left = 4;

    // pts_left_match
    result.pts_left_match = corners;

    // 模板匹配数据
    if (best_angle >= 0) {
        for (const auto& tmpl : templates_) {
            if (tmpl.angle == best_angle) {
                double sx = static_cast<double>(gray.cols) / tmpl.image.cols;
                double sy = static_cast<double>(gray.rows) / tmpl.image.rows;
                result.pts_template_match = tmpl.corners;
                for (auto& pt : result.pts_template_match) {
                    pt.x *= static_cast<float>(sx);
                    pt.y *= static_cast<float>(sy);
                }
                result.n_template_match = 4;
                break;
            }
        }
    }

    // DMatch 1:1
    for (int i = 0; i < 4; ++i)
        result.good_matches.emplace_back(i, i, 0.0f);

    // 3D 物方点已在构造时由 initPts3d() 初始化。

    result.timing["tiny_target"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_extract_start).count();

    result.n_matched = 4;
    result.success = true;
    if (g_verbose_console)
        std::cout << "[TinyTarget] Mono extracted 4 corners, angle=" << best_angle
                  << "°, overlap=" << best_overlap << std::endl;
    return result;
}

} // namespace gpnp
