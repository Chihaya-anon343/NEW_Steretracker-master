#include "feature/TinyTargetExtractor.hpp"
#include "utils/PoseUtils.hpp"
#include "common/GeometryUtils.hpp"
#include "common/LogConfig.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace gpnp {

// ============================================================================
// 构造 —— 从 NewMuBan 加载24个角点模板
// ============================================================================

TinyTargetExtractor::TinyTargetExtractor(const Config& config,
                                           const std::string& template_dir)
    : config_(config)
{
    templates_ = loadTemplates(template_dir, false);
    if (!templates_.empty()) {
        if (g_verbose_console)
            std::cout << "[TinyTarget] Loaded " << templates_.size()
                      << " templates from " << template_dir << std::endl;
    } else {
        std::cerr << "[TinyTarget] WARNING: No templates loaded from "
                  << template_dir << std::endl;
    }

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
    if (!templates_.empty()) {
        if (g_verbose_console)
            std::cout << "[TinyTarget] Reloaded " << templates_.size()
                      << " templates" << std::endl;
    }
}

// ============================================================================
// 公共 extract() —— FeatureExtractor 接口
// ============================================================================

PipelineResult TinyTargetExtractor::extract(const cv::Mat& left_gray,
                                             const cv::Mat& right_gray,
                                             const cv::Mat& left_color,
                                             const cv::Mat& right_color) {
    PipelineResult result;
    result.left_color = left_color;
    result.right_color = right_color;

    if (left_gray.empty()) {
        std::cerr << "[TinyTarget] extract() called with empty left image." << std::endl;
        return result;
    }

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

    if (s_left != Status::Success || left_corners.size() != 4) {
        std::cerr << "[TinyTarget] Left extraction failed (status="
                  << static_cast<int>(s_left) << ", n=" << left_corners.size() << ")"
                  << std::endl;
        return result;
    }

    last_best_angle_ = best_angle;
    last_best_overlap_ = best_overlap;

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

    result.timing["tiny_target"] = 0.0;

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

    if (roi_gray.empty()) return Status::EmptyInput;

    last_call_debug_.roi_gray = roi_gray.clone();

    // ---- 1. 超分辨率（×scale_factor）+ Otsu ----
    int sf = config_.scale_factor;
    int new_w = roi_gray.cols * sf;
    int new_h = roi_gray.rows * sf;

    cv::Mat large;
    cv::resize(roi_gray, large, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);
    cv::GaussianBlur(large, large, cv::Size(3, 3), 0.3);

    cv::Mat binary;
    cv::threshold(large, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    last_call_debug_.otsu_binary = binary.clone();  // 清理前快照

    // ---- 2. BC 品质清理链（超分空间，3×3 核等效原尺度 <1px）----
    // 最大连通域（无触边排除——小目标常贴 ROI 边）→ 填洞 → CLOSE→OPEN
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8);
    if (num_labels <= 1) return Status::NoSuitableComponent;

    int best_label = 1;
    int best_area = stats.at<int>(1, cv::CC_STAT_AREA);
    for (int i = 2; i < num_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > best_area) { best_area = area; best_label = i; }
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
    last_call_debug_.super_binary = cleaned.clone();

    // ---- 3. 角度匹配：清理后的二值图（与角点提取共用同一张图）----
    if (!templates_.empty()) {
        auto match_res = matchTemplate(cleaned);
        best_angle = match_res.best_angle;
        best_overlap = match_res.best_overlap;
    }

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
    corners_large = orderPoints(corners_large);

    // ---- 5. 亚像素细化 ----
    std::vector<cv::Point2f> refined = refineCorners(large, corners_large, 5);

    // ---- 6. 角度对齐：旋至规范方向 ----
    if (best_angle >= 0) {
        int quadrant = static_cast<int>(std::round((360 - best_angle) / 90.0)) % 4;
        std::rotate(refined.begin(), refined.begin() + quadrant, refined.end());
    }

    // ---- 7. 缩放回原始 ROI 坐标 ----
    double inv_sf = 1.0 / sf;
    out_corners.resize(4);
    for (int i = 0; i < 4; ++i) {
        out_corners[i].x = static_cast<float>(refined[i].x * inv_sf);
        out_corners[i].y = static_cast<float>(refined[i].y * inv_sf);
    }

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

    cv::Mat component_mask = (labels == best_label);
    component_mask.convertTo(component_mask, CV_8UC1, 255.0);

    std::vector<cv::Point> pts;
    cv::findNonZero(component_mask, pts);
    if (pts.empty()) return result;

    // 裁剪最大分量周围的边界正方形
    int x_min = component_mask.cols, x_max = 0;
    int y_min = component_mask.rows, y_max = 0;
    for (const auto& pt : pts) {
        x_min = std::min(x_min, pt.x); x_max = std::max(x_max, pt.x);
        y_min = std::min(y_min, pt.y); y_max = std::max(y_max, pt.y);
    }

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
    PipelineResult result;
    result.left_color = color;

    if (gray.empty()) {
        std::cerr << "[TinyTarget::extractMono] empty image" << std::endl;
        return result;
    }

    if (g_verbose_console) std::cout << "[TinyTarget] Mono ROI=" << gray.cols << "x" << gray.rows << std::endl;

    // 单图提取4个角点
    std::vector<cv::Point2f> corners;
    int best_angle = -1;
    double best_overlap = 0.0;
    Status s = extract4Corners(gray, corners, best_angle, best_overlap);
    last_left_debug_ = last_call_debug_;

    if (s != Status::Success || corners.size() != 4) {
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

    result.timing["tiny_target"] = 0.0;

    result.n_matched = 4;
    result.success = true;
    if (g_verbose_console)
        std::cout << "[TinyTarget] Mono extracted 4 corners, angle=" << best_angle
                  << "°, overlap=" << best_overlap << std::endl;
    return result;
}

} // namespace gpnp
