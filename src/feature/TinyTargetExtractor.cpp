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

    // 4d. 生成3D物体点（矩形目标，z=0）— 单位为毫米
    {
        double half_mm = config_.square_size_m * 1000.0 / 2.0;  // m → mm
        template_data_.pts_3d = {
            {-half_mm, -half_mm, 0.0},  // 左上
            { half_mm, -half_mm, 0.0},  // 右上
            { half_mm,  half_mm, 0.0},  // 右下
            {-half_mm,  half_mm, 0.0},  // 左下
        };
    }

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

    if (roi_gray.empty()) return Status::EmptyInput;

    // ---- 1. Otsu + 模板匹配 → 最佳角度 ----
    cv::Mat roi_binary;
    cv::threshold(roi_gray, roi_binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    if (!templates_.empty()) {
        auto match_res = matchTemplate(roi_binary);
        best_angle = match_res.best_angle;
        best_overlap = match_res.best_overlap;
    }

    // ---- 2. 超分辨率（×scale_factor）----
    int sf = config_.scale_factor;
    int new_w = roi_gray.cols * sf;
    int new_h = roi_gray.rows * sf;

    cv::Mat large;
    cv::resize(roi_gray, large, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);
    cv::GaussianBlur(large, large, cv::Size(3, 3), 0.3);

    cv::Mat binary;
    cv::threshold(large, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    // 形态学清理
    cv::Mat k_open  = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat k_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, k_open);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, k_close);

    // ---- 3. 连通分量分析 + 评分 ----
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8);

    int best_label = selectBestComponent(binary, labels, num_labels, stats, centroids);
    if (best_label < 0) return Status::NoSuitableComponent;

    // 最佳分量的掩膜
    cv::Mat best_mask = (labels == best_label);
    best_mask.convertTo(best_mask, CV_8UC1, 255.0);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(best_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
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
// 分量选择（评分系统，与旧版相同）
// ============================================================================

int TinyTargetExtractor::selectBestComponent(
    const cv::Mat& binary, const cv::Mat& label_map,
    int num_labels, const cv::Mat& stats, const cv::Mat& centroids) {

    double roi_cx = binary.cols / 2.0;
    double roi_cy = binary.rows / 2.0;
    double total_area = static_cast<double>(binary.cols) * binary.rows;
    double max_dist = std::sqrt(roi_cx * roi_cx + roi_cy * roi_cy);

    int best_label = -1;
    double best_score = -1.0;

    for (int i = 1; i < num_labels; ++i) {
        int x    = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y    = stats.at<int>(i, cv::CC_STAT_TOP);
        int bw   = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int bh   = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area < 200) continue;

        double bbox_area  = static_cast<double>(bw) * bh;
        double rect_ratio = area / bbox_area;
        double aspect     = std::max(bw, bh) / (std::min(bw, bh) + 1e-6);

        double comp_cx = centroids.at<double>(i, 0);
        double comp_cy = centroids.at<double>(i, 1);
        double dist = std::sqrt((comp_cx - roi_cx) * (comp_cx - roi_cx) +
                                 (comp_cy - roi_cy) * (comp_cy - roi_cy));
        double center_score = 1.0 - dist / max_dist;

        double area_ratio = area / total_area;
        double area_score = (area_ratio >= 0.15 && area_ratio <= 0.6)
            ? 1.0 : std::max(0.0, 1.0 - std::abs(area_ratio - 0.35) * 3.0);

        double aspect_score = 1.0 / aspect;

        double score = rect_ratio * 0.25 + area_score * 0.3 +
                       center_score * 0.3 + aspect_score * 0.15;

        if (score > best_score) {
            best_score = score;
            best_label = i;
        }
    }

    return best_label;
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

    // 3D 物点（mm）
    {
        double half_mm = config_.square_size_m * 1000.0 / 2.0;
        template_data_.pts_3d = {
            {-half_mm, -half_mm, 0.0},
            { half_mm, -half_mm, 0.0},
            { half_mm,  half_mm, 0.0},
            {-half_mm,  half_mm, 0.0},
        };
    }

    result.timing["tiny_target"] = 0.0;

    result.n_matched = 4;
    result.success = true;
    if (g_verbose_console)
        std::cout << "[TinyTarget] Mono extracted 4 corners, angle=" << best_angle
                  << "°, overlap=" << best_overlap << std::endl;
    return result;
}

} // namespace gpnp
