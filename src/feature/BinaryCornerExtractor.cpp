#include "feature/BinaryCornerExtractor.hpp"
#include "utils/PoseUtils.hpp"
#include "common/GeometryUtils.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace gpnp {

namespace {

// ============================================================================
// rotate_and_clean_image —— 自由函数（非成员，与旧版保持一致）
//
// 将二值图像旋转 `angle` 度，然后进行形态学开+闭+阈值操作
// 以清理旋转造成的边缘模糊。
// 返回 (旋转+清理后的图像, 原始中心点, 旋转后中心点)。
// ============================================================================
constexpr double CPSAGL = 20.0;  // 角点重排序角度容差

std::tuple<cv::Mat, cv::Point2f, cv::Point2f>
rotate_and_clean_image(const cv::Mat& binary_image, double angle) {
    int h = binary_image.rows;
    int w = binary_image.cols;

    // 计算旋转后的图像尺寸
    double rad = std::abs(angle) * CV_PI / 180.0;
    double cos_a = std::abs(std::cos(rad));
    double sin_a = std::abs(std::sin(rad));
    int new_w = static_cast<int>(h * sin_a + w * cos_a);
    int new_h = static_cast<int>(h * cos_a + w * sin_a);

    // 绕图像中心构建旋转矩阵
    cv::Point2f center(w / 2.0f, h / 2.0f);
    cv::Mat M = cv::getRotationMatrix2D(center, -angle, 1.0);

    // 调整平移量使旋转后的图像居中
    M.at<double>(0, 2) += (new_w - w) / 2.0;
    M.at<double>(1, 2) += (new_h - h) / 2.0;

    cv::Mat rotated;
    cv::warpAffine(binary_image, rotated, M, cv::Size(new_w, new_h),
                   cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::Point2f rotation_center_original = center;
    cv::Point2f rotation_center_rotated(new_w / 2.0f, new_h / 2.0f);

    // 清理旋转造成的模糊边缘
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat opened;
    cv::morphologyEx(rotated, opened, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);

    cv::Mat closed;
    cv::morphologyEx(opened, closed, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

    cv::Mat cleaned;
    cv::threshold(closed, cleaned, 50, 255, cv::THRESH_BINARY);

    return {cleaned, rotation_center_original, rotation_center_rotated};
}

} // anonymous namespace

// ============================================================================
// 构造 —— 加载模板，初始化形态学核
// ============================================================================

BinaryCornerExtractor::BinaryCornerExtractor(const Config& config,
                                               const std::string& template_dir)
    : config_(config)
{
    // 强制使用奇数核尺寸
    if (config_.kernel_size % 2 == 0) {
        config_.kernel_size += 1;
    }
    kernel_ = cv::getStructuringElement(cv::MORPH_RECT,
                                         cv::Size(config_.kernel_size, config_.kernel_size));

    // 从 NewMuBan 加载角点模板
    templates_ = loadTemplates(template_dir, false);
    if (!templates_.empty()) {
        std::cout << "[BinaryCorner] Loaded " << templates_.size()
                  << " templates from " << template_dir << std::endl;
    } else {
        std::cerr << "[BinaryCorner] WARNING: No templates loaded from "
                  << template_dir << std::endl;
    }
}

// ============================================================================
// 模板数据加载
// ============================================================================

void BinaryCornerExtractor::setTemplateData(const std::string& template_dir,
                                             double /*real_width_mm*/,
                                             double /*real_height_mm*/) {
    templates_ = loadTemplates(template_dir, false);
    if (!templates_.empty()) {
        std::cout << "[BinaryCorner] Reloaded " << templates_.size()
                  << " templates" << std::endl;
    }
}

// ============================================================================
// 主入口 —— FeatureExtractor 接口：灰度ROI → Otsu二值化 → 角点提取 → PipelineResult
// ============================================================================

PipelineResult BinaryCornerExtractor::extract(const cv::Mat& left_gray,
                                               const cv::Mat& right_gray,
                                               const cv::Mat& left_color,
                                               const cv::Mat& right_color) {
    PipelineResult result;
    result.left_color = left_color;
    result.right_color = right_color;

    // ---- 输入校验 ----
    if (left_gray.empty()) {
        std::cerr << "[BinaryCorner] extract() called with empty left image." << std::endl;
        return result;
    }

    // ---- 第1步: Otsu 二值化（两幅图像） ----
    cv::Mat left_binary, right_binary;
    cv::threshold(left_gray, left_binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    last_left_binary_ = left_binary.clone();   // 保存用于可视化

    bool has_right = !right_gray.empty();
    if (has_right) {
        cv::threshold(right_gray, right_binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        last_right_binary_ = right_binary.clone();
    }

    std::cout << "[BinaryCorner] Left ROI=" << left_gray.cols << "x" << left_gray.rows
              << ", white=" << cv::countNonZero(left_binary);
    if (has_right) {
        std::cout << " | Right ROI=" << right_gray.cols << "x" << right_gray.rows
                  << ", white=" << cv::countNonZero(right_binary);
    }
    std::cout << std::endl;

    // ---- 第2步: 从左图提取角点 ----
    std::vector<cv::Point2f> left_corners;
    Status s_left = extractFromBinary(left_binary, left_gray, left_corners);
    if (s_left != Status::Success || left_corners.empty()) {
        std::cerr << "[BinaryCorner] Left extraction failed (status="
                  << static_cast<int>(s_left) << ")" << std::endl;
        return result;
    }

    // 保存左图的模板匹配结果（右图提取可能会覆盖）
    const TemplateData* matched_tmpl = last_matched_template_;
    double matched_overlap = last_match_overlap_;

    // ---- 第3步: 从右图提取角点（复用左图模板） ----
    std::vector<cv::Point2f> right_corners;
    if (has_right) {
        // 右目复用左目的模板，避免独立 findBestMatch 导致左右匹配不同模板
        Status s_right = extractFromBinary(right_binary, right_gray, right_corners,
                                           matched_tmpl);
        if (s_right != Status::Success || right_corners.empty()) {
            std::cerr << "[BinaryCorner] Right extraction failed (status="
                      << static_cast<int>(s_right) << "), stereo disabled." << std::endl;
            right_corners.clear();
        }
    }

    // 恢复左图的模板匹配结果（右图的 extractFromBinary 可能会覆盖）
    last_matched_template_ = matched_tmpl;
    last_match_overlap_ = matched_overlap;

    // ---- 第4步: 按索引匹配左右角点（1:1 映射，同形状，同顺序） ----
    int n_stereo = std::min(static_cast<int>(left_corners.size()),
                             static_cast<int>(right_corners.size()));
    if (n_stereo > 0 && !has_right) n_stereo = 0;  // no right image → no stereo

    // ---- 第5步: 填充 PipelineResult ----

    // 5a. kp_left: 角点转为 KeyPoints
    result.kp_left.reserve(left_corners.size());
    for (const auto& pt : left_corners) {
        result.kp_left.emplace_back(pt, 1.0f);
    }
    result.n_kp_left = static_cast<int>(left_corners.size());

    // 5b. 双目数据: 角点存为 pts_left_good/pts_right_good（ROI 局部坐标）。
    //     视差在 StereoTracker::process() 中 offsetResultToOriginal 之后计算，
    //     此时坐标为全图空间（左右 ROI 可能有不同的偏移量）。
    if (n_stereo > 0) {
        result.pts_left_good.resize(n_stereo);
        result.pts_right_good.resize(n_stereo);
        result.dx_filtered.resize(n_stereo);
        result.disparity.resize(n_stereo);
        result.idx_from_filtered.resize(n_stereo);

        for (int i = 0; i < n_stereo; ++i) {
            result.pts_left_good[i]  = left_corners[i];
            result.pts_right_good[i] = right_corners[i];
            // 占位 —— 真实视差在 process() 中偏移后计算
            result.dx_filtered[i] = 0.0;
            result.disparity[i] = 0.0;
            result.idx_from_filtered[i] = i;
        }
    }

    // 5c. 模板匹配数据（用于 GPNP: good_matches → queryIdx↔trainIdx 映射）
    result.pts_left_match = left_corners;

    // 查找 0° 模板（始终以 0° 作为 3D 参考，无论匹配到哪个角度）
    const TemplateData* tmpl_0 = nullptr;
    for (const auto& t : templates_) {
        if (t.angle == 0) {
            tmpl_0 = &t;
            break;
        }
    }
    // 若无可用的 0° 模板，回退到匹配到的模板
    if (tmpl_0 == nullptr) tmpl_0 = matched_tmpl;

    if (tmpl_0 != nullptr) {
        // 将模板角点缩放到左图 ROI 尺寸
        double scale_x = static_cast<double>(left_gray.cols)
                       / tmpl_0->image.cols;
        double scale_y = static_cast<double>(left_gray.rows)
                       / tmpl_0->image.rows;

        result.pts_template_match = tmpl_0->corners;
        for (auto& pt : result.pts_template_match) {
            pt.x *= static_cast<float>(scale_x);
            pt.y *= static_cast<float>(scale_y);
        }
        result.n_template_match = static_cast<int>(result.pts_template_match.size());

        // 合成 DMatch: queryIdx=i（左图关键点索引）, trainIdx=i（模板角点索引）
        // GPNP 使用 train_lut[queryIdx]=trainIdx 将双目点映射到 3D 点。
        // 1:1 对应关系下: train_lut[i] = i。
        int n_match = std::min(static_cast<int>(left_corners.size()),
                               static_cast<int>(tmpl_0->corners.size()));
        result.good_matches.reserve(n_match);
        for (int i = 0; i < n_match; ++i) {
            result.good_matches.emplace_back(i, i, 0.0f);  // queryIdx, trainIdx, distance=0
        }
    }

    // 5d. 生成 3D 物方点（平面靶标，z=0）—— 单位为毫米（GPNP 单位制）。
    //     始终使用 0° 模板角点，使得 PnP 可解算出有意义的旋转。
    if (tmpl_0 != nullptr && config_.pixel_to_meter_scale > 0.0) {
        double s_mm_per_px = config_.pixel_to_meter_scale * 1000.0;  // m/px → mm/px
        const auto& tc = tmpl_0->corners;
        template_data_.pts_3d.clear();
        template_data_.pts_3d.reserve(tc.size());
        for (const auto& pt : tc) {
            template_data_.pts_3d.emplace_back(pt.x * s_mm_per_px,
                                               pt.y * s_mm_per_px, 0.0);
        }
        std::cout << "[BinaryCorner] 3D pts (mm): " << template_data_.pts_3d.size()
                  << " (angle=0, scale=" << s_mm_per_px << " mm/px)" << std::endl;
    }

    // desc_left 保持为空（无 AKAZE 描述子）
    // pts_left_used/pts_right_used/pts_right_projected 保持为空
    //   （GPNP 直接通过 idx_from_filtered 使用 pts_left_good/pts_right_good）

    result.timing["binary_corner"] = 0.0;

    std::cout << "[BinaryCorner] Extracted " << left_corners.size()
              << " corners (L), " << right_corners.size() << " corners (R)"
              << ", stereo=" << n_stereo
              << ", angle=" << (matched_tmpl ? std::to_string(matched_tmpl->angle) : "none")
              << ", overlap=" << matched_overlap
              << ", template_corners=" << (matched_tmpl ? static_cast<int>(matched_tmpl->corners.size()) : 0)
              << ", 3d_pts=" << template_data_.pts_3d.size()
              << std::endl;

    return result;
}

// ============================================================================
// 内部流水线 —— 7步二值图像角点提取
// ============================================================================

Status BinaryCornerExtractor::extractFromBinary(const cv::Mat& binary_img,
                                                 const cv::Mat& gray_roi,
                                                 std::vector<cv::Point2f>& out_corners,
                                                 const TemplateData* preset_template) {
    process_log_.clear();
    last_matched_template_ = nullptr;
    last_match_overlap_ = 0.0;

    if (binary_img.empty()) {
        logStep("Input", "Empty image");
        return Status::EmptyInput;
    }

    logStep("Input",
            "Size: " + std::to_string(binary_img.cols) + "x" + std::to_string(binary_img.rows));

    // ---- 预处理：确保单通道二值图像 ----
    cv::Mat work_img;
    if (binary_img.channels() == 3) {
        cv::cvtColor(binary_img, work_img, cv::COLOR_BGR2GRAY);
    } else if (binary_img.channels() == 1) {
        work_img = binary_img.clone();
        if (work_img.type() != CV_8UC1) {
            work_img.convertTo(work_img, CV_8UC1);
        }
    } else {
        logStep("Preprocess", "Invalid channel count");
        return Status::InvalidSize;
    }

    // 确保二值化 (0/255)
    if (cv::countNonZero(work_img != 0) > 0 &&
        cv::countNonZero(work_img != 255) > 0) {
        cv::threshold(work_img, work_img, 127, 255, cv::THRESH_BINARY);
    }

    // ---- 第1步: 保留最大连通域 ----
    cv::Mat largest_region = keepLargestRegion(work_img);

    // ---- 第2步: 填充孔洞 ----
    cv::Mat filled = fillHoles(largest_region);

    // ---- 第3步: 形态学平滑 ----
    cv::Mat smoothed = smoothBoundary(filled);

    // ---- 第4步: 模板匹配 ----
    if (preset_template != nullptr) {
        // 使用预设模板（来自另一目的匹配结果），跳过独立的 findBestMatch
        last_matched_template_ = preset_template;
        last_match_overlap_ = -1.0;  // 标记为非独立匹配
        logStep("FindBestMatch",
                "Preset: angle=" + std::to_string(preset_template->angle) +
                " deg (reused from other eye)");
    } else if (!templates_.empty() && config_.target_size.width > 0) {
        cv::Mat resized_binary;
        cv::resize(smoothed, resized_binary, config_.target_size, 0, 0, cv::INTER_NEAREST);

        BestMatch best = findBestMatch(resized_binary);

        if (best.template_index >= 0) {
            last_match_overlap_ = best.overlap;
            last_matched_template_ = &templates_[best.template_index];
        }
    }

    // ---- Step 4.5: 旋转回正 ----
    // 优先用灰度图旋转(INTER_CUBIC)再Otsu → 边缘平滑无锯齿
    // 无灰度图时回退到二值图旋转(INTER_NEAREST) + 形态学清理
    cv::Mat cleaned;
    cv::Point2f center_orig, center_rot;
    double rot_angle = 0.0;
    bool did_rotate = false;

    if (last_matched_template_ != nullptr) {
        rot_angle = static_cast<double>(last_matched_template_->angle);

        if (!gray_roi.empty()) {
            // ★ 新方法：旋转灰度图 → Otsu → 干净的正位二值图
            int h = gray_roi.rows, w = gray_roi.cols;
            double rad = std::abs(rot_angle) * CV_PI / 180.0;
            double cos_a = std::abs(std::cos(rad));
            double sin_a = std::abs(std::sin(rad));
            int new_w = static_cast<int>(h * sin_a + w * cos_a);
            int new_h = static_cast<int>(h * cos_a + w * sin_a);

            center_orig = cv::Point2f(w / 2.0f, h / 2.0f);
            cv::Mat M = cv::getRotationMatrix2D(center_orig, -rot_angle, 1.0);
            M.at<double>(0, 2) += (new_w - w) / 2.0;
            M.at<double>(1, 2) += (new_h - h) / 2.0;
            center_rot = cv::Point2f(new_w / 2.0f, new_h / 2.0f);

            cv::Mat gray_rotated;
            cv::warpAffine(gray_roi, gray_rotated, M, cv::Size(new_w, new_h),
                           cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0));
            // Otsu 自动阈值 × 系数（>1 提高阈值，减少背景被误判为白色）
            double otsu_val = cv::threshold(gray_rotated, cleaned, 0, 255,
                                             cv::THRESH_BINARY + cv::THRESH_OTSU);
            double adjusted = otsu_val * config_.otsu_ratio;
            cv::threshold(gray_rotated, cleaned, adjusted, 255, cv::THRESH_BINARY);
            // 过滤：从中心蔓延找目标 → 填洞 → 平滑
            cleaned = keepRegionFromCenter(cleaned);
            cleaned = fillHoles(cleaned);
            cleaned = smoothBoundary(cleaned);
            did_rotate = true;
        } else {
            // 回退：二值图旋转 + 形态学清理（旧方法）
            auto [cl, co, cr] = rotate_and_clean_image(smoothed, rot_angle);
            cleaned = std::move(cl);
            center_orig = co;
            center_rot = cr;
            did_rotate = true;
        }
        last_upright_binary_ = cleaned.clone();
    } else {
        cleaned = smoothed;
        last_upright_binary_ = cleaned.clone();
    }

    // ---- 第5步: 提取最大轮廓 ----
    std::vector<cv::Point> contour = extractLargestContour(cleaned);
    if (contour.empty()) {
        logStep("Result", "Contour extraction failed");
        return Status::InsufficientFeatures;
    }

    // ---- 第6步: approxPolyDP 二分搜索 → 角点 ----
    std::vector<cv::Point2f> corners;
    try {
        corners = extractCornersFromContour(contour, cleaned.size());
    } catch (const std::exception& e) {
        logStep("Result", std::string("Corner extraction failed: ") + e.what());
        return Status::InsufficientFeatures;
    }

    // ---- 第6.5步: 逆旋转角点回原始图像 ----
    if (did_rotate) {
        double rad = rot_angle * CV_PI / 180.0;
        double cos_a = std::cos(rad);
        double sin_a = std::sin(rad);
        for (auto& pt : corners) {
            float dx = pt.x - center_rot.x;
            float dy = pt.y - center_rot.y;
            float rx = static_cast<float>(dx * cos_a + dy * sin_a);
            float ry = static_cast<float>(-dx * sin_a + dy * cos_a);
            pt.x = rx + center_orig.x;
            pt.y = ry + center_orig.y;
        }
    }

    // ---- 第7步: 按模板几何重排角点顺序 ----
    last_corners_before_reorder_ = corners;
    if (last_matched_template_ != nullptr) {
        int tw = last_matched_template_->image.cols;
        int th = last_matched_template_->image.rows;

        // 将模板角点缩放到平滑后的图像空间
        double scale_x = static_cast<double>(smoothed.cols) / tw;
        double scale_y = static_cast<double>(smoothed.rows) / th;

        std::vector<cv::Point2f> scaled_template = last_matched_template_->corners;
        for (auto& pt : scaled_template) {
            pt.x *= static_cast<float>(scale_x);
            pt.y *= static_cast<float>(scale_y);
        }

        auto matches = matchCorners(scaled_template, corners,
                                     static_cast<double>(last_matched_template_->angle) + CPSAGL,
                                     smoothed.size());

        if (!matches.empty()) {
            std::vector<cv::Point2f> ordered(corners.size());
            for (const auto& [tmpl_idx, bin_idx] : matches) {
                if (static_cast<size_t>(bin_idx) < corners.size() &&
                    static_cast<size_t>(tmpl_idx) < corners.size()) {
                    ordered[tmpl_idx] = corners[bin_idx];
                }
            }
            corners = std::move(ordered);
        }
    }

    out_corners = std::move(corners);
    logStep("Result", "Extracted " + std::to_string(out_corners.size()) + " corners");
    return Status::Success;
}

// ============================================================================
// 第1步: 保留最大连通域
// ============================================================================

cv::Mat BinaryCornerExtractor::keepLargestRegion(const cv::Mat& binary_img) {
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary_img, labels, stats, centroids, 8);

    if (num_labels <= 1) {
        logStep("LargestRegion", "No white regions found");
        return binary_img.clone();
    }

    // 查找面积最大的标签（跳过背景标签0）
    int best_label = 1;
    int best_area = stats.at<int>(1, cv::CC_STAT_AREA);
    for (int i = 2; i < num_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > best_area) {
            best_area = area;
            best_label = i;
        }
    }

    cv::Mat result = cv::Mat::zeros(binary_img.size(), CV_8UC1);
    result.setTo(255, labels == best_label);

    logStep("LargestRegion",
            "Kept 1 region / " + std::to_string(num_labels - 1) +
            ", area=" + std::to_string(best_area));
    return result;
}

cv::Mat BinaryCornerExtractor::keepRegionFromCenter(const cv::Mat& binary_img) {
    // 从图像几何中心向外螺旋搜索，遇到的第一个白色像素所在连通域 = 目标
    int cx = binary_img.cols / 2;
    int cy = binary_img.rows / 2;

    // 如果中心点本身是白色，直接作为种子
    cv::Point seed(-1, -1);
    if (binary_img.at<uchar>(cy, cx) == 255) {
        seed = cv::Point(cx, cy);
    } else {
        // 从中心向外逐层搜索
        int max_r = std::max(binary_img.cols, binary_img.rows);
        for (int r = 1; r < max_r && seed.x < 0; ++r) {
            // 遍历半径为 r 的方形边界上的点
            for (int dy = -r; dy <= r && seed.x < 0; ++dy) {
                for (int dx = -r; dx <= r && seed.x < 0; ++dx) {
                    if (std::abs(dx) != r && std::abs(dy) != r) continue;
                    int sx = cx + dx, sy = cy + dy;
                    if (sx < 0 || sx >= binary_img.cols || sy < 0 || sy >= binary_img.rows)
                        continue;
                    if (binary_img.at<uchar>(sy, sx) == 255)
                        seed = cv::Point(sx, sy);
                }
            }
        }
    }

    if (seed.x < 0) {
        logStep("FromCenter", "No white pixel found");
        return binary_img.clone();
    }

    // Flood fill：将命中的连通域染成灰色(128)，其余像素不变
    cv::Mat work = binary_img.clone();
    cv::Rect fill_rect;
    cv::floodFill(work, seed, cv::Scalar(128), &fill_rect,
                  cv::Scalar(0), cv::Scalar(0), 4);

    // 提取灰色区域 → 二值
    cv::Mat region = (work == 128);
    cv::Mat result;
    region.convertTo(result, CV_8UC1, 255.0);

    logStep("FromCenter",
            "Flood fill from (" + std::to_string(seed.x) + "," + std::to_string(seed.y) +
            "), bbox=" + std::to_string(fill_rect.width) + "x" + std::to_string(fill_rect.height));
    return result;
}

// ============================================================================
// 第2步: 填充孔洞
// ============================================================================

cv::Mat BinaryCornerExtractor::fillHoles(const cv::Mat& binary_img) {
    int orig_count = cv::countNonZero(binary_img);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat filled = cv::Mat::zeros(binary_img.size(), CV_8UC1);
    cv::drawContours(filled, contours, -1, cv::Scalar(255), cv::FILLED);

    int filled_count = cv::countNonZero(filled);
    logStep("FillHoles",
            "White pixels: " + std::to_string(orig_count) +
            " -> " + std::to_string(filled_count));
    return filled;
}

// ============================================================================
// 第3步: 形态学平滑
// ============================================================================

cv::Mat BinaryCornerExtractor::smoothBoundary(const cv::Mat& binary_img) {
    cv::Mat closed, opened;
    cv::morphologyEx(binary_img, closed, cv::MORPH_CLOSE, kernel_);
    cv::morphologyEx(closed, opened, cv::MORPH_OPEN, kernel_);

    int smoothed_count = cv::countNonZero(opened);
    logStep("SmoothBoundary",
            "Kernel: " + std::to_string(config_.kernel_size) +
            ", white pixels: " + std::to_string(smoothed_count));
    return opened;
}

// ============================================================================
// 第5步: 提取最大轮廓
// ============================================================================

std::vector<cv::Point> BinaryCornerExtractor::extractLargestContour(
    const cv::Mat& binary_img) {

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        logStep("LargestContour", "No contours found");
        return {};
    }

    auto largest = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });

    double area = cv::contourArea(*largest);
    logStep("LargestContour",
            "Area: " + std::to_string(static_cast<int>(area)) +
            ", points: " + std::to_string(largest->size()));
    return *largest;
}

// ============================================================================
// 第6步: 从轮廓提取角点 (approxPolyDP 二分搜索)
// ============================================================================

std::vector<cv::Point2f> BinaryCornerExtractor::extractCornersFromContour(
    const std::vector<cv::Point>& contour, const cv::Size& /*img_size*/) {

    std::vector<cv::Point2f> work_contour;
    work_contour.reserve(contour.size());

    // 可选地对轮廓进行缩放，以便在小图像上获得亚像素精度
    if (std::abs(config_.scale - 1.0) > 1e-6) {
        float s = static_cast<float>(config_.scale);
        for (const auto& pt : contour) {
            work_contour.emplace_back(pt.x * s, pt.y * s);
        }
    } else {
        for (const auto& pt : contour) {
            work_contour.emplace_back(static_cast<float>(pt.x),
                                       static_cast<float>(pt.y));
        }
    }

    double perimeter = cv::arcLength(work_contour, true);

    // Epsilon 二分搜索: 较大的 epsilon → 较少的角点
    double lo = 0.001, hi = 0.05;
    std::vector<cv::Point2f> approx;
    std::vector<cv::Point2f> best_approx;   // 跟踪最接近目标角点数的近似结果
    int best_diff = std::numeric_limits<int>::max();
    bool exact_hit = false;

    for (int iter = 0; iter < 8; ++iter) {
        double mid = (lo + hi) / 2.0;
        cv::approxPolyDP(work_contour, approx, mid * perimeter, true);
        int n = static_cast<int>(approx.size());
        int diff = std::abs(n - config_.corners);

        // 跟踪最佳近似
        if (diff < best_diff) {
            best_diff = diff;
            best_approx = approx;
        }

        if (n == config_.corners) {
            exact_hit = true;
            break;
        } else if (n < config_.corners) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    // 如果未精确命中，使用最接近的一次近似结果
    if (!exact_hit && !best_approx.empty()) {
        std::cout << "[BinaryCorner] approxPolyDP binary search: no exact hit, "
                  << "best=" << best_approx.size() << " corners (target="
                  << config_.corners << ", diff=" << best_diff << ")" << std::endl;
        approx = std::move(best_approx);
    }

    if (approx.empty()) {
        throw std::runtime_error("approxPolyDP returned empty");
    }

    // 缩放回原始坐标
    if (std::abs(config_.scale - 1.0) > 1e-6) {
        float inv_scale = 1.0f / static_cast<float>(config_.scale);
        for (auto& pt : approx) {
            pt.x *= inv_scale;
            pt.y *= inv_scale;
        }
    }

    logStep("ExtractCorners",
            "Extracted " + std::to_string(approx.size()) + " corners (target " +
            std::to_string(config_.corners) + ")");
    return approx;
}

// ============================================================================
// 模板匹配 —— IoU遍历24个模板找最佳角度
// ============================================================================

BinaryCornerExtractor::BestMatch BinaryCornerExtractor::findBestMatch(
    const cv::Mat& binary_img) {

    BestMatch best;
    if (templates_.empty()) {
        logStep("FindBestMatch", "No templates available");
        return best;
    }

    auto [norm_binary, success] = extractAndNormalizeRoi(binary_img, 50);
    if (!success) {
        logStep("FindBestMatch", "ROI normalization failed");
        return best;
    }

    for (size_t i = 0; i < templates_.size(); ++i) {
        double overlap = calculateOverlap(norm_binary, templates_[i].image_bool);
        if (overlap > best.overlap) {
            best.overlap = overlap;
            best.template_index = static_cast<int>(i);
        }
    }

    logStep("FindBestMatch",
            "Best: angle=" + std::to_string(templates_[best.template_index].angle) +
            " deg, overlap=" + std::to_string(best.overlap));
    return best;
}

// ============================================================================
// 角点重排序 —— 极角对齐（零轴=正上方，CCW正向遍历）
// ============================================================================

std::vector<int> BinaryCornerExtractor::reorderByGeometry(
    const std::vector<cv::Point2f>& corners,
    const cv::Point2f& center,
    double reference_angle_deg,
    double ref_dist) {

    int n = static_cast<int>(corners.size());
    if (n == 0) return {};

    // 将参考角度归一化到 [0, 2π): 零轴 = 正上方（图像y负方向），CCW = 正方向
    double ref_deg = std::fmod(reference_angle_deg, 360.0);
    if (ref_deg < 0.0) ref_deg += 360.0;
    double ref_rad = ref_deg * CV_PI / 180.0;

    // 找到极角最接近参考角度的角点（以及次近的角点，用于距离去歧义）
    int best_idx = 0, second_idx = -1;
    double best_diff = std::numeric_limits<double>::max();
    double second_diff = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        double dx = corners[i].x - center.x;
        double dy = corners[i].y - center.y;
        // atan2(-dx, -dy): 零轴 = 正上方，CCW = 正方向
        double a = std::atan2(-dx, -dy);
        if (a < 0.0) a += 2.0 * CV_PI;
        double diff = std::abs(a - ref_rad);
        if (diff > CV_PI) diff = 2.0 * CV_PI - diff;  // 角度环绕处理
        if (diff < best_diff) {
            second_diff = best_diff;
            second_idx  = best_idx;
            best_diff = diff;
            best_idx = i;
        } else if (diff < second_diff) {
            second_diff = diff;
            second_idx = i;
        }
    }

    int start_idx = best_idx;

    // 距离去歧义：当提供了 ref_dist 时，在两个角度最接近的候选点中选择距离更近的那个
    if (ref_dist > 0.0 && second_idx >= 0) {
        double d1 = std::hypot(corners[best_idx].x - center.x,
                               corners[best_idx].y - center.y);
        double d2 = std::hypot(corners[second_idx].x - center.x,
                               corners[second_idx].y - center.y);
        if (std::abs(d1 - ref_dist) > std::abs(d2 - ref_dist)) {
            start_idx = second_idx;
        }
    }

    // 轮廓为逆时针方向（cv::findContours 保证白色外轮廓为 CCW），始终正向遍历
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) {
        order[i] = (start_idx + i) % n;
    }
    return order;
}

// ============================================================================
// 角点匹配 —— matchCorners
// ============================================================================

std::vector<std::pair<int, int>> BinaryCornerExtractor::matchCorners(
    const std::vector<cv::Point2f>& template_corners,
    const std::vector<cv::Point2f>& binary_corners,
    double template_angle,
    const cv::Size& coord_size) {

    int n_tmpl = static_cast<int>(template_corners.size());
    int n_bin  = static_cast<int>(binary_corners.size());

    if (n_tmpl == 0 || n_bin == 0) {
        logStep("MatchCorners",
                "Cannot match: template=" + std::to_string(n_tmpl) +
                ", binary=" + std::to_string(n_bin));
        return {};
    }

    // 两者在同一坐标空间，使用统一的图像中心
    cv::Point2f img_center(coord_size.width / 2.0f, coord_size.height / 2.0f);

    // 模板角点: 按角度约束排序
    auto tmpl_order = reorderByGeometry(template_corners, img_center, template_angle);

    // 二值角点: 相同的角度约束排序
    auto bin_order = reorderByGeometry(binary_corners, img_center, template_angle);

    int n_pairs = std::min(n_tmpl, n_bin);
    std::vector<std::pair<int, int>> matches;
    matches.reserve(n_pairs);
    for (int k = 0; k < n_pairs; ++k) {
        matches.emplace_back(tmpl_order[k], bin_order[k]);
    }

    logStep("MatchCorners", "Matched " + std::to_string(matches.size()) + " pairs");
    return matches;
}

// ============================================================================
// 静态可视化
// ============================================================================

cv::Mat BinaryCornerExtractor::drawCorners(const cv::Mat& img,
                                            const std::vector<cv::Point2f>& corners) {
    cv::Mat vis;
    if (img.channels() == 1) {
        cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);
    } else {
        vis = img.clone();
    }

    const cv::Scalar colors[] = {
        {0, 0, 255},    // 红
        {0, 255, 0},    // 绿
        {0, 255, 255},  // 黄
    };
    const cv::Scalar default_color(255, 0, 0); // 蓝

    for (size_t i = 0; i < corners.size(); ++i) {
        cv::Scalar color = (i < 3) ? colors[i] : default_color;
        cv::Point pt(static_cast<int>(std::round(corners[i].x)),
                      static_cast<int>(std::round(corners[i].y)));
        cv::circle(vis, pt, 0, color, -1);
    }
    return vis;
}

cv::Mat BinaryCornerExtractor::drawMatchedCorners(
    const cv::Mat& input_img,
    const std::vector<cv::Point2f>& input_corners,
    const cv::Mat& template_img,
    const std::vector<cv::Point2f>& template_corners,
    double template_angle) {

    int h1 = input_img.rows, w1 = input_img.cols;
    int h2 = template_img.rows, w2 = template_img.cols;

    cv::Mat left, right;
    if (input_img.channels() == 1)
        cv::cvtColor(input_img, left, cv::COLOR_GRAY2BGR);
    else
        left = input_img.clone();

    if (template_img.channels() == 1)
        cv::cvtColor(template_img, right, cv::COLOR_GRAY2BGR);
    else
        right = template_img.clone();

    int h = std::max(h1, h2);
    cv::Mat canvas(h, w1 + w2, CV_8UC3, cv::Scalar(0, 0, 0));
    left.copyTo(canvas(cv::Rect(0, 0, w1, h1)));
    right.copyTo(canvas(cv::Rect(w1, 0, w2, h2)));

    const cv::Scalar colors[] = {
        {0, 0, 255}, {0, 255, 0}, {0, 255, 255},
        {255, 0, 0}, {255, 0, 255}, {255, 255, 0},
        {128, 0, 255}, {0, 128, 255}, {255, 128, 0}, {128, 255, 0}
    };
    const int n_colors = sizeof(colors) / sizeof(colors[0]);

    int n_pairs = std::min(static_cast<int>(input_corners.size()),
                            static_cast<int>(template_corners.size()));

    for (int i = 0; i < n_pairs; ++i) {
        cv::Scalar color = colors[i % n_colors];
        cv::Point ipt(static_cast<int>(std::round(input_corners[i].x)),
                       static_cast<int>(std::round(input_corners[i].y)));
        cv::Point tpt(static_cast<int>(std::round(template_corners[i].x)) + w1,
                       static_cast<int>(std::round(template_corners[i].y)));

        cv::circle(canvas, ipt, 0, color, -1);
        cv::circle(canvas, tpt, 0, color, -1);
    }

    // 绘制中心 + 模板角度射线（两侧）
    double a_rad = (template_angle * CV_PI / 180.0);

    cv::Point left_center(w1 / 2, h1 / 2);
    int left_r = std::min(w1, h1) / 3;
    cv::Point left_end(left_center.x + static_cast<int>(-left_r * std::sin(a_rad)),
                       left_center.y + static_cast<int>(-left_r * std::cos(a_rad)));
    cv::circle(canvas, left_center, 1, cv::Scalar(0, 255, 255), -1);
    cv::line(canvas, left_center, left_end, cv::Scalar(0, 255, 255), 1);

    cv::Point right_center(w1 + w2 / 2, h2 / 2);
    int right_r = std::min(w2, h2) / 3;
    cv::Point right_end(right_center.x + static_cast<int>(-right_r * std::sin(a_rad)),
                        right_center.y + static_cast<int>(-right_r * std::cos(a_rad)));
    cv::circle(canvas, right_center, 1, cv::Scalar(0, 255, 255), -1);
    cv::line(canvas, right_center, right_end, cv::Scalar(0, 255, 255), 1);

    return canvas;
}

std::vector<cv::Mat> BinaryCornerExtractor::debugVisualizeReordering(
    const cv::Mat& binary_img,
    const cv::Mat& template_img,
    const std::vector<cv::Point2f>& binary_corners,
    const std::vector<cv::Point2f>& template_corners,
    double template_angle,
    const cv::Size& coord_size) {

    const cv::Scalar RED(0, 0, 255), YLW(0, 255, 255), BLU(255, 0, 0), GRN(0, 255, 0);
    const cv::Scalar WHT(255, 255, 255), YEL(0, 255, 255), CYAN(255, 255, 0);

    cv::Point2f ctr(coord_size.width / 2.0f, coord_size.height / 2.0f);

    // 仅按角度找首个角点（无距离去歧义）
    auto findFirst = [&](const std::vector<cv::Point2f>& pts, double ref_deg) {
        return reorderByGeometry(pts, ctr, ref_deg)[0];
    };
    int tmpl_first = findFirst(template_corners, template_angle);
    int bin_first  = findFirst(binary_corners, template_angle);

    auto toBgr = [&](const cv::Mat& img) -> cv::Mat {
        cv::Mat out;
        if (img.channels() == 1)
            cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
        else
            out = img.clone();
        return out;
    };

    auto drawCornerSet = [&](cv::Mat& img, const std::vector<cv::Point2f>& pts, int first_idx) {
        const cv::Scalar colors[] = {RED, YLW, BLU};
        for (size_t i = 0; i < pts.size(); ++i) {
            cv::Scalar c = (i < 3) ? colors[i] : GRN;
            cv::Point p(static_cast<int>(std::round(pts[i].x)),
                        static_cast<int>(std::round(pts[i].y)));
            cv::circle(img, p, 1, c, -1);
            if (static_cast<int>(i) == first_idx)
                cv::circle(img, p, 7, CYAN, 1);
        }
    };

    auto drawPolarView = [&](cv::Mat& img, const std::vector<cv::Point2f>& pts, int first_idx) {
        cv::Point pc(static_cast<int>(ctr.x), static_cast<int>(ctr.y));
        double a_rad = template_angle * CV_PI / 180.0;
        int r = std::min(coord_size.width, coord_size.height) / 2;

        cv::Point z_end(pc.x, pc.y - r);
        cv::line(img, pc, z_end, WHT, 1);

        cv::Point t_end(pc.x + static_cast<int>(-r * std::sin(a_rad)),
                        pc.y + static_cast<int>(-r * std::cos(a_rad)));
        cv::line(img, pc, t_end, YEL, 1);

        for (size_t i = 0; i < pts.size(); ++i) {
            int px = static_cast<int>(std::round(pts[i].x));
            int py = static_cast<int>(std::round(pts[i].y));
            cv::Scalar clr = (static_cast<int>(i) == first_idx) ? CYAN : GRN;
            cv::circle(img, cv::Point(px, py), 1, clr, -1);
        }

        cv::circle(img, pc, 3, YEL, -1);
    };

    cv::Mat bin_bgr = toBgr(binary_img);
    cv::Mat tmpl_bgr = toBgr(template_img);
    if (tmpl_bgr.size() != binary_img.size())
        cv::resize(tmpl_bgr, tmpl_bgr, binary_img.size(), 0, 0, cv::INTER_NEAREST);

    cv::Mat img0 = bin_bgr.clone();
    drawCornerSet(img0, binary_corners, bin_first);

    cv::Mat img1 = tmpl_bgr.clone();
    drawCornerSet(img1, template_corners, tmpl_first);

    cv::Mat img2 = bin_bgr.clone();
    drawPolarView(img2, binary_corners, bin_first);

    cv::Mat img3 = tmpl_bgr.clone();
    drawPolarView(img3, template_corners, tmpl_first);

    return {img0, img1, img2, img3};
}

// ============================================================================
// 日志记录
// ============================================================================

void BinaryCornerExtractor::logStep(const std::string& step, const std::string& info) {
    process_log_.emplace_back(step, info);
}

// ============================================================================
// 单目特征提取 —— 仅左图 Otsu → 模板匹配 → approxPolyDP 角点
// ============================================================================

PipelineResult BinaryCornerExtractor::extractMono(const cv::Mat& gray,
                                                   const cv::Mat& color) {
    PipelineResult result;
    result.left_color = color;

    if (gray.empty()) {
        std::cerr << "[BinaryCorner::extractMono] empty image" << std::endl;
        return result;
    }

    // Otsu 二值化
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    last_left_binary_ = binary.clone();

    // 从二值图像提取角点
    std::vector<cv::Point2f> corners;
    Status s = extractFromBinary(binary, gray, corners);
    if (s != Status::Success || corners.empty()) {
        std::cerr << "[BinaryCorner::extractMono] extraction failed (status="
                  << static_cast<int>(s) << ")" << std::endl;
        return result;
    }

    // kp_left
    result.kp_left.reserve(corners.size());
    for (const auto& pt : corners) {
        result.kp_left.emplace_back(pt, 1.0f);
    }
    result.n_kp_left = static_cast<int>(corners.size());

    // pts_left_match = 提取到的角点
    result.pts_left_match = corners;

    // 模板匹配数据（使用 0° 模板）
    const TemplateData* matched_tmpl = last_matched_template_;
    const TemplateData* tmpl_0 = nullptr;
    for (const auto& t : templates_) {
        if (t.angle == 0) { tmpl_0 = &t; break; }
    }
    if (tmpl_0 == nullptr) tmpl_0 = matched_tmpl;

    if (tmpl_0 != nullptr) {
        double scale_x = static_cast<double>(gray.cols) / tmpl_0->image.cols;
        double scale_y = static_cast<double>(gray.rows) / tmpl_0->image.rows;

        result.pts_template_match = tmpl_0->corners;
        for (auto& pt : result.pts_template_match) {
            pt.x *= static_cast<float>(scale_x);
            pt.y *= static_cast<float>(scale_y);
        }
        result.n_template_match = static_cast<int>(result.pts_template_match.size());

        int n_match = std::min(static_cast<int>(corners.size()),
                               static_cast<int>(tmpl_0->corners.size()));
        result.good_matches.reserve(n_match);
        for (int i = 0; i < n_match; ++i) {
            result.good_matches.emplace_back(i, i, 0.0f);
        }

        // 3D 物方点
        if (config_.pixel_to_meter_scale > 0.0) {
            double s_mm = config_.pixel_to_meter_scale * 1000.0;
            const auto& tc = tmpl_0->corners;
            template_data_.pts_3d.clear();
            template_data_.pts_3d.reserve(tc.size());
            for (const auto& pt : tc) {
                template_data_.pts_3d.emplace_back(pt.x * s_mm, pt.y * s_mm, 0.0);
            }
        }
    }

    return result;
}

} // namespace gpnp
