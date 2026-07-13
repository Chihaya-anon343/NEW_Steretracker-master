#pragma once

/**
 * @file BinaryCornerExtractor.hpp
 * @brief 通过二值轮廓分析 + approxPolyDP 角点检测进行特征提取。
 *
 * 在 ROI 面积介于 801 到 40000 像素之间时被选中。
 *
 * 流水线：
 *   1. Otsu 二值化（内部——调用方提供灰度 ROI）
 *   2. 保留最大连通分量
 *   3. 填充空洞
 *   4. 形态学平滑（闭运算 → 开运算）
 *   5. 模板匹配（基于 IoU，24 个模板 0°~345°）
 *   6. 旋转回正（如果匹配到模板）
 *   7. 提取最大轮廓
 *   8. approxPolyDP 二分搜索 → N 个角点
 *   9. 将角点旋转回原坐标系（如果曾旋转）
 *  10. 按模板几何排序角点
 *
 * 从 legacy/BinaryCorner.cpp 迁移而来（原始 Python 端口）。
 */

#include "feature/FeatureExtractor.hpp"
#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace gpnp {

class BinaryCornerExtractor : public FeatureExtractor {
public:
    struct Config {
        int corners = 10;                     ///< 期望的角点数量
        int kernel_size = 3;                  ///< 形态学核大小（强制为奇数）
        double scale = 1.0;                   ///< 内部放大系数，用于亚像素精度
        cv::Size target_size{100, 100};       ///< 模板匹配归一化尺寸
        double pixel_to_meter_scale = 0.0;    ///< 比例：1 模板像素 = ? 米
        int roi_pad_pixels = 0;               ///< 在所有方向上扩展 ROI N 个像素
        double otsu_ratio = 1.3;              ///< Otsu 阈值乘数（>1 = 更严格）
    };

    /// 用配置和模板目录构造。
    /// 模板通过 PoseUtils 从 NewMuBan 目录加载。
    BinaryCornerExtractor(const Config& config, const std::string& template_dir);

    ~BinaryCornerExtractor() override = default;

    // ---- FeatureExtractor interface ----

    std::string name() const override { return "BinaryCorner"; }

    StrategyType strategyType() const override { return StrategyType::BinaryCorner; }

    void setTemplateData(const std::string& template_dir,
                         double real_width_mm,
                         double real_height_mm) override;

    /// 完整流水线入口点。
    ///
    /// 输入：left_gray（CV_8UC1 灰度 ROI），right_gray/color 透传。
    /// 输出：PipelineResult 包含：
    ///   - kp_left：提取的角点 KeyPoints
    ///   - pts_left_match：角点坐标（用于调试/PnP）
    ///   - pts_template_match：匹配到的模板角点（缩放到 ROI 尺寸）
    ///   - desc_left、disparity、pts_left_good 等：空（无立体/AKAZE 描述子）
    PipelineResult extract(const cv::Mat& left_gray,
                           const cv::Mat& right_gray,
                           const cv::Mat& left_color,
                           const cv::Mat& right_color) override;

    PipelineResult extractMono(const cv::Mat& gray,
                                const cv::Mat& color) override;

    const TemplateData& templateData() const override { return template_data_; }

    // ---- 提取后状态 ----

    const TemplateData* lastMatchedTemplate() const { return last_matched_template_; }
    double lastMatchOverlap() const { return last_match_overlap_; }
    const std::vector<cv::Point2f>& lastCornersBeforeReorder() const {
        return last_corners_before_reorder_;
    }

    /// 返回最近一次提取的二值图像（用于可视化调试）
    const cv::Mat& lastLeftBinary()  const { return last_left_binary_; }
    const cv::Mat& lastRightBinary() const { return last_right_binary_; }

    /// 返回旋转回正后的二值图像（用于可视化调试）
    const cv::Mat& lastUprightBinary() const { return last_upright_binary_; }

    // ---- 诊断信息 ----

    const std::vector<std::pair<std::string, std::string>>& processLog() const {
        return process_log_;
    }

    // ---- 静态可视化辅助函数 ----

    static cv::Mat drawCorners(const cv::Mat& img,
                                const std::vector<cv::Point2f>& corners);

    static cv::Mat drawMatchedCorners(
        const cv::Mat& input_img,
        const std::vector<cv::Point2f>& input_corners,
        const cv::Mat& template_img,
        const std::vector<cv::Point2f>& template_corners,
        double template_angle = 0.0);

    static std::vector<cv::Mat> debugVisualizeReordering(
        const cv::Mat& binary_img,
        const cv::Mat& template_img,
        const std::vector<cv::Point2f>& binary_corners,
        const std::vector<cv::Point2f>& template_corners,
        double template_angle,
        const cv::Size& coord_size);

    /// 几何角点排序：计算极角，从参考角度开始，逆时针排列。
    static std::vector<int> reorderByGeometry(
        const std::vector<cv::Point2f>& corners,
        const cv::Point2f& center,
        double reference_angle_deg,
        double ref_dist = -1.0);

private:

    /// 从二值图像 (0/255) 提取角点的核心函数。
    /// @param binary_img      Otsu 二值化后的 ROI
    /// @param gray_roi        原始灰度 ROI（用于先旋转再二值化，提高精度）
    /// @param out_corners     输出的角点（ROI 局部坐标）
    /// @param preset_template 可选：预设模板，非空时跳过 findBestMatch() 直接使用
    Status extractFromBinary(const cv::Mat& binary_img,
                              const cv::Mat& gray_roi,
                              std::vector<cv::Point2f>& out_corners,
                              const TemplateData* preset_template = nullptr);

    cv::Mat keepLargestRegion(const cv::Mat& binary_img);
    cv::Mat keepRegionFromCenter(const cv::Mat& binary_img);
    cv::Mat fillHoles(const cv::Mat& binary_img);
    cv::Mat smoothBoundary(const cv::Mat& binary_img);
    std::vector<cv::Point> extractLargestContour(const cv::Mat& binary_img);
    std::vector<cv::Point2f> extractCornersFromContour(
        const std::vector<cv::Point>& contour, const cv::Size& img_size);

    // 模板匹配
    struct BestMatch {
        int template_index = -1;
        double overlap = 0.0;
    };
    BestMatch findBestMatch(const cv::Mat& binary_img);
    std::vector<std::pair<int, int>> matchCorners(
        const std::vector<cv::Point2f>& template_corners,
        const std::vector<cv::Point2f>& binary_corners,
        double template_angle,
        const cv::Size& coord_size);

    // 日志记录
    void logStep(const std::string& step, const std::string& info);

    // ========================================================================
    // 状态
    // ========================================================================

    Config config_;
    cv::Mat kernel_;
    std::vector<TemplateData> templates_;       ///< 来自 NewMuBan 的角点模板
    TemplateData template_data_;                 ///< 空占位符（无 AKAZE 模板）

    // 提取后状态
    const TemplateData* last_matched_template_ = nullptr;
    double last_match_overlap_ = 0.0;
    std::vector<cv::Point2f> last_corners_before_reorder_;
    std::vector<std::pair<std::string, std::string>> process_log_;

    // 最近一次提取的二值图像（用于可视化）
    cv::Mat last_left_binary_;
    cv::Mat last_right_binary_;
    cv::Mat last_upright_binary_;    // 旋转回正后的二值图
};

} // namespace gpnp
