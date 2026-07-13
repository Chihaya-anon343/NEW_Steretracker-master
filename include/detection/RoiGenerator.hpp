#pragma once

/**
 * @file RoiGenerator.hpp
 * @brief 将 YOLO 检测结果转换为 StereoTracker 所需的 RoiRect。
 *
 * 处理类别过滤、基于置信度的选择、ROI 扩展、
 * 边界裁剪和最小尺寸强制。
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace gpnp {

// ============================================================================
// ROI 生成器
// ============================================================================

class RoiGenerator {
public:
    struct Config {
        int target_class_id{0};         ///< 仅考虑此类的检测结果
        float roi_expand_ratio{0.1f};   ///< 在每侧按该比例扩展 ROI
        int roi_min_size{100};          ///< ROI 的最小宽度和高度（像素）
    };

    RoiGenerator();
    explicit RoiGenerator(const Config& cfg);

    /// 从检测结果生成指定类别的单个 ROI。
    /// @param class_id  目标类别进行过滤；-1 表示使用 config_.target_class_id。
    /// 成功时返回有效的 RoiRect，无检测结果时返回无效（width=0）的 RoiRect。
    RoiRect generate(const std::vector<Detection>& detections,
                     const cv::Size& image_size,
                     int class_id = -1) const;

    /// 最多生成两个 ROI：class 0（始终生成），class 1（仅当 class 0 面积 > 700*700）。
    /// 存在次级 ROI 时返回 is_dual=true 的 RoiGroup。
    RoiGroup generateGroup(const std::vector<Detection>& detections,
                           const cv::Size& image_size) const;

    /// 从分别的检测集生成左右 ROI 对。
    /// 若右侧检测为空，则将左侧 ROI 复制到右侧。
    std::pair<RoiRect, RoiRect> generateStereo(
        const std::vector<Detection>& detections_left,
        const std::vector<Detection>& detections_right,
        const cv::Size& left_img_size,
        const cv::Size& right_img_size) const;

    /// 从分别的检测集生成左右 RoiGroup 对。
    /// 当两侧 class 0 面积均 > 700*700 时支持双 ROI 模式。
    std::pair<RoiGroup, RoiGroup> generateStereoGroup(
        const std::vector<Detection>& detections_left,
        const std::vector<Detection>& detections_right,
        const cv::Size& left_img_size,
        const cv::Size& right_img_size) const;

private:
    Config config_;

    /// 将单个检测结果转换为扩展并裁剪后的 RoiRect。
    static RoiRect detectionToRoi(const Detection& det, const cv::Size& img_size,
                                   float expand_ratio, int min_size);

    /// 规范化双目 ROI 对：右图锚定缩放、裁剪、尺寸同步。
    static void normalizeStereoPair(RoiRect& left, RoiRect& right,
                                    const Detection* right_det,
                                    const cv::Size& left_img_size,
                                    const cv::Size& right_img_size);
};

} // namespace gpnp
