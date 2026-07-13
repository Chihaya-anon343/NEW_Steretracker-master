#pragma once

/**
 * @file AkazeExtractor.hpp
 * @brief AKAZE 特征检测器/描述子，支持缩放因子处理。
 *
 * 模块：feature
 * 功能：AKAZE 关键点检测 + 描述子计算，可选图像下采样加速，
 *        随后还原坐标缩放。
 * 输入：cv::Mat 灰度图像
 * 输出：FeatureSet {关键点, 描述子, 点坐标}
 * 依赖：OpenCV features2d, Types.hpp
 * 关系：由 StereoTracker 调用；为 OpticalFlowTracker 和 TemplateMatcher 提供数据
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <memory>

namespace gpnp {

class AkazeExtractor {
public:
    /**
     * @brief 构造 AKAZE 提取器。
     *
     * @param scale  图像缩放因子 (0~1)。特征在缩放后图像上提取，
     *               随后关键点坐标除以 scale，映射回原始图像坐标。
     *               scale=1.0 表示不使用缩放。
     */
    explicit AkazeExtractor(double scale = 0.5);

    ~AkazeExtractor();

    // 不可拷贝（持有 cv::Ptr）
    AkazeExtractor(const AkazeExtractor&) = delete;
    AkazeExtractor& operator=(const AkazeExtractor&) = delete;

    // 可移动
    AkazeExtractor(AkazeExtractor&&) noexcept;
    AkazeExtractor& operator=(AkazeExtractor&&) noexcept;

    // ========================================================================
    // Feature Extraction
    // ========================================================================

    /**
     * @brief 从灰度图像提取 AKAZE 特征。
     *
     * 如果 scale < 1.0，先对图像进行下采样，提取特征后，
     * 再将关键点坐标除以 scale 还原。
     *
     * @param gray  输入灰度图像 (CV_8UC1)
     * @return 包含关键点、描述子和点坐标的 FeatureSet
     */
    FeatureSet extract(const cv::Mat& gray);

    /**
     * @brief 提取并缓存模板特征（初始化时调用一次）。
     *
     * 以原始分辨率提取特征（不使用下采样）。
     * 同时根据模板物理尺寸计算三维坐标。
     *
     * @param template_path  模板图像文件路径
     * @param real_width_mm  模板实际宽度（mm）
     * @param real_height_mm 模板实际高度（mm）
     * @return 准备好匹配的 TemplateData
     */
    TemplateData extractTemplate(const std::string& template_path,
                                 double real_width_mm,
                                 double real_height_mm);

    // ========================================================================
    // 特征提取
    // ========================================================================
    // ========================================================================
    // 访问器
    // ========================================================================

    double scale() const { return scale_; }

private:
    cv::Ptr<cv::AKAZE> akaze_;   ///< OpenCV AKAZE 检测器实例
    double scale_;                ///< 图像缩放因子
};

} // namespace gpnp
