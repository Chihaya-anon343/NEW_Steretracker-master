#pragma once

/**
 * @file FeatureExtractor.hpp
 * @brief 特征提取的抽象策略接口。
 *
 * 策略：
 *   - AkazeGpnpExtractor   (ROI 面积 > 40000)：AKAZE + LK 光流 + GPNP
 *   - BinaryCornerExtractor (801 ~ 40000)：    轮廓角点检测
 *   - TinyTargetExtractor   (≤ 800)：           超分辨率 minAreaRect
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace gpnp {

/// 退化链条分发的策略类型。
enum class StrategyType {
    Akaze,          ///< AKAZE 特征提取 + 光流 + GPNP
    BinaryCorner,   ///< Otsu 二值轮廓角点提取 + GPNP
    TinyTarget      ///< 超分辨率 minAreaRect 四点提取 + solvePnP
};

/// 特征提取策略的抽象接口。
///
/// 每种策略实现特征提取 + 模板匹配，
/// 生成可供下游流水线（MAD 滤波、GPNP 位姿估计）使用的 PipelineResult。
class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;

    /// 供日志使用的可读策略名称。
    virtual std::string name() const = 0;

    /// PnP 分发和退化链条排序所需的策略类型。
    virtual StrategyType strategyType() const = 0;

    /// 接受并加载模板数据（构造后调用一次）。
    ///
    /// 对于 AKAZE 策略，从参考模板图像中提取 AKAZE 特征。
    /// 对于 BinaryCorner/TinyTarget 策略，从目录加载 PNG+TXT 角点模板。
    ///
    /// @param template_dir     模板图像或模板目录的路径
    /// @param real_width_mm    模板实际宽度（mm）—— 仅 AKAZE 使用
    /// @param real_height_mm   模板实际高度（mm）—— 仅 AKAZE 使用
    virtual void setTemplateData(const std::string& template_dir,
                                  double real_width_mm,
                                  double real_height_mm) = 0;

    /// 从双目 ROI 图像对中提取特征。
    ///
    /// 对于 AKAZE 返回完整填充的 PipelineResult（关键点、描述子、追踪点、双目投影、模板匹配）。
    /// 对于桩策略，返回空/零结果并附带错误状态。
    ///
    /// @param left_gray    左相机灰度 ROI
    /// @param right_gray   右相机灰度 ROI
    /// @param left_color   左相机彩色 ROI（用于可视化）
    /// @param right_color  右相机彩色 ROI（用于可视化）
    /// @return             包含提取特征数据的 PipelineResult
    virtual PipelineResult extract(const cv::Mat& left_gray,
                                    const cv::Mat& right_gray,
                                    const cv::Mat& left_color,
                                    const cv::Mat& right_color) = 0;

    /// 返回此策略使用的模板数据。
    virtual const TemplateData& templateData() const = 0;
};

} // namespace gpnp
