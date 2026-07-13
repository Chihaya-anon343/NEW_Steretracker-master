#pragma once

/**
 * @file TinyTargetExtractor.hpp
 * @brief 微小矩形目标（≤800 px 面积）的特征提取器。
 *
 * 使用超分辨率 + 最小外接矩形 + 亚像素角点优化。
 * 当 ROI 面积 ≤ 800 px 时被选中。
 *
 * 流水线:
 *   1. Otsu 二值化 → 模板 IoU 匹配 → 最佳角度
 *   2. 超分辨率（×scale_factor）→ Otsu → 形态学开+闭操作
 *   3. 连通域评分（矩形度、面积、中心、纵横比）
 *   4. 最小外接矩形 → 4个角点 → 亚像素优化
 *   5. 角度对齐 → 缩放还原 → 4个有序角点
 *
 * 从 legacy/TinyTargetPose.cpp 迁移（原始 Python 移植）。
 */

#include "feature/FeatureExtractor.hpp"
#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace gpnp {

class TinyTargetExtractor : public FeatureExtractor {
public:
    struct Config {
        cv::Size target_size{50, 50};         ///< 模板匹配归一化尺寸
        int scale_factor = 4;                 ///< 超分辨率放大因子（2~8）
        double square_size_m = 0.20;          ///< 目标物理边长（米）
        int roi_pad_pixels = 0;               ///< ROI 四周扩展像素数
    };

    /// 使用配置和模板目录构造。
    /// 模板通过 PoseUtils 从 NewMuBan 目录加载。
    TinyTargetExtractor(const Config& config, const std::string& template_dir);

    ~TinyTargetExtractor() override = default;

    // ---- FeatureExtractor interface ----

    std::string name() const override { return "TinyTarget"; }

    StrategyType strategyType() const override { return StrategyType::TinyTarget; }

    void setTemplateData(const std::string& template_dir,
                         double real_width_mm,
                         double real_height_mm) override;

    /// 完整流水线：从左右图像各提取4个角点，填充 PipelineResult
    /// 用于双目 GPNP（与 BinaryCornerExtractor 模式相同）。
    PipelineResult extract(const cv::Mat& left_gray,
                           const cv::Mat& right_gray,
                           const cv::Mat& left_color,
                           const cv::Mat& right_color) override;

    const TemplateData& templateData() const override { return template_data_; }

    // ---- Post-extraction state ----

    int lastMatchedAngle() const { return last_best_angle_; }
    double lastMatchOverlap() const { return last_best_overlap_; }

    /// 返回匹配到的模板（用于可视化），未匹配时返回 nullptr。
    const TemplateData* lastMatchedTemplate() const {
        if (last_best_angle_ < 0) return nullptr;
        for (const auto& t : templates_)
            if (t.angle == last_best_angle_) return &t;
        return nullptr;
    }

private:
    /// 从单张灰度 ROI 中提取4个角点。
    /// 返回 ROI 局部坐标中的角点（TL→TR→BR→BL）。
    Status extract4Corners(const cv::Mat& roi_gray,
                            std::vector<cv::Point2f>& out_corners,
                            int& best_angle,
                            double& best_overlap);

    // Template matching
    struct TemplateMatchResult {
        int best_angle = -1;
        double best_overlap = 0.0;
        std::vector<std::pair<int, double>> all_overlaps;
    };
    TemplateMatchResult matchTemplate(const cv::Mat& roi_binary);

    // Component scoring & selection
    int selectBestComponent(const cv::Mat& binary, const cv::Mat& label_map,
                            int num_labels, const cv::Mat& stats,
                            const cv::Mat& centroids);

    // Sub-pixel corner refinement
    std::vector<cv::Point2f> refineCorners(const cv::Mat& image,
                                            const std::vector<cv::Point2f>& corners,
                                            int win_size);

    Config config_;
    std::vector<TemplateData> templates_;       ///< 来自 NewMuBan 的角点模板
    TemplateData template_data_;                 ///< 存储用于 GPNP 的 pts_3d

    // 提取后状态
    int last_best_angle_ = -1;
    double last_best_overlap_ = 0.0;
};

} // namespace gpnp
