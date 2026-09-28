#pragma once

/**
 * @file ObbCornerExtractor.hpp
 * @brief YOLO-OBB 四边形 → 透视归一化 → 白色靶标 10 角点提取。
 *
 * D:\Codes\YOLOv8-OBB\main.py 的 C++ 移植（pipeline 分支，最小端到端）：
 *   quad_check（几何检查）→ order_corners → H1 透视归一化 (norm_size 画布)
 *   → process_tile（中值去噪 → Otsu → 最大白色连通域 → 孔洞填充 →
 *      minAreaRect 估角 → H2 二次转正 → 形态学开闭 → approxPolyDP DP 角点）
 *   → k=16 模板恒等比对（4 朝向, 16×16 白占比图）→ 角点校验替换
 *      （野点剔除 / 贪心一对一指派 / mask 1px 验证 / 轮廓贴附）
 *   → (H2·H1)^-1 反投影回原图。
 *
 * 输出角点按 0° 模板 Corner_0..9 统一编号；3D 点由 0° 模板角点经
 * template_real_size_mm 缩放得到（mm, z=0, 中心化），直接供 PnP 使用。
 */

#include "common/Types.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

namespace gpnp {

class ObbCornerExtractor {
public:
    struct Config {
        std::string template_dir;         ///< 4 朝向模板目录 (png + txt)
        int norm_size = 224;              ///< 归一化画布边长（勿低于 224）
        int morph_k = 5;                  ///< 转正后形态学核（<3 关闭）
        int denoise_k = 3;                ///< Otsu 前中值去噪核（<3 关闭）
        double poly_eps = 0.02;           ///< approxPolyDP 系数（占周长比例）
        double tmpl_score = 0.93;         ///< 靶标判定 k=16 分数阈值
        double corner_tau = 6.0;          ///< 角点校验距离阈值（画布 px）
        double snap_r = 24.0;             ///< mask 轮廓贴附搜索半径上限（画布 px）
        double min_area_frac = 1e-4;      ///< 四边形最小面积占整图比例
        double min_side = 8.0;            ///< 四边形最短边阈值（px）
        double template_real_size_mm = 0.0;  ///< 靶标外接方形物理边长 mm（PnP 尺度来源, 必填）
    };

    struct Result {
        bool success = false;             ///< 几何检查 + 提取 + 模板判定均通过
        bool is_target = false;           ///< k=16 模板比对 score >= tmpl_score
        double score = 0.0;               ///< k=16 比对分数
        double extract_ms = 0.0;

        std::vector<cv::Point2f> corners;      ///< 最终 10 角点（原图坐标, 0° 编号）
        std::vector<Eigen::Vector3d> pts_3d;   ///< 与 corners 一一对应（mm, z=0, 中心化）
        std::array<cv::Point2f, 4> quad;       ///< 有序检测四边形 [TL,TR,BR,BL]（原图）
        std::array<cv::Point2f, 4> rect_corners;  ///< 反投影外接方形 4 角点（原图, 0°右下起顺时针）
        int n_dp = 0;                     ///< DP 实测位数
        int n_keep = 0;                   ///< mask 验证保留位数
        int n_snap = 0;                   ///< mask 轮廓贴附位数

        // 调试图像（画布坐标）
        cv::Mat debug_tile;               ///< H1 归一化方块
        cv::Mat debug_mask;               ///< H2 转正后掩码
    };

    explicit ObbCornerExtractor(const Config& cfg);

    bool templatesReady() const { return !templates_.empty() && canonical_ready_; }
    size_t numTemplates() const { return templates_.size(); }

    /// 对单个 OBB 四边形执行完整提取流水线。quad_raw 顺序任意（内部排序）。
    Result extract(const cv::Mat& bgr_img, const std::array<cv::Point2f, 4>& quad_raw);

private:
    /// 模板（归一化画布坐标）：mask / 16×16 占比图 / 质心 / 角点 / 滚转与编号映射。
    struct Template {
        std::string name;
        int deg = 0;                            ///< 朝向（文件名前缀）
        cv::Mat mask;                           ///< 画布二值模板
        cv::Mat frac16;                         ///< 16×16 白占比图 (CV_32F, [0,1])
        cv::Point2d cent{0.0, 0.0};             ///< 模板质心（画布坐标）
        std::vector<cv::Point2f> corners;       ///< 模板角点（画布坐标, 文件原编号）
        double roll = 0.0;                      ///< 相对 0° 模板滚转（度, k*90）
        std::vector<int> perm;                  ///< 文件编号 → 0° 编号映射
    };

    /// process_tile 输出。
    struct TileProc {
        cv::Mat binary;                         ///< tile1 Otsu 二值图
        cv::Mat mask;                           ///< 转正 + 优化后掩码
        cv::RotatedRect rect;                   ///< 转正系最小外接矩形
        std::vector<cv::Point2f> poly;          ///< 转正系 DP 角点
        cv::Mat H2;                             ///< tile1 → 转正系单应
        bool ok = false;                        ///< 存在有效前景
    };

    /// match_template 输出。
    struct MatchOut {
        bool valid = false;
        bool is_target = false;
        double score = 0.0;
        cv::Point2d dt{0.0, 0.0};               ///< 质心配准偏移
        const Template* tmpl = nullptr;
        std::vector<cv::Point2f> corners;       ///< 画布坐标, 0° 编号
        int n_dp = 0, n_keep = 0, n_snap = 0;
    };

    bool loadTemplates();
    static bool quadCheck(const std::array<cv::Point2f, 4>& pts, int w, int h,
                          double min_area, double min_side);
    static std::array<cv::Point2f, 4> orderCorners(const std::array<cv::Point2f, 4>& pts);
    TileProc processTile(const cv::Mat& tile) const;
    MatchOut matchTemplate(const cv::Mat& mask, const std::vector<cv::Point2f>& poly) const;

    Config cfg_;
    std::vector<Template> templates_;
    std::vector<cv::Point2f> canonical_corners_;  ///< 0° 模板角点（0° 编号, 画布坐标）
    bool canonical_ready_ = false;
    int margin_ = 7;                              ///< 转正画布边距（morph_k+2 或 2）
};

} // namespace gpnp
