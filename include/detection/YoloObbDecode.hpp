#pragma once

/**
 * @file YoloObbDecode.hpp
 * @brief 原始 YOLOv8-OBB 输出解码 + 旋转 NMS 的纯函数实现（无 ONNX Runtime 依赖）。
 *
 * 与 YoloDecode.hpp 的普通检测输出 [1, 4+nc, N] 不同，OBB 导出的输出为
 * [1, 4+nc+1, N]（BCN）或 [1, N, 4+nc+1]（BNC）：
 *   前 4 通道为 cx,cy,w,h（输入尺度），随后 nc 个类别分数，最后 1 通道为
 *   旋转角（弧度，ultralytics xywhr 约定）。
 *
 * letterbox 反变换只作用于 cx,cy,w,h（缩放与平移），角度不受影响。
 * NMS 采用贪心类内旋转 IoU（cv::rotatedRectangleIntersection）。
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace gpnp {

/// OBB 检测结果：四角点（原图坐标，顺序未定）+ 旋转矩形。
struct ObbDetection {
    int class_id{0};
    float confidence{0.0f};
    std::array<cv::Point2f, 4> quad{};   ///< 旋转矩形 4 角点（原图坐标）
    cv::RotatedRect rrect{};             ///< 中心/尺寸/角度（度）
};

/// 两个旋转矩形的 IoU（交集多边形面积 / 并集面积）。
inline float rotatedIoU(const cv::RotatedRect& a, const cv::RotatedRect& b) {
    std::vector<cv::Point2f> inter;
    try {
        cv::rotatedRectangleIntersection(a, b, inter);
    } catch (const cv::Exception&) {
        return 0.0f;
    }
    if (inter.empty()) {
        return 0.0f;
    }
    const float inter_area = std::fabs(cv::contourArea(inter));
    const float uni = a.size.area() + b.size.area() - inter_area;
    return uni > 0.0f ? inter_area / uni : 0.0f;
}

/**
 * 解码原始 YOLOv8-OBB 输出为 ObbDetection 列表
 * （含置信度过滤 + 类内旋转 NMS + 反向 letterbox + clamp）。
 *
 * @param data           原始输出张量的 float 数据指针（已去除 batch 维）
 * @param shape_is_bcn   true: 布局 [4+nc+1, N]; false: [N, 4+nc+1]
 * @param num_preds      预测框数 N
 * @param num_classes    类别数 nc（需 >= 1）
 * @param ratio/dw/dh    letterbox 缩放比与左上 padding（仅作用于 xywh）
 * @param img_w/img_h    原始图像宽高
 * @param conf_threshold 置信度阈值
 * @param iou_threshold  NMS 旋转 IoU 阈值
 */
inline std::vector<ObbDetection> decodeYoloObbOutput(
    const float* data, bool shape_is_bcn,
    size_t num_preds, size_t num_classes,
    float ratio, float dw, float dh,
    int img_w, int img_h,
    float conf_threshold, float iou_threshold)
{
    std::vector<ObbDetection> candidates;
    if (!data || num_preds == 0 || num_classes == 0) {
        return candidates;
    }

    const size_t num_attrs = 4 + num_classes + 1;   // cx,cy,w,h + nc 类分数 + 角度

    for (size_t i = 0; i < num_preds; ++i) {
        float cx, cy, w, h, ang_rad;
        int cls = -1;
        float score = -1.0f;

        if (shape_is_bcn) {
            cx = data[0 * num_preds + i];
            cy = data[1 * num_preds + i];
            w  = data[2 * num_preds + i];
            h  = data[3 * num_preds + i];
            for (size_t c = 0; c < num_classes; ++c) {
                float s = data[(4 + c) * num_preds + i];
                if (s > score) { score = s; cls = static_cast<int>(c); }
            }
            ang_rad = data[(4 + num_classes) * num_preds + i];
        } else {
            const float* row = data + i * num_attrs;
            cx = row[0]; cy = row[1]; w = row[2]; h = row[3];
            for (size_t c = 0; c < num_classes; ++c) {
                float s = row[4 + c];
                if (s > score) { score = s; cls = static_cast<int>(c); }
            }
            ang_rad = row[4 + num_classes];
        }

        if (score < conf_threshold) continue;

        // 反向 letterbox：仅 xywh 受缩放/平移影响，角度不变
        cx = (cx - dw) / ratio;
        cy = (cy - dh) / ratio;
        w  = w / ratio;
        h  = h / ratio;

        if (!std::isfinite(cx) || !std::isfinite(cy) ||
            !std::isfinite(w) || !std::isfinite(h) ||
            !std::isfinite(ang_rad) || w <= 0.0f || h <= 0.0f) {
            continue;
        }

        ObbDetection det;
        det.class_id = cls;
        det.confidence = score;
        det.rrect = cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(w, h),
                                    ang_rad * 180.0f / static_cast<float>(CV_PI));
        {
            cv::Mat pts;
            cv::boxPoints(det.rrect, pts);
            for (int k = 0; k < 4; ++k) {
                det.quad[k] = cv::Point2f(pts.at<float>(k, 0), pts.at<float>(k, 1));
            }
        }

        // 中心点必须落在图像范围内（允许部分出画，角点后续 clamp 由几何检查处理）
        if (cx < 0.0f || cy < 0.0f || cx > static_cast<float>(img_w) ||
            cy > static_cast<float>(img_h)) {
            continue;
        }

        candidates.push_back(det);
    }

    // 类内贪心 NMS：按置信度降序，抑制同类旋转 IoU 超阈框
    std::vector<int> order(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) {
                  return candidates[a].confidence > candidates[b].confidence;
              });

    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<ObbDetection> detections;
    detections.reserve(candidates.size());
    for (int i : order) {
        if (suppressed[i]) continue;
        detections.push_back(candidates[i]);
        for (int j : order) {
            if (i == j || suppressed[j]) continue;
            if (candidates[j].class_id != candidates[i].class_id) continue;
            if (rotatedIoU(candidates[i].rrect, candidates[j].rrect) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return detections;
}

} // namespace gpnp
