#pragma once

/**
 * @file YoloDecode.hpp
 * @brief 原始 YOLOv8 输出解码 + NMS 的纯函数实现（无 ONNX Runtime 依赖）。
 *
 * 从 YoloDetector::postprocess 抽取，便于单元测试直接构造合成原始张量
 * 验证解码/NMS 逻辑，无需加载 ONNX 模型。
 *
 * 原始 YOLOv8 输出为 [1, 4+nc, N] (BCN) 或 [1, N, 4+nc] (BNC)，
 * 前 4 通道为 cx,cy,w,h（输入尺度），随后 nc 个类别分数。
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace gpnp {

/**
 * 从原始输出张量维度推导 (num_attrs, num_preds, is_bcn)。
 * 兼容 [1,A,N] / [A,N] / [1,A,N,1] 等布局：剥离前导 batch 维与尾部单元素维后
 * 必须恰好剩 2 维；属性维(4+nc 或 4+nc+1) = 较小者，预测框维 N = 较大者，
 * is_bcn = 属性维在前。两维相等（无法区分）或维度异常时返回 false。
 * 模板同时覆盖 ONNX 的 int64_t shape 与 RKNN 的 uint32_t dims。
 */
template <typename T>
inline bool resolveYoloOutputLayout(const T* dims, size_t n_dims,
                                    size_t& num_preds, size_t& num_attrs,
                                    bool& is_bcn) {
    if (dims == nullptr || n_dims < 2) {
        return false;
    }
    size_t b = 0;
    size_t e = n_dims;
    if (dims[b] == 1 && e - b > 2) {
        ++b;  // 剥离前导 batch 维
    }
    while (e - b > 2 && dims[e - 1] == 1) {
        --e;  // 剥离尾部单元素维
    }
    if (e - b != 2) {
        return false;
    }
    const size_t d0 = static_cast<size_t>(dims[b]);
    const size_t d1 = static_cast<size_t>(dims[b + 1]);
    if (d0 == 0 || d1 == 0 || d0 == d1) {
        return false;
    }
    is_bcn = (d0 < d1);
    num_attrs = is_bcn ? d0 : d1;
    num_preds = is_bcn ? d1 : d0;
    return true;
}

/**
 * 解码原始 YOLOv8 输出为 Detection 列表（含置信度过滤 + NMS + 反向 letterbox + clamp）。
 *
 * @param data           原始输出张量的 float 数据指针（已去除 batch 维）
 * @param shape_is_bcn   true: 布局 [4+nc, N]（属性维在前）; false: [N, 4+nc]
 * @param num_preds      预测框数 N
 * @param num_classes    类别数 nc（需 >= 1）
 * @param ratio/dw/dh    letterbox 缩放比与左上 padding（用于反变换到原图坐标）
 * @param img_w/img_h    原始图像宽高
 * @param conf_threshold 置信度阈值
 * @param iou_threshold  NMS IoU 阈值
 */
inline std::vector<Detection> decodeYoloOutput(
    const float* data, bool shape_is_bcn,
    size_t num_preds, size_t num_classes,
    float ratio, float dw, float dh,
    int img_w, int img_h,
    float conf_threshold, float iou_threshold)
{
    std::vector<Detection> candidates;
    if (!data || num_preds == 0 || num_classes == 0) {
        return candidates;
    }

    const size_t num_attrs = 4 + num_classes;   // cx,cy,w,h + nc 类分数

    for (size_t i = 0; i < num_preds; ++i) {
        float cx, cy, w, h;
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
        } else {
            const float* row = data + i * num_attrs;
            cx = row[0]; cy = row[1]; w = row[2]; h = row[3];
            for (size_t c = 0; c < num_classes; ++c) {
                float s = row[4 + c];
                if (s > score) { score = s; cls = static_cast<int>(c); }
            }
        }

        if (score < conf_threshold) continue;

        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // 反向 letterbox 变换
        x1 = (x1 - dw) / ratio;
        x2 = (x2 - dw) / ratio;
        y1 = (y1 - dh) / ratio;
        y2 = (y2 - dh) / ratio;

        if (!std::isfinite(x1) || !std::isfinite(y1) ||
            !std::isfinite(x2) || !std::isfinite(y2)) {
            continue;
        }
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        x1 = std::clamp(x1, 0.0f, static_cast<float>(img_w));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(img_w));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(img_h));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(img_h));

        Detection det;
        det.class_id = cls;
        det.confidence = score;
        det.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        candidates.push_back(std::move(det));
    }

    // NMS（贪心：按置信度降序，抑制 IoU 超阈框）
    auto iou = [](const cv::Rect2f& a, const cv::Rect2f& b) {
        float ix1 = std::max(a.x, b.x);
        float iy1 = std::max(a.y, b.y);
        float ix2 = std::min(a.x + a.width,  b.x + b.width);
        float iy2 = std::min(a.y + a.height, b.y + b.height);
        float iw  = std::max(0.0f, ix2 - ix1);
        float ih  = std::max(0.0f, iy2 - iy1);
        float inter = iw * ih;
        float uni = a.width * a.height + b.width * b.height - inter;
        return uni > 0.0f ? inter / uni : 0.0f;
    };

    std::vector<int> order(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return candidates[a].confidence > candidates[b].confidence; });

    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<Detection> detections;
    detections.reserve(candidates.size());
    for (int i : order) {
        if (suppressed[i]) continue;
        detections.push_back(candidates[i]);
        for (int j : order) {
            if (i == j || suppressed[j]) continue;
            if (iou(candidates[i].bbox, candidates[j].bbox) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return detections;
}

} // namespace gpnp
