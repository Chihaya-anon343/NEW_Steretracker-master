#pragma once

/**
 * @file YoloDecode.hpp
 * @brief YOLO 输出解码 + NMS 的纯函数实现（无推理后端依赖）。
 *
 * 从 YoloDetector::postprocess 抽取，便于单元测试直接构造合成原始张量
 * 验证解码/NMS 逻辑，无需加载模型。
 *
 * 两种输出格式：
 *   - ONNX 原始 YOLOv8 导出: [1, 4+nc, N] (BCN) 或 [1, N, 4+nc] (BNC)，
 *     前 4 通道为 cx,cy,w,h（输入尺度），随后 nc 个类别分数。
 *   - RKNN split-head 导出 (RK3588 NPU): 每尺度 3 输出
 *     [1,64,H,W] DFL raw + [1,nc,H,W] cls(sigmoid) + [1,1,H,W] max，
 *     尺度 H∈{160,80,40,20} (stride 4/8/16/32)。
 *
 * 两种格式共享 finalizeDetections（反 letterbox + 几何校验 + clamp + NMS）。
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace gpnp {

/**
 * 候选框收尾（输入尺度坐标 → 原图坐标 + NMS）。
 *
 * candidates 中的 bbox 为**模型输入尺度**（letterbox 后）的 x1y1x2y2；
 * 本函数做反 letterbox、有限性/退化校验、clamp 与贪心 NMS。
 * 供 decodeYoloOutput / decodeYoloRknnSplit 共用。
 */
inline std::vector<Detection> finalizeDetections(
    std::vector<Detection>& candidates,
    float ratio, float dw, float dh,
    int img_w, int img_h,
    float iou_threshold)
{
    std::vector<Detection> out;
    out.reserve(candidates.size());

    for (auto& det : candidates) {
        float x1 = (det.bbox.x - dw) / ratio;
        float x2 = (det.bbox.x + det.bbox.width - dw) / ratio;
        float y1 = (det.bbox.y - dh) / ratio;
        float y2 = (det.bbox.y + det.bbox.height - dh) / ratio;

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

        det.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        out.push_back(std::move(det));
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

    std::vector<int> order(out.size());
    for (size_t i = 0; i < out.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return out[a].confidence > out[b].confidence; });

    std::vector<bool> suppressed(out.size(), false);
    std::vector<Detection> detections;
    detections.reserve(out.size());
    for (int i : order) {
        if (suppressed[i]) continue;
        detections.push_back(out[i]);
        for (int j : order) {
            if (i == j || suppressed[j]) continue;
            if (out[i].class_id != out[j].class_id) continue;
            if (iou(out[i].bbox, out[j].bbox) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return detections;
}

/**
 * 解码 ONNX 原始 YOLOv8 输出（含置信度过滤 + NMS + 反向 letterbox + clamp）。
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

        Detection det;
        det.class_id = cls;
        det.confidence = score;
        det.bbox = cv::Rect2f(cx - w / 2.0f, cy - h / 2.0f, w, h);
        candidates.push_back(std::move(det));
    }

    return finalizeDetections(candidates, ratio, dw, dh, img_w, img_h, iou_threshold);
}

/**
 * 解码 RKNN split-head YOLOv8 输出（RK3588 NPU 导出格式）。
 *
 * 每尺度两组特征图（NCHW，已 want_float=1 反量化）：
 *   - dfl_maps[s]:  [1, 64, H_s, W_s]  DFL 原始 logits
 *       通道布局: box i (l/t/r/b) 占 [i*16, i*16+16)，每 bin 一个通道
 *   - cls_maps[s]:  [1, nc, H_s, W_s]  类别分数（导出时已过 sigmoid）
 *
 * 解码（上板探针实测验证）：
 *   anchor 中心 ax=(gx+0.5)*stride, ay=(gy+0.5)*stride
 *   DFL: 16-bin softmax 期望值 × stride = ltrb 距离（输入尺度像素）
 *   x1 = ax - l*st, y1 = ay - t*st, x2 = ax + r*st, y2 = ay + b*st
 *
 * 惰性 DFL：仅对类别分数过阈值的 anchor 解码框（64×H×W 全量 softmax 无必要）。
 *
 * @param dfl_maps/cls_maps  每尺度特征图数据指针数组（长度 num_scales）
 * @param map_h/map_w/strides 每尺度特征图尺寸与步长（长度 num_scales）
 * @param num_classes  类别数 nc（cls 通道数，需 >= 1）
 */
inline std::vector<Detection> decodeYoloRknnSplit(
    const float* const* dfl_maps, const float* const* cls_maps,
    const int* map_h, const int* map_w, const int* strides,
    int num_scales, int num_classes,
    float ratio, float dw, float dh,
    int img_w, int img_h,
    float conf_threshold, float iou_threshold)
{
    std::vector<Detection> candidates;
    if (num_scales <= 0 || num_classes <= 0) {
        return candidates;
    }

    for (int s = 0; s < num_scales; ++s) {
        if (!dfl_maps[s] || !cls_maps[s]) continue;
        const int H = map_h[s], W = map_w[s];
        const int hw = H * W;
        const int stride = strides[s];
        const float* dfl = dfl_maps[s];
        const float* cls = cls_maps[s];

        for (int gy = 0; gy < H; ++gy) {
            for (int gx = 0; gx < W; ++gx) {
                const int anchor = gy * W + gx;

                // 类别分数 argmax（通道优先布局: cls + c*hw + anchor）
                int best_cls = 0;
                float score = cls[anchor];
                for (int c = 1; c < num_classes; ++c) {
                    float v = cls[(size_t)c * hw + anchor];
                    if (v > score) { score = v; best_cls = c; }
                }
                if (score < conf_threshold) continue;

                // DFL 解码 ltrb（stride 单位 → 像素）
                float ltrb[4];
                for (int side = 0; side < 4; ++side) {
                    const float* p = dfl + (size_t)(side * 16) * hw + anchor;
                    float mx = p[0];
                    for (int k = 1; k < 16; ++k) mx = std::max(mx, p[(size_t)k * hw]);
                    float sum = 0.0f;
                    float e[16];
                    for (int k = 0; k < 16; ++k) {
                        e[k] = std::exp(p[(size_t)k * hw] - mx);
                        sum += e[k];
                    }
                    float ev = 0.0f;
                    for (int k = 0; k < 16; ++k) ev += k * (e[k] / sum);
                    ltrb[side] = ev * stride;
                }

                const float ax = (gx + 0.5f) * stride;
                const float ay = (gy + 0.5f) * stride;

                Detection det;
                det.class_id = best_cls;
                det.confidence = score;
                det.bbox = cv::Rect2f(ax - ltrb[0], ay - ltrb[1],
                                      ltrb[0] + ltrb[2], ltrb[1] + ltrb[3]);
                candidates.push_back(std::move(det));
            }
        }
    }

    return finalizeDetections(candidates, ratio, dw, dh, img_w, img_h, iou_threshold);
}

} // namespace gpnp
