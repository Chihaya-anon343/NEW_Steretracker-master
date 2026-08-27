#include "../framework/TestAssert.hpp"
#include "detection/YoloDecode.hpp"

#include <vector>

using namespace gpnp;

namespace {

// 单预测框（2 类）的测试描述
struct Pred {
    float cx, cy, w, h;
    float c0, c1;   // class0 / class1 分数
};

// 构造 BCN 布局 [4+nc, N] = [6, N] 的原始张量
std::vector<float> makeBcn(const std::vector<Pred>& preds) {
    const size_t n = preds.size();
    std::vector<float> d(6 * n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        d[0 * n + i] = preds[i].cx;
        d[1 * n + i] = preds[i].cy;
        d[2 * n + i] = preds[i].w;
        d[3 * n + i] = preds[i].h;
        d[4 * n + i] = preds[i].c0;
        d[5 * n + i] = preds[i].c1;
    }
    return d;
}

// 构造 BNC 布局 [N, 4+nc] = [N, 6] 的原始张量
std::vector<float> makeBnc(const std::vector<Pred>& preds) {
    const size_t n = preds.size();
    std::vector<float> d(6 * n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float* row = d.data() + i * 6;
        row[0] = preds[i].cx;
        row[1] = preds[i].cy;
        row[2] = preds[i].w;
        row[3] = preds[i].h;
        row[4] = preds[i].c0;
        row[5] = preds[i].c1;
    }
    return d;
}

// 便捷调用：默认 ratio=1, dw=dh=0, conf=0.5, iou=0.45, 图 640x480
std::vector<Detection> run(const std::vector<float>& d, bool bcn,
                           size_t n, size_t nc,
                           float ratio = 1.0f, float dw = 0.0f, float dh = 0.0f,
                           float conf = 0.5f, float iou = 0.45f,
                           int iw = 640, int ih = 480) {
    return decodeYoloOutput(d.data(), bcn, n, nc, ratio, dw, dh, iw, ih, conf, iou);
}

} // namespace

// ============================================================================
// 1) 布局解码：BCN 与 BNC 应得到一致结果
// ============================================================================

static void test_bcn_decode() {
    // 单框 cx=100,cy=100,w=40,h=20, class0=0.9
    // → x1=80, y1=90, x2=120, y2=110
    auto d = makeBcn({{100, 100, 40, 20, 0.9f, 0.1f}});
    auto r = run(d, true, 1, 2);
    TEST_ASSERT_EQ(r.size(), size_t(1));
    TEST_ASSERT_EQ(r[0].class_id, 0);
    TEST_ASSERT_NEAR(r[0].confidence, 0.9f, 1e-5);
    TEST_ASSERT_NEAR(r[0].bbox.x, 80.0f, 1e-4);
    TEST_ASSERT_NEAR(r[0].bbox.y, 90.0f, 1e-4);
    TEST_ASSERT_NEAR(r[0].bbox.width, 40.0f, 1e-4);
    TEST_ASSERT_NEAR(r[0].bbox.height, 20.0f, 1e-4);
}

static void test_bnc_decode() {
    auto d = makeBnc({{100, 100, 40, 20, 0.9f, 0.1f}});
    auto r = run(d, false, 1, 2);
    TEST_ASSERT_EQ(r.size(), size_t(1));
    TEST_ASSERT_EQ(r[0].class_id, 0);
    TEST_ASSERT_NEAR(r[0].bbox.x, 80.0f, 1e-4);
    TEST_ASSERT_NEAR(r[0].bbox.y, 90.0f, 1e-4);
    TEST_ASSERT_NEAR(r[0].bbox.width, 40.0f, 1e-4);
    TEST_ASSERT_NEAR(r[0].bbox.height, 20.0f, 1e-4);
}

// ============================================================================
// 2) 置信度阈值过滤
// ============================================================================

static void test_conf_threshold_filter() {
    // 两个框：0.9 与 0.3；conf=0.5 → 仅保留 0.9
    auto d = makeBcn({{100, 100, 40, 20, 0.9f, 0.0f},
                      {200, 200, 40, 20, 0.3f, 0.0f}});
    auto r = run(d, true, 2, 2, 1.0f, 0.0f, 0.0f, 0.5f);
    TEST_ASSERT_EQ(r.size(), size_t(1));
    TEST_ASSERT_NEAR(r[0].confidence, 0.9f, 1e-5);
}

static void test_all_below_threshold_empty() {
    auto d = makeBcn({{100, 100, 40, 20, 0.4f, 0.1f}});
    auto r = run(d, true, 1, 2, 1.0f, 0.0f, 0.0f, 0.5f);
    TEST_ASSERT(r.empty());
}

// ============================================================================
// 3) NMS
// ============================================================================

static void test_nms_suppress_overlap() {
    // 两框高度重叠 (IoU≈0.68 > 0.45)，高分 0.9 保留，0.8 被抑制
    auto d = makeBcn({{50, 50, 100, 100, 0.9f, 0.0f},
                      {60, 60, 100, 100, 0.8f, 0.0f}});
    auto r = run(d, true, 2, 2, 1.0f, 0.0f, 0.0f, 0.5f, 0.45f);
    TEST_ASSERT_EQ(r.size(), size_t(1));
    TEST_ASSERT_NEAR(r[0].confidence, 0.9f, 1e-5);
}

static void test_nms_keep_distinct() {
    // 两框相距远 (IoU=0)，均保留
    auto d = makeBcn({{50, 50, 100, 100, 0.9f, 0.0f},
                      {250, 250, 100, 100, 0.8f, 0.0f}});
    auto r = run(d, true, 2, 2, 1.0f, 0.0f, 0.0f, 0.5f, 0.45f);
    TEST_ASSERT_EQ(r.size(), size_t(2));
}

// ============================================================================
// 4) 反向 letterbox 变换
// ============================================================================

static void test_letterbox_inverse() {
    // ratio=0.5, dw=10, dh=20；输入尺度框 cx=60,cy=70,w=20,h=10
    // → 输入尺度 x1=50,y1=65,x2=70,y2=75
    // → 反变换 x1'=(50-10)/0.5=80, y1'=(65-20)/0.5=90, w'=40, h'=20
    auto d = makeBcn({{60, 70, 20, 10, 0.9f, 0.0f}});
    auto r = run(d, true, 1, 2, 0.5f, 10.0f, 20.0f);
    TEST_ASSERT_EQ(r.size(), size_t(1));
    TEST_ASSERT_NEAR(r[0].bbox.x, 80.0f, 1e-3);
    TEST_ASSERT_NEAR(r[0].bbox.y, 90.0f, 1e-3);
    TEST_ASSERT_NEAR(r[0].bbox.width, 40.0f, 1e-3);
    TEST_ASSERT_NEAR(r[0].bbox.height, 20.0f, 1e-3);
}

// ============================================================================
// 5) clamp 到图像边界
// ============================================================================

static void test_clamp_to_image() {
    // 框中心为负 (cx=-50,cy=-50,w=200,h=200) → x1=-150 被 clamp 到 0
    auto d = makeBcn({{-50, -50, 200, 200, 0.9f, 0.0f}});
    auto r = run(d, true, 1, 2);
    TEST_ASSERT_EQ(r.size(), size_t(1));
    TEST_ASSERT(r[0].bbox.x >= 0.0f);
    TEST_ASSERT(r[0].bbox.y >= 0.0f);
    TEST_ASSERT(r[0].bbox.x + r[0].bbox.width <= 640.0f);
    TEST_ASSERT(r[0].bbox.y + r[0].bbox.height <= 480.0f);
}

// ============================================================================
// 6) 类别 argmax
// ============================================================================

static void test_class_argmax() {
    // class1 分数更高 → class_id == 1
    auto d = makeBcn({{100, 100, 40, 20, 0.1f, 0.9f}});
    auto r = run(d, true, 1, 2);
    TEST_ASSERT_EQ(r.size(), size_t(1));
    TEST_ASSERT_EQ(r[0].class_id, 1);
    TEST_ASSERT_NEAR(r[0].confidence, 0.9f, 1e-5);
}

// ============================================================================
// 7) 边界：空数据指针
// ============================================================================

static void test_null_data_empty() {
    auto r = decodeYoloOutput(nullptr, true, 10, 2, 1.0f, 0.0f, 0.0f,
                             640, 480, 0.5f, 0.45f);
    TEST_ASSERT(r.empty());
}

// ============================================================================
// 注册与入口
// ============================================================================

REGISTER_TEST(test_bcn_decode);
REGISTER_TEST(test_bnc_decode);
REGISTER_TEST(test_conf_threshold_filter);
REGISTER_TEST(test_all_below_threshold_empty);
REGISTER_TEST(test_nms_suppress_overlap);
REGISTER_TEST(test_nms_keep_distinct);
REGISTER_TEST(test_letterbox_inverse);
REGISTER_TEST(test_clamp_to_image);
REGISTER_TEST(test_class_argmax);
REGISTER_TEST(test_null_data_empty);

int main() {
    return gpnp_test::TestRegistry::instance().runAll();
}
