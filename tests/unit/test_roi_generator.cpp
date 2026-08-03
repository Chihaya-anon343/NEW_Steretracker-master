#include "../framework/TestAssert.hpp"
#include "detection/RoiGenerator.hpp"

#include <opencv2/core.hpp>

#include <vector>

using namespace gpnp;

namespace {

Detection makeDet(int class_id, int x, int y, int w, int h, float conf = 0.9f) {
    Detection d;
    d.class_id = class_id;
    d.confidence = conf;
    d.bbox = cv::Rect2f(static_cast<float>(x), static_cast<float>(y),
                        static_cast<float>(w), static_cast<float>(h));
    return d;
}

RoiGenerator makeGenerator() {
    RoiGenerator::Config cfg;
    cfg.target_class_id = 0;
    cfg.roi_expand_ratio = 0.0f;   // 禁用扩展，便于精确断言面积
    cfg.roi_min_size = 50;
    cfg.dual_trigger_area = 490000;
    return RoiGenerator(cfg);
}

RoiGenerator makeCloseRangeGenerator() {
    auto g = makeGenerator();
    RoiGenerator::CloseRangeConfig cr;
    cr.enabled = true;
    cr.class1_min_area = 100000;
    cr.roi_expand_ratio = 1.5f;
    cr.min_expand_pixels = 0;
    g.setCloseRangeConfig(cr);
    return g;
}

} // namespace

// ============================================================================
// 1) 五状态判定核心：ROI 面积区间 → State 1~4
// ============================================================================

static void test_state1_tiny_area() {
    auto gen = makeGenerator();
    // State 1 远: class0 面积 <= 800  (如 28×28)
    auto group = gen.generateGroup({makeDet(0, 10, 10, 28, 28)}, cv::Size(640, 480));
    TEST_ASSERT(group.valid());
    TEST_ASSERT(!group.is_dual);
    TEST_ASSERT(group.primary.width * group.primary.height <= 800);
    TEST_ASSERT(!group.secondary.valid());
}

static void test_state2_medium_area() {
    auto gen = makeGenerator();
    // State 2 中: 801 ~ 40000  (如 100×100)
    auto group = gen.generateGroup({makeDet(0, 10, 10, 100, 100)}, cv::Size(640, 480));
    TEST_ASSERT(group.valid());
    int area = group.primary.width * group.primary.height;
    TEST_ASSERT(area > 800 && area <= 40000);
    TEST_ASSERT(!group.secondary.valid());
}

static void test_state3_medium_close_area() {
    auto gen = makeGenerator();
    // State 3 中近: 40001 ~ 489999 (如 300×300=90000)
    auto group = gen.generateGroup({makeDet(0, 10, 10, 300, 300)}, cv::Size(640, 480));
    TEST_ASSERT(group.valid());
    int area = group.primary.width * group.primary.height;
    TEST_ASSERT(area > 40000 && area < 490000);
    TEST_ASSERT(!group.secondary.valid());
}

static void test_state4_dual_roi() {
    auto gen = makeGenerator();
    // State 4 近: class0 面积 >= 490000 且存在 class1 (如 720×720)
    auto group = gen.generateGroup({makeDet(0, 0, 0, 720, 720), makeDet(1, 300, 300, 80, 80)},
                                   cv::Size(1280, 960));
    TEST_ASSERT(group.valid());
    TEST_ASSERT(group.is_dual);
    TEST_ASSERT(group.secondary.valid());
    TEST_ASSERT_EQ(group.secondary.width, 80);
    TEST_ASSERT_EQ(group.secondary.height, 80);
}

// ============================================================================
// 2) 边界条件
// ============================================================================

static void test_no_detection_invalid() {
    auto gen = makeGenerator();
    auto group = gen.generateGroup({}, cv::Size(640, 480));
    TEST_ASSERT(!group.valid());
}

static void test_only_class1_invalid_without_close_range() {
    auto gen = makeGenerator(); // close_range 未启用
    auto group = gen.generateGroup({makeDet(1, 10, 10, 200, 200)}, cv::Size(640, 480));
    TEST_ASSERT(!group.valid());
}

static void test_large_class0_without_class1_not_dual() {
    auto gen = makeGenerator();
    // 面积超阈值但无 class1 → 降级为单 ROI (State 3)
    auto group = gen.generateGroup({makeDet(0, 0, 0, 720, 720)}, cv::Size(1280, 960));
    TEST_ASSERT(group.valid());
    TEST_ASSERT(!group.is_dual);
}

static void test_roi_not_above_dual_trigger() {
    auto gen = makeGenerator();
    // class0 面积 699*699 < 490000，即使有 class1 也不触发双 ROI
    auto group = gen.generateGroup({makeDet(0, 0, 0, 699, 699), makeDet(1, 300, 300, 80, 80)},
                                   cv::Size(1280, 960));
    TEST_ASSERT(group.valid());
    TEST_ASSERT(!group.is_dual);
}

// ============================================================================
// 3) State 5 极近：class1 回退 (tryCloseRange)
// ============================================================================

static void test_close_range_recovery() {
    auto gen = makeCloseRangeGenerator();
    // 无 class0，有 class1 且面积 >= class1_min_area
    auto group = gen.tryCloseRange({makeDet(1, 100, 100, 400, 400)}, cv::Size(1280, 960));
    TEST_ASSERT(group.valid());
    // class1 ROI 外扩 1.5 倍 → 400*1.5=600
    TEST_ASSERT_EQ(group.primary.width, 600);
    TEST_ASSERT_EQ(group.primary.height, 600);
}

static void test_close_range_disabled() {
    auto gen = makeGenerator(); // close_range 默认 disabled
    auto group = gen.tryCloseRange({makeDet(1, 100, 100, 400, 400)}, cv::Size(1280, 960));
    TEST_ASSERT(!group.valid());
}

static void test_close_range_below_min_area() {
    auto gen = makeCloseRangeGenerator();
    auto group = gen.tryCloseRange({makeDet(1, 100, 100, 100, 100)}, cv::Size(1280, 960));
    TEST_ASSERT(!group.valid());
}

// ============================================================================
// 4) ROI 几何：扩展 / 最小尺寸 / 边界裁剪
// ============================================================================

static void test_roi_expand_ratio() {
    RoiGenerator::Config cfg;
    cfg.target_class_id = 0;
    cfg.roi_expand_ratio = 0.1f;
    cfg.roi_min_size = 50;
    auto gen = RoiGenerator(cfg);
    auto group = gen.generateGroup({makeDet(0, 100, 100, 100, 100)}, cv::Size(640, 480));
    // 每侧扩展 10% → 100 + 2*10 = 120
    TEST_ASSERT_EQ(group.primary.width, 120);
    TEST_ASSERT_EQ(group.primary.height, 120);
}

static void test_roi_min_size() {
    RoiGenerator::Config cfg;
    cfg.target_class_id = 0;
    cfg.roi_expand_ratio = 0.0f;
    cfg.roi_min_size = 50;
    auto gen = RoiGenerator(cfg);
    // 检测框 20×20 < roi_min_size → 强制 50×50
    auto group = gen.generateGroup({makeDet(0, 100, 100, 20, 20)}, cv::Size(640, 480));
    TEST_ASSERT(group.primary.width >= 50);
    TEST_ASSERT(group.primary.height >= 50);
}

static void test_roi_clamp_to_image() {
    RoiGenerator::Config cfg;
    cfg.target_class_id = 0;
    cfg.roi_expand_ratio = 0.5f;
    cfg.roi_min_size = 50;
    auto gen = RoiGenerator(cfg);
    // 检测框贴近图像右下角，扩展后必须裁剪回图像边界
    auto group = gen.generateGroup({makeDet(0, 600, 400, 100, 100)}, cv::Size(640, 480));
    int right = group.primary.x + group.primary.width;
    int bottom = group.primary.y + group.primary.height;
    TEST_ASSERT(right <= 640);
    TEST_ASSERT(bottom <= 480);
}

// ============================================================================
// 5) 双目 ROI 生成
// ============================================================================

static void test_generate_stereo_right_fallback() {
    auto gen = makeGenerator();
    // 右侧无检测 → 复制左侧 ROI
    auto pair = gen.generateStereo({makeDet(0, 10, 10, 100, 100)}, {},
                                   cv::Size(640, 480), cv::Size(640, 480));
    TEST_ASSERT(pair.first.valid());
    TEST_ASSERT(pair.second.valid());
    TEST_ASSERT_EQ(pair.second.x, pair.first.x);
    TEST_ASSERT_EQ(pair.second.width, pair.first.width);
}

static void test_generate_stereo_group_dual() {
    auto gen = makeGenerator();
    auto pair = gen.generateStereoGroup(
        {makeDet(0, 0, 0, 720, 720), makeDet(1, 300, 300, 80, 80)},
        {makeDet(0, 0, 0, 720, 720), makeDet(1, 300, 300, 80, 80)},
        cv::Size(1280, 960), cv::Size(1280, 960));
    TEST_ASSERT(pair.first.valid());
    TEST_ASSERT(pair.first.is_dual);
}

// ============================================================================
// 注册与入口
// ============================================================================

REGISTER_TEST(test_state1_tiny_area);
REGISTER_TEST(test_state2_medium_area);
REGISTER_TEST(test_state3_medium_close_area);
REGISTER_TEST(test_state4_dual_roi);
REGISTER_TEST(test_no_detection_invalid);
REGISTER_TEST(test_only_class1_invalid_without_close_range);
REGISTER_TEST(test_large_class0_without_class1_not_dual);
REGISTER_TEST(test_roi_not_above_dual_trigger);
REGISTER_TEST(test_close_range_recovery);
REGISTER_TEST(test_close_range_disabled);
REGISTER_TEST(test_close_range_below_min_area);
REGISTER_TEST(test_roi_expand_ratio);
REGISTER_TEST(test_roi_min_size);
REGISTER_TEST(test_roi_clamp_to_image);
REGISTER_TEST(test_generate_stereo_right_fallback);
REGISTER_TEST(test_generate_stereo_group_dual);

int main() {
    return gpnp_test::TestRegistry::instance().runAll();
}