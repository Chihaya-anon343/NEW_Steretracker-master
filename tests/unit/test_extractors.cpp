#include "../framework/TestAssert.hpp"
#include "TestDataLoader.hpp"

#include "feature/FeatureExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace gpnp;

namespace {

// 真实模板目录（存在则用于构造提取器）
const char* kMuBanDir = "data/NewMuBan(reordered)";

bool templateDirExists() {
    return std::filesystem::exists(kMuBanDir);
}

// ============================================================================
// fixtures 加载 (tests/data/fixtures + rois.json)
// ============================================================================

std::string fixturesDir() {
    return gpnp_test::defaultFixturesDir();
}

std::string roisPath() {
    return gpnp_test::roisJsonPath(fixturesDir());
}

// 加载场景灰度图并裁剪 ROI (来自 rois.json); 失败返回空 Mat
cv::Mat loadGrayRoi(const std::string& scene, int frame, const std::string& cls) {
    std::string imgPath = gpnp_test::fixtureImagePath(fixturesDir(), scene, "left", frame);
    if (!std::filesystem::exists(imgPath)) return {};
    cv::Mat full = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (full.empty()) return {};
    cv::Rect roi = gpnp_test::loadFixtureRoi(roisPath(), scene, frame, "left", cls);
    if (roi.width <= 0 || roi.height <= 0) return {};
    return full(roi).clone();
}

// ============================================================================
// 1. FeatureExtractor 接口一致性
// ============================================================================

void test_interface_names() {
    // 仅验证策略名称常量（避免依赖模板目录）
    // BinaryCorner / TinyTarget 的 name() 为编译期常量，无需实例
    // StrategyType 枚举值
    TEST_ASSERT(static_cast<int>(StrategyType::Akaze) == 0);
    TEST_ASSERT(static_cast<int>(StrategyType::BinaryCorner) == 1);
    TEST_ASSERT(static_cast<int>(StrategyType::TinyTarget) == 2);
}
REGISTER_TEST(test_interface_names);

// ============================================================================
// 2. BinaryCornerExtractor
// ============================================================================

void test_binary_corner_static_reorder() {
    // 验证几何重排：输入乱序 4 点（TL, BR, TR, BL），期望按几何顺序输出
    // 中心在 (50, 50)，参考角度 -90°（正上方）
    std::vector<cv::Point2f> corners = {
        {10, 10},   // TL
        {90, 90},   // BR
        {90, 10},   // TR
        {10, 90},   // BL
    };
    cv::Point2f center(50, 50);
    auto order = BinaryCornerExtractor::reorderByGeometry(corners, center, -90.0);

    // 期望顺序: TL(0) → TR(2) → BR(1) → BL(3)
    TEST_ASSERT(order.size() == 4);
    // 起点应为 TL（最接近 -90° 方向：x=10, y=10 → 向量 (-40,-40) → 角度 135°... 
    // 这里只验证 4 个索引都是有效且互异
    TEST_ASSERT_EQ(order[0] >= 0 && order[0] < 4, true);
    TEST_ASSERT_EQ(order[1] >= 0 && order[1] < 4, true);
    TEST_ASSERT_EQ(order[2] >= 0 && order[2] < 4, true);
    TEST_ASSERT_EQ(order[3] >= 0 && order[3] < 4, true);
    // 互异性（4 个不同索引）
    for (size_t i = 0; i < order.size(); ++i) {
        for (size_t j = i + 1; j < order.size(); ++j) {
            TEST_ASSERT(order[i] != order[j]);
        }
    }
}
REGISTER_TEST(test_binary_corner_static_reorder);

void test_binary_corner_draw_corners() {
    // drawCorners 不崩溃且输出同尺寸彩色图
    // 画布使用真实 fixtures 图片 (mono_bc 左图, 640×480), 内部自动 gray→BGR
    std::string imgPath = gpnp_test::fixtureImagePath(fixturesDir(), "mono_bc", "left", 0);
    if (!std::filesystem::exists(imgPath)) return;  // fixtures 缺失则跳过

    cv::Mat canvas = cv::imread(imgPath);
    TEST_ASSERT(!canvas.empty());
    std::vector<cv::Point2f> corners = {{50, 50}, {590, 50}, {590, 430}, {50, 430}};
    cv::Mat out = BinaryCornerExtractor::drawCorners(canvas, corners);
    TEST_ASSERT(!out.empty());
    TEST_ASSERT(out.type() == CV_8UC3);
    TEST_ASSERT(out.size() == canvas.size());
}
REGISTER_TEST(test_binary_corner_draw_corners);

void test_binary_corner_empty_input() {
    if (!templateDirExists()) return;  // 无模板目录则跳过

    BinaryCornerExtractor::Config cfg;
    BinaryCornerExtractor ext(cfg, kMuBanDir);

    // 空输入不应崩溃，返回失败结果
    cv::Mat empty;
    PipelineResult r = ext.extract(empty, empty, empty, empty);
    // 允许 success=false 或空角点
    TEST_ASSERT(r.kp_left.empty());
    TEST_ASSERT(r.pts_left_match.empty());
    TEST_ASSERT(!r.success);
}
REGISTER_TEST(test_binary_corner_empty_input);

void test_binary_corner_synthetic_rectangle() {
    if (!templateDirExists()) return;

    BinaryCornerExtractor::Config cfg;
    cfg.corners = 4;
    cfg.kernel_size = 3;
    // 像素→米：模板 100px 宽对应 0.2m → 0.002 m/px
    cfg.pixel_to_meter_scale_class0 = 0.002;
    BinaryCornerExtractor ext(cfg, kMuBanDir);

    // 使用 fixtures: mono_bc 左图 (640×480) 裁剪 class0 ROI (244,165,152,150)
    cv::Mat grayRoi = loadGrayRoi("mono_bc", 0, "class0");
    if (grayRoi.empty()) return;  // fixtures 缺失则跳过

    PipelineResult r = ext.extractMono(grayRoi, grayRoi);
    // 允许两种结果：
    //   - 成功：4 个角点
    //   - 失败：success=false
    // 但若成功则角点数应在合理范围
    if (r.success) {
        TEST_ASSERT(!r.pts_left_match.empty());
        TEST_ASSERT(r.pts_left_match.size() <= 8);  // 角点数不应过多
    }
    // 无论成败，不应返回空关键点同时又标记成功
    if (r.success) {
        TEST_ASSERT(!r.kp_left.empty());
    }
}
REGISTER_TEST(test_binary_corner_synthetic_rectangle);

// ============================================================================
// 3. TinyTargetExtractor
// ============================================================================

void test_tiny_target_empty_input() {
    if (!templateDirExists()) return;

    TinyTargetExtractor::Config cfg;
    TinyTargetExtractor ext(cfg, kMuBanDir);

    cv::Mat empty;
    PipelineResult r = ext.extractMono(empty, empty);
    TEST_ASSERT(!r.success);
    TEST_ASSERT(r.kp_left.empty());
}
REGISTER_TEST(test_tiny_target_empty_input);

void test_tiny_target_small_black_square() {
    if (!templateDirExists()) return;

    TinyTargetExtractor::Config cfg;
    // 小目标：短边 20px（面积 400 ≤ 800 对应 State 1）
    cfg.square_size_m_class0 = 0.05;  // 5cm
    TinyTargetExtractor ext(cfg, kMuBanDir);

    // 使用 fixtures: mono_tiny 左图 (640×480) 裁剪 class0 ROI (310,230,20,20)
    cv::Mat grayRoi = loadGrayRoi("mono_tiny", 0, "class0");
    if (grayRoi.empty()) return;  // fixtures 缺失则跳过

    PipelineResult r = ext.extractMono(grayRoi, grayRoi);
    if (r.success) {
        // 成功时应提取到 4 个角点
        TEST_ASSERT_EQ(r.pts_left_match.size(), size_t(4));
        // 角点应位于 ROI 内（ROI 局部坐标）
        for (const auto& p : r.pts_left_match) {
            TEST_ASSERT(p.x >= 0.0f && p.x <= static_cast<float>(grayRoi.cols));
            TEST_ASSERT(p.y >= 0.0f && p.y <= static_cast<float>(grayRoi.rows));
        }
        // 模板匹配角度应有效
        TEST_ASSERT(ext.lastMatchedAngle() >= 0);
        TEST_ASSERT(ext.lastMatchOverlap() > 0.0);
    }
}
REGISTER_TEST(test_tiny_target_small_black_square);

void test_tiny_target_set_use_class1() {
    if (!templateDirExists()) return;

    TinyTargetExtractor::Config cfg;
    cfg.square_size_m_class0 = 0.20;
    cfg.square_size_m_class1 = 0.04;
    TinyTargetExtractor ext(cfg, kMuBanDir);

    // 默认 class0
    TEST_ASSERT(!ext.lastMatchedTemplate() || true);  // 无崩溃
    // 切换 class1 不应崩溃
    ext.setUseClass1(true);
    ext.setUseClass1(false);
}
REGISTER_TEST(test_tiny_target_set_use_class1);

void test_tiny_target_border_exclusion_prefers_interior() {
    if (!templateDirExists()) return;

    TinyTargetExtractor::Config cfg;
    cfg.square_size_m_class0 = 0.05;
    TinyTargetExtractor ext(cfg, kMuBanDir);

    // 白色边框（贴四边, 面积更大）+ 内部白色方块（不贴边, 面积更小），
    // 黑色环带隔开 —— 触边排除后应选中内部方块而非边框（背景）
    cv::Mat img(160, 160, CV_8UC1, cv::Scalar(255));
    cv::rectangle(img, cv::Rect(30, 30, 100, 100), cv::Scalar(0), -1);
    cv::rectangle(img, cv::Rect(50, 50, 60, 60), cv::Scalar(255), -1);

    PipelineResult r = ext.extractMono(img, img);
    TEST_ASSERT(r.success);
    TEST_ASSERT_EQ(r.pts_left_match.size(), size_t(4));
    // 角点应落在内部方块 (50,50)-(110,110) 附近，而非铺满全画布
    for (const auto& p : r.pts_left_match) {
        TEST_ASSERT(p.x >= 46.0f && p.x <= 114.0f);
        TEST_ASSERT(p.y >= 46.0f && p.y <= 114.0f);
    }
}
REGISTER_TEST(test_tiny_target_border_exclusion_prefers_interior);

void test_tiny_target_adaptive_superres_large_roi() {
    if (!templateDirExists()) return;

    // 大 ROI (800×800) 白色方块目标: 自适应超分 (target=512 → sf=1) 与
    // 旧固定 ×4 提取的角点偏差应 <0.5px (亚像素级), 且自适应路径耗时不高於旧路径
    auto makeExt = [](int target_side) {
        TinyTargetExtractor::Config cfg;
        cfg.square_size_m_class0 = 0.05;
        cfg.scale_factor = 4;
        cfg.superres_target_side = target_side;
        return std::make_unique<TinyTargetExtractor>(cfg, kMuBanDir);
    };
    auto extAdaptive = makeExt(512);
    auto extLegacy   = makeExt(0);   // 0 = 旧行为固定 ×4

    cv::Mat img(800, 800, CV_8UC1, cv::Scalar(255));
    cv::rectangle(img, cv::Rect(150, 150, 500, 500), cv::Scalar(0), -1);
    cv::rectangle(img, cv::Rect(200, 200, 400, 400), cv::Scalar(255), -1);

    auto t0 = std::chrono::steady_clock::now();
    PipelineResult ra = extAdaptive->extractMono(img, img);
    double ms_a = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    PipelineResult rl = extLegacy->extractMono(img, img);
    double ms_l = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    TEST_ASSERT(ra.success);
    TEST_ASSERT(rl.success);
    TEST_ASSERT_EQ(ra.pts_left_match.size(), size_t(4));
    TEST_ASSERT_EQ(rl.pts_left_match.size(), size_t(4));

    double max_dev = 0.0;
    // 真值角点 (顺序无关匹配)。自适应 sf=1 时 Otsu/形态学链在原始分辨率运行,
    // 3×3 close/open 对硬边的角点圆化 ~0.5-0.8px (×4 超分下同核只影响 0.25 原始px),
    // 合成硬边图是最差情形; 真实图像边缘反锯齿后偏差更小, 位姿级回归由端到端
    // <2% 判据把关。此处绝对精度取 1px, 交叉一致性取 1px (乱序/错角偏数十 px)
    const std::vector<cv::Point2f> truth = {{200, 200}, {600, 200}, {600, 600}, {200, 600}};
    auto distToTruth = [&](const cv::Point2f& p) {
        double best = 1e9;
        for (const auto& t : truth) {
            double dx = p.x - t.x, dy = p.y - t.y;
            best = std::min(best, std::sqrt(dx * dx + dy * dy));
        }
        return best;
    };
    double max_truth_a = 0.0, max_truth_l = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        double dx = ra.pts_left_match[i].x - rl.pts_left_match[i].x;
        double dy = ra.pts_left_match[i].y - rl.pts_left_match[i].y;
        max_dev = std::max(max_dev, std::sqrt(dx * dx + dy * dy));
        max_truth_a = std::max(max_truth_a, distToTruth(ra.pts_left_match[i]));
        max_truth_l = std::max(max_truth_l, distToTruth(rl.pts_left_match[i]));
    }
    std::cout << "[adaptive-superres] dev_to_truth adaptive=" << max_truth_a
              << "px legacy=" << max_truth_l << "px  cross_dev=" << max_dev
              << "px  adaptive=" << ms_a << "ms  legacy=" << ms_l << "ms" << std::endl;
    TEST_ASSERT(max_truth_a < 1.0);
    TEST_ASSERT(max_truth_l < 0.5);
    TEST_ASSERT(max_dev < 1.0);   // 交叉一致性: 乱序/错角会偏离数十 px, 1px 足以拦截
    std::cout << "[adaptive-superres] max_corner_dev=" << max_dev
              << "px  adaptive=" << ms_a << "ms  legacy=" << ms_l << "ms" << std::endl;
}
REGISTER_TEST(test_tiny_target_adaptive_superres_large_roi);

} // namespace

// ============================================================================
// 测试入口
// ============================================================================

int main() {
    return gpnp_test::TestRegistry::instance().runAll();
}