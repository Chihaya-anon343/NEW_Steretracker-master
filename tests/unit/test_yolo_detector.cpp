// ============================================================================
// test_yolo_detector.cpp — YOLO 检测器端到端冒烟测试
//
// 覆盖: YoloDetector (模型加载 / detect 状态码) 与
//       YoloRoiProvider (YOLO → RoiGroup 外观)。
//
// 数据依赖 (缺失即跳过, 不 FAIL, 便于无模型 CI):
//   - 模型: best.onnx (NMS-export 已解码格式 [1,300,6], 2 类)
//   - 图片: tests/data/fixtures/synthetic_akaze/left_000.png (仅冒烟, 不强断言检测结果)
//
// 定位: 冒烟测试 —— 断言模型可加载、detect() 返回合理状态码、外观层不崩溃;
//       不对具体检测框/置信度做强断言 (合成图非真实目标, 结果无确定性保证)。
// ============================================================================

#include "../framework/TestAssert.hpp"

#include "common/Config.hpp"
#include "detection/RoiGenerator.hpp"
#include "detection/YoloDetector.hpp"
#include "detection/YoloRoiProvider.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <vector>

using namespace gpnp;

namespace {

const char* kModelPath = "best.onnx";
const char* kImagePath = "tests/data/fixtures/synthetic_akaze/left_000.png";

bool fileExists(const char* p) {
    return std::filesystem::exists(p);
}

bool modelExists()   { return fileExists(kModelPath); }
bool fixtureExists() { return fileExists(kImagePath); }

YoloConfig makeCfg(const std::string& path) {
    return makeYoloConfig(path, DeviceType::CPU, 0.5f);
}

void skip(const char* why) {
    std::cout << "  [SKIP] " << why << "\n";
}

} // namespace

// ============================================================================
// 1) 模型加载
// ============================================================================

static void test_model_load_success() {
    if (!modelExists()) { skip("模型缺失 (best.onnx)"); return; }
    YoloDetector det(makeCfg(kModelPath));
    TEST_ASSERT(det.isInitialized());
}

static void test_model_missing_throws() {
    // 不存在的路径 → 构造函数抛异常（无需真实模型）
    TEST_ASSERT_THROWS(YoloDetector(makeCfg("nonexistent_model_xyz.onnx")), std::exception);
}

// ============================================================================
// 2) detect() 状态码
// ============================================================================

static void test_detect_empty_image() {
    if (!modelExists()) { skip("模型缺失"); return; }
    YoloDetector det(makeCfg(kModelPath));
    std::vector<Detection> dets;
    Status st = det.detect(cv::Mat(), dets);
    TEST_ASSERT_EQ(static_cast<int>(st), static_cast<int>(Status::EmptyInput));
    TEST_ASSERT(dets.empty());
}

static void test_detect_fixture_no_crash() {
    if (!modelExists())  { skip("模型缺失"); return; }
    if (!fixtureExists()) { skip("fixture 缺失"); return; }

    YoloDetector det(makeCfg(kModelPath));
    cv::Mat img = cv::imread(kImagePath);
    TEST_ASSERT(!img.empty());

    std::vector<Detection> dets;
    Status st = det.detect(img, dets);
    // 空检测也是 Success（无目标时 dets 为空但不报错）
    TEST_ASSERT_EQ(static_cast<int>(st), static_cast<int>(Status::Success));
}

// ============================================================================
// 3) YoloRoiProvider 外观端到端
// ============================================================================

static void test_roi_provider_e2e() {
    if (!modelExists())  { skip("模型缺失"); return; }
    if (!fixtureExists()) { skip("fixture 缺失"); return; }

    RoiGenerator::Config roi_cfg;
    roi_cfg.target_class_id = 0;
    roi_cfg.roi_expand_ratio = 0.0f;
    roi_cfg.roi_min_size = 0;
    roi_cfg.dual_trigger_area = 490000;

    YoloRoiProvider yolo;
    TEST_ASSERT(yolo.initialize(makeCfg(kModelPath), roi_cfg));
    TEST_ASSERT(yolo.isReady());

    cv::Mat img = cv::imread(kImagePath);
    TEST_ASSERT(!img.empty());

    // 单目路径：可能无检测 → invalid，但不应崩溃
    RoiGroup mono = yolo.detectMono(img);
    (void)mono;

    // 双目路径：右图=左图副本，仅冒烟（不崩溃即可）
    auto stereo = yolo.detect(img, img);
    (void)stereo;
}

// ============================================================================
// 注册与入口
// ============================================================================

REGISTER_TEST(test_model_load_success);
REGISTER_TEST(test_model_missing_throws);
REGISTER_TEST(test_detect_empty_image);
REGISTER_TEST(test_detect_fixture_no_crash);
REGISTER_TEST(test_roi_provider_e2e);

int main() {
    return gpnp_test::TestRegistry::instance().runAll();
}
