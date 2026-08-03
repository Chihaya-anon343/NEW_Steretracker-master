#include "../framework/TestAssert.hpp"
#include "common/Config.hpp"
#include "common/Types.hpp"

#include <Eigen/Dense>
#include <string>

using namespace gpnp;

// ============================================================================
// makeStereoCameraParams
// ============================================================================

static Eigen::Matrix3d makeK() {
    Eigen::Matrix3d K;
    K << 1000.0, 0.0, 640.0,
         0.0, 1000.0, 512.0,
         0.0, 0.0, 1.0;
    return K;
}

static void test_stereo_params_valid() {
    auto p = makeStereoCameraParams(makeK(), Eigen::Matrix3d::Identity(), Eigen::Vector3d(-120.0, 0.0, 0.0));
    TEST_ASSERT_NEAR(p.focal_length, 1000.0, 1e-9);
    TEST_ASSERT_NEAR(p.baseline, 120.0, 1e-9);
    TEST_ASSERT((p.K * p.K_inv - Eigen::Matrix3d::Identity()).norm() < 1e-12);
}

static void test_stereo_params_invalid_k() {
    auto K = makeK();
    K(2, 0) = 0.1;
    TEST_ASSERT_THROWS(makeStereoCameraParams(K, Eigen::Matrix3d::Identity(), Eigen::Vector3d(-120.0, 0.0, 0.0)), std::invalid_argument);
}

static void test_stereo_params_invalid_rotation() {
    auto R = Eigen::Matrix3d::Identity();
    R(0, 0) = 2.0;
    TEST_ASSERT_THROWS(makeStereoCameraParams(makeK(), R, Eigen::Vector3d(-120.0, 0.0, 0.0)), std::invalid_argument);
}

static void test_stereo_params_nan() {
    auto K = makeK();
    K(1, 1) = std::numeric_limits<double>::quiet_NaN();
    TEST_ASSERT_THROWS(makeStereoCameraParams(K, Eigen::Matrix3d::Identity(), Eigen::Vector3d(-120.0, 0.0, 0.0)), std::invalid_argument);
}

// ============================================================================
// makeTrackerConfig
// ============================================================================

static void test_tracker_config_valid() {
    auto cfg = makeTrackerConfig(0.5, 4, true, 200.0, 150.0, 40000, 800, 10, 0.5);
    TEST_ASSERT_NEAR(cfg.scale, 0.5, 1e-9);
    TEST_ASSERT(cfg.gpnp_min_pts == 4);
    TEST_ASSERT(cfg.use_initial_pnp == true);
    TEST_ASSERT_NEAR(cfg.template_real_width_mm, 200.0, 1e-9);
    TEST_ASSERT_NEAR(cfg.template_real_height_mm, 150.0, 1e-9);
    TEST_ASSERT(cfg.akaze_min_area == 40000);
    TEST_ASSERT(cfg.tiny_max_area == 800);
    TEST_ASSERT(cfg.dual_roi_secondary_expand == 10);
    TEST_ASSERT_NEAR(cfg.dual_roi_akaze_scale, 0.5, 1e-9);
}

static void test_tracker_config_invalid() {
    TEST_ASSERT_THROWS(makeTrackerConfig(0.0), std::invalid_argument);
    TEST_ASSERT_THROWS(makeTrackerConfig(1.5), std::invalid_argument);
    TEST_ASSERT_THROWS(makeTrackerConfig(0.5, 2), std::invalid_argument);
    TEST_ASSERT_THROWS(makeTrackerConfig(0.5, -3), std::invalid_argument);
    TEST_ASSERT_THROWS(makeTrackerConfig(0.5, 4, true, 0.0, 150.0), std::invalid_argument);
    TEST_ASSERT_THROWS(makeTrackerConfig(0.5, 4, true, 200.0, -1.0), std::invalid_argument);
    TEST_ASSERT_THROWS(makeTrackerConfig(0.5, 4, true, 200.0, 150.0, 0), std::invalid_argument);
    TEST_ASSERT_THROWS(makeTrackerConfig(0.5, 4, true, 200.0, 150.0, 40000, 0), std::invalid_argument);
}

// ============================================================================
// makeYoloConfig
// ============================================================================

static void test_yolo_config_valid() {
    auto cfg = makeYoloConfig("best.onnx", DeviceType::CPU, 0.5f, 0.45f, cv::Size(640, 640), 4);
    TEST_ASSERT(cfg.model_path == "best.onnx");
    TEST_ASSERT(cfg.device == DeviceType::CPU);
    TEST_ASSERT_NEAR(cfg.conf_threshold, 0.5f, 1e-6);
    TEST_ASSERT_NEAR(cfg.iou_threshold, 0.45f, 1e-6);
    TEST_ASSERT(cfg.input_size.width == 640 && cfg.input_size.height == 640);
    TEST_ASSERT(cfg.intra_op_threads == 4);
}

static void test_yolo_config_invalid() {
    TEST_ASSERT_THROWS(makeYoloConfig(""), std::invalid_argument);
    TEST_ASSERT_THROWS(makeYoloConfig("m.onnx", DeviceType::CPU, 0.0f), std::invalid_argument);
    TEST_ASSERT_THROWS(makeYoloConfig("m.onnx", DeviceType::CPU, 1.5f), std::invalid_argument);
    TEST_ASSERT_THROWS(makeYoloConfig("m.onnx", DeviceType::CPU, 0.5f, 0.0f), std::invalid_argument);
    TEST_ASSERT_THROWS(makeYoloConfig("m.onnx", DeviceType::CPU, 0.5f, 0.45f, cv::Size(0, 640)), std::invalid_argument);
    TEST_ASSERT_THROWS(makeYoloConfig("m.onnx", DeviceType::CPU, 0.5f, 0.45f, cv::Size(640, 640), 0), std::invalid_argument);
}

REGISTER_TEST(test_stereo_params_valid);
REGISTER_TEST(test_stereo_params_invalid_k);
REGISTER_TEST(test_stereo_params_invalid_rotation);
REGISTER_TEST(test_stereo_params_nan);
REGISTER_TEST(test_tracker_config_valid);
REGISTER_TEST(test_tracker_config_invalid);
REGISTER_TEST(test_yolo_config_valid);
REGISTER_TEST(test_yolo_config_invalid);

// ============================================================================
// 测试入口
// ============================================================================

int main() {
    return gpnp_test::TestRegistry::instance().runAll();
}