// ============================================================================
// test_input_system.cpp
// 输入系统单元测试: FileStereoSource / DirectoryStereoSource /
//                   SequenceSource / InputProvider / RingBuffer
// ============================================================================
//
// 使用临时目录 + 合成图像 (cv::imwrite) 验证图像源扫描、配对、
// 帧读取与 InputProvider 组装逻辑。
//
// ============================================================================

#include "../framework/TestAssert.hpp"

#include "input/DirectoryStereoSource.hpp"
#include "input/FileStereoSource.hpp"
#include "input/InputProvider.hpp"
#include "input/RingBuffer.hpp"
#include "input/SequenceSource.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace gpnp;
using namespace gpnp::input;

namespace {

// ============================================================================
// 工具函数
// ============================================================================

/// 生成单通道或三通道合成图像。
cv::Mat makeTestImage(int width, int height, int value = 128) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(value, value, value));
    return img;
}

/// 创建唯一临时目录并返回路径。
fs::path makeTempDir(const std::string& tag) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path dir = fs::temp_directory_path() /
        ("gpnp_test_" + tag + "_" + std::to_string(now));
    fs::create_directories(dir);
    return dir;
}

/// 清理临时目录。
void cleanupDir(const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
}

/// 计算当前时间戳 (us) —— 用于验证 nextFrame 输出时间戳非零。
int64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// 返回通过断言数量。
int g_pass = 0;

// ---------------------------------------------------------------------------
// 轻量级断言宏: REQUIRE 失败时抛出 AssertionError, 由 RUN 宏外层/main
// 捕获统计。与 TestAssert.hpp 不同, 该文件使用独立的 RUN/REQUIRE 体系。
// ---------------------------------------------------------------------------
struct AssertionError : public std::exception {
    std::string msg;
    explicit AssertionError(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (cond) {                                                            \
            ++g_pass;                                                          \
        } else {                                                               \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__            \
                      << "  REQUIRE(" #cond ")" << std::endl;                   \
            throw AssertionError("REQUIRE(" #cond ") 失败");                    \
        }                                                                      \
    } while (0)

#define REQUIRE_FALSE(cond) REQUIRE(!(cond))

#define REQUIRE_EQUAL(a, b)                                                    \
    do {                                                                       \
        auto&& _va = (a);                                                      \
        auto&& _vb = (b);                                                      \
        if (_va == _vb) {                                                      \
            ++g_pass;                                                          \
        } else {                                                               \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__            \
                      << "  REQUIRE_EQUAL(" #a ", " #b ")" << std::endl;        \
            throw AssertionError("REQUIRE_EQUAL(" #a ", " #b ") 失败");         \
        }                                                                      \
    } while (0)

#define PASS(msg) do { (void)(msg); } while (0)

#define RUN(name, body)                                      \
    do {                                                     \
        std::cout << "  [RUN ] " << name << std::endl;      \
        std::cout << std::unitbuf;                           \
        body;                                                \
        PASS(std::string("case: ") + name);                 \
        std::cout << "  [PASS] " << name << std::endl;      \
    } while (0)

// ============================================================================
// FileStereoSource
// ============================================================================

void testFileSource() {
    RUN("FileStereoSource: 正常加载一对图像", {
        fs::path dir = makeTempDir("file_ok");
        cv::Mat left = makeTestImage(64, 48, 200);
        cv::Mat right = makeTestImage(64, 48, 100);
        fs::path lp = dir / "L.png";
        fs::path rp = dir / "R.png";
        REQUIRE(cv::imwrite(lp.string(), left));
        REQUIRE(cv::imwrite(rp.string(), right));

        FileStereoSource src;
        REQUIRE(!src.isOpen());
        REQUIRE(src.open(lp.string(), rp.string()));
        REQUIRE(src.isOpen());
        REQUIRE_EQUAL(src.totalFrames(), 1);
        REQUIRE_EQUAL(src.currentFrame(), -1); // 尚未取帧

        cv::Mat L;
        cv::Mat R;
        int64_t ts = 0;
        REQUIRE(src.nextFrame(L, R, ts));
        REQUIRE_FALSE(L.empty());
        REQUIRE_FALSE(R.empty());
        REQUIRE_EQUAL(L.cols, 64);
        REQUIRE_EQUAL(L.rows, 48);
        REQUIRE(ts > 0);
        REQUIRE_EQUAL(src.currentFrame(), 0);

        // File 源多次 nextFrame 返回同一帧（warm-start 兼容）
        cv::Mat L2;
        cv::Mat R2;
        int64_t ts2 = 0;
        REQUIRE(src.nextFrame(L2, R2, ts2));

        // reset 后重播
        REQUIRE(src.reset());
        REQUIRE_EQUAL(src.currentFrame(), -1);

        src.close();
        REQUIRE_FALSE(src.isOpen());
        cleanupDir(dir);
    });

    RUN("FileStereoSource: 路径无效返回 false", {
        FileStereoSource src;
        REQUIRE_FALSE(src.open("", ""));
        REQUIRE_FALSE(src.open("nonexistent_L.png", "nonexistent_R.png"));
    });

    RUN("FileStereoSource: URI 分号格式", {
        fs::path dir = makeTempDir("file_uri");
        fs::path lp = dir / "L2.png";
        fs::path rp = dir / "R2.png";
        REQUIRE(cv::imwrite(lp.string(), makeTestImage(32, 32)));
        REQUIRE(cv::imwrite(rp.string(), makeTestImage(32, 32)));

        FileStereoSource src;
        REQUIRE(src.open(lp.string() + ";" + rp.string()));
        REQUIRE(src.isOpen());

        cv::Mat L;
        cv::Mat R;
        int64_t ts = 0;
        REQUIRE(src.nextFrame(L, R, ts));
        REQUIRE_FALSE(L.empty());
        cleanupDir(dir);
    });
}

// ============================================================================
// DirectoryStereoSource
// ============================================================================

void testDirectorySource() {
    RUN("DirectoryStereoSource: 扫描配对 + 排序读取", {
        fs::path dir = makeTempDir("dir_ok");
        // 写入编号图像对 left_XXXX.png / right_XXXX.png
        const int N = 3;
        for (int i = 0; i < N; ++i) {
            char name[64];
            std::snprintf(name, sizeof(name), "left_%04d.png", i);
            REQUIRE(cv::imwrite((dir / name).string(),
                                makeTestImage(40 + i, 30 + i, 128)));
            std::snprintf(name, sizeof(name), "right_%04d.png", i);
            REQUIRE(cv::imwrite((dir / name).string(),
                                makeTestImage(40 + i, 30 + i, 64)));
        }

        DirectoryStereoSource src;
        REQUIRE(src.open(dir.string(), "left", "right"));
        REQUIRE(src.isOpen());
        REQUIRE_EQUAL(src.totalFrames(), N);

        cv::Mat L;
        cv::Mat R;
        int64_t ts = 0;
        // 第 1 帧尺寸应为最小 (40×30)
        REQUIRE(src.nextFrame(L, R, ts));
        REQUIRE_EQUAL(L.cols, 40);
        REQUIRE_EQUAL(R.cols, 40);
        REQUIRE_EQUAL(src.currentFrame(), 0);

        // 第 3 帧尺寸应为最大 (42×32)
        REQUIRE(src.nextFrame(L, R, ts));
        REQUIRE(src.nextFrame(L, R, ts));
        REQUIRE_EQUAL(L.cols, 42);
        REQUIRE_EQUAL(src.currentFrame(), 2);

        // 超出末尾
        REQUIRE_FALSE(src.nextFrame(L, R, ts));

        // reset 回放
        REQUIRE(src.reset());
        REQUIRE_EQUAL(src.currentFrame(), 0);
        REQUIRE(src.nextFrame(L, R, ts));
        REQUIRE_EQUAL(L.cols, 40);

        src.close();
        REQUIRE_FALSE(src.isOpen());
        cleanupDir(dir);
    });

    RUN("DirectoryStereoSource: 右图缺失自动跳过", {
        fs::path dir = makeTempDir("dir_skip");
        // left_0000 有配对, left_0001 无配对 → 应只扫描到 1 对
        REQUIRE(cv::imwrite((dir / "left_0000.png").string(), makeTestImage(50, 50)));
        REQUIRE(cv::imwrite((dir / "right_0000.png").string(), makeTestImage(50, 50)));
        REQUIRE(cv::imwrite((dir / "left_0001.png").string(), makeTestImage(50, 50)));

        DirectoryStereoSource src;
        REQUIRE(src.open(dir.string(), "left", "right"));
        REQUIRE_EQUAL(src.totalFrames(), 1);

        cv::Mat L;
        cv::Mat R;
        int64_t ts = 0;
        REQUIRE(src.nextFrame(L, R, ts));
        REQUIRE_FALSE(L.empty());
        REQUIRE_FALSE(src.nextFrame(L, R, ts));
        cleanupDir(dir);
    });

    RUN("DirectoryStereoSource: 目录不存在 / 无匹配均失败", {
        DirectoryStereoSource src;
        REQUIRE_FALSE(src.open("nonexistent_dir_xyz", "left", "right"));

        fs::path dir = makeTempDir("dir_empty");
        DirectoryStereoSource src2;
        REQUIRE_FALSE(src2.open(dir.string(), "left", "right")); // 空目录
        cleanupDir(dir);
    });
}

// ============================================================================
// SequenceSource
// ============================================================================

void testSequenceSource() {
    RUN("SequenceSource: 单目序列扫描 + 右图=左图副本", {
        fs::path dir = makeTempDir("seq_ok");
        const int N = 2;
        for (int i = 0; i < N; ++i) {
            char name[64];
            std::snprintf(name, sizeof(name), "frame_%04d.png", i);
            REQUIRE(cv::imwrite((dir / name).string(),
                                makeTestImage(32 + i, 24 + i, 200)));
        }

        SequenceSource src;
        REQUIRE(src.open(dir.string(), "frame"));
        REQUIRE(src.isOpen());
        REQUIRE_EQUAL(src.totalFrames(), N);
        REQUIRE_EQUAL(src.currentFrame(), -1);

        cv::Mat L;
        cv::Mat R;
        int64_t ts = 0;
        REQUIRE(src.nextFrame(L, R, ts));
        REQUIRE_EQUAL(L.cols, 32);
        REQUIRE_FALSE(R.empty());
        REQUIRE_EQUAL(R.cols, 32);   // 右图 = 左图副本
        REQUIRE_EQUAL(L.type(), R.type());

        REQUIRE(src.nextFrame(L, R, ts));
        REQUIRE_EQUAL(L.cols, 33);
        REQUIRE_FALSE(src.nextFrame(L, R, ts));

        REQUIRE(src.reset());
        REQUIRE(src.nextFrame(L, R, ts));
        REQUIRE_EQUAL(L.cols, 32);

        src.close();
        REQUIRE_FALSE(src.isOpen());
        cleanupDir(dir);
    });
}

// ============================================================================
// InputProvider
// ============================================================================

InputSystemConfig makeFileConfig(const std::string& left, const std::string& right) {
    InputSystemConfig cfg;
    cfg.image.type = ImageSourceType::File;
    cfg.image.left_path = left;
    cfg.image.right_path = right;
    return cfg;
}

InputSystemConfig makeDirConfig(const std::string& dir) {
    InputSystemConfig cfg;
    cfg.image.type = ImageSourceType::Directory;
    cfg.image.directory_path = dir;
    cfg.image.left_pattern = "left";
    cfg.image.right_pattern = "right";
    return cfg;
}

InputSystemConfig makeSeqConfig(const std::string& dir) {
    InputSystemConfig cfg;
    cfg.image.type = ImageSourceType::Sequence;
    cfg.image.directory_path = dir;
    cfg.image.sequence_pattern = "frame";
    return cfg;
}

void testInputProvider() {
    RUN("InputProvider: File 配置初始化 + 取帧", {
        fs::path dir = makeTempDir("prov_file");
        fs::path lp = dir / "L.png";
        fs::path rp = dir / "R.png";
        REQUIRE(cv::imwrite(lp.string(), makeTestImage(64, 48)));
        REQUIRE(cv::imwrite(rp.string(), makeTestImage(64, 48)));

        InputProvider provider;
        REQUIRE(provider.initialize(makeFileConfig(lp.string(), rp.string())));
        REQUIRE(provider.isOpen());
        REQUIRE_EQUAL(provider.totalFrames(), 1);
        REQUIRE_EQUAL(provider.currentFrame(), 0);

        SensorPacket packet;
        REQUIRE(provider.getNextPacket(packet));
        REQUIRE(packet.valid);
        REQUIRE_FALSE(packet.left_image.empty());
        REQUIRE_FALSE(packet.right_image.empty());
        REQUIRE(packet.timestamp_us > 0);
        REQUIRE_FALSE(packet.imu.has_value());   // 未启用 IMU
        REQUIRE_FALSE(packet.height.has_value()); // 未启用高度计
        REQUIRE_EQUAL(provider.currentFrame(), 1);

        // 到达末尾
        REQUIRE_FALSE(provider.getNextPacket(packet));

        // reset 重播
        REQUIRE(provider.reset());
        REQUIRE(provider.getNextPacket(packet));
        REQUIRE(packet.valid);

        cleanupDir(dir);
    });

    RUN("InputProvider: Directory 配置扫描", {
        fs::path dir = makeTempDir("prov_dir");
        for (int i = 0; i < 2; ++i) {
            char name[64];
            std::snprintf(name, sizeof(name), "left_%04d.png", i);
            REQUIRE(cv::imwrite((dir / name).string(), makeTestImage(40 + i, 30 + i)));
            std::snprintf(name, sizeof(name), "right_%04d.png", i);
            REQUIRE(cv::imwrite((dir / name).string(), makeTestImage(40 + i, 30 + i)));
        }

        InputProvider provider;
        REQUIRE(provider.initialize(makeDirConfig(dir.string())));
        REQUIRE_EQUAL(provider.totalFrames(), 2);

        SensorPacket p1;
        SensorPacket p2;
        REQUIRE(provider.getNextPacket(p1));
        REQUIRE_EQUAL(p1.left_image.cols, 40);
        REQUIRE(provider.getNextPacket(p2));
        REQUIRE_EQUAL(p2.left_image.cols, 41);
        REQUIRE_FALSE(provider.getNextPacket(p2));
        cleanupDir(dir);
    });

    RUN("InputProvider: Sequence 配置 (单目)", {
        fs::path dir = makeTempDir("prov_seq");
        REQUIRE(cv::imwrite((dir / "frame_0000.png").string(),
                            makeTestImage(30, 30)));

        InputProvider provider;
        REQUIRE(provider.initialize(makeSeqConfig(dir.string())));

        SensorPacket packet;
        REQUIRE(provider.getNextPacket(packet));
        REQUIRE(packet.valid);
        REQUIRE_FALSE(packet.left_image.empty());
        REQUIRE_FALSE(packet.right_image.empty());
        cleanupDir(dir);
    });

    RUN("InputProvider: 无效配置初始化失败", {
        InputProvider provider;
        REQUIRE_FALSE(provider.initialize(makeFileConfig("no_L.png", "no_R.png")));
        REQUIRE_FALSE(provider.isOpen());
    });
}

// ============================================================================
// RingBuffer
// ============================================================================

void testRingBuffer() {
    RUN("RingBuffer: 写入/读取顺序", {
        RingBuffer<int> buf(4);
        REQUIRE_EQUAL(buf.capacity(), 4);
        REQUIRE(buf.empty());
        REQUIRE_FALSE(buf.full());

        buf.push(10);
        buf.push(20);
        buf.push(30);
        REQUIRE_EQUAL(buf.size(), 3);

        int v = 0;
        REQUIRE(buf.popOldest(v));
        REQUIRE_EQUAL(v, 10);
        REQUIRE(buf.popOldest(v));
        REQUIRE_EQUAL(v, 20);
        REQUIRE(buf.popOldest(v));
        REQUIRE_EQUAL(v, 30);
        REQUIRE_FALSE(buf.popOldest(v));
        REQUIRE(buf.empty());
    });

    RUN("RingBuffer: 满缓冲区行为", {
        RingBuffer<int> buf(2);
        buf.push(1);
        buf.push(2);
        REQUIRE(buf.full());
        // 覆盖最旧元素 (FIFO): push 满时丢弃最老元素并计数 dropped
        buf.push(3);
        REQUIRE_EQUAL(buf.dropped(), 1);
        REQUIRE_EQUAL(buf.size(), 2);

        // 清空
        buf.clear();
        REQUIRE(buf.empty());
        REQUIRE_EQUAL(buf.dropped(), 0);
    });
}

} // anonymous namespace

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << "=== 输入系统单元测试 ===" << std::endl;
    int failures = 0;

    auto runAll = [&](const char* name, void (*fn)()) {
        try {
            fn();  // 成功时 RUN 宏已打印 [PASS]
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] " << name << ": " << e.what() << std::endl;
            ++failures;
        }
    };

    runAll("testFileSource", testFileSource);
    runAll("testDirectorySource", testDirectorySource);
    runAll("testSequenceSource", testSequenceSource);
    runAll("testInputProvider", testInputProvider);
    runAll("testRingBuffer", testRingBuffer);

    std::cout << "断言通过 " << g_pass << " 次, 失败 " << failures << " 组"
              << std::endl;
    return failures == 0 ? 0 : 1;
}