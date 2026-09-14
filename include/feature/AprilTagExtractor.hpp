#pragma once

/**
 * @file AprilTagExtractor.hpp
 * @brief AprilTag 特征提取 + 位姿模块 (Python sidecar 版)。
 *
 * Docker 环境无 C++ AprilTag 库, 检测与位姿解算放在常驻 Python 子进程
 * (scripts/apriltag_worker.py, pupil-apriltags) 中完成:
 *   C++ ROI 灰度图 + ROI 局部相机参数 → worker 检测+PnP → 回标签角点与 R/t
 *
 * class0 / class1 各自独立配置标签模板 (家族/ID 白名单/物理边长);
 * 位姿为 pupil-apriltags 自带解算 (tag 系→相机系, mm)。
 */

#include <opencv2/core.hpp>

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace gpnp {

class AprilTagExtractor {
public:
    /// 单个 class 的标签模板配置
    struct ClassTagConfig {
        std::string family = "tag36h11";   ///< 标签家族 (tag36h11 / tag25h9 / ...)
        std::vector<int> tag_ids;          ///< ID 白名单 (空 = 接受任意 ID)
        double tag_size_mm = 100.0;        ///< 黑边外沿物理边长 (mm), 位姿尺度来源
    };

    struct Config {
        std::string python_cmd = "python3";
        std::string worker_script = "scripts/apriltag_worker.py";
        int nthreads = 4;                  ///< worker 检测线程数
        ClassTagConfig class0;
        ClassTagConfig class1;
    };

    struct Result {
        bool success = false;
        std::string message;
        std::vector<cv::Point2f> pts_2d;   ///< 标签角点 (ROI 局部, 可视化用)
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  ///< 标签系→相机系
        Eigen::Vector3d t = Eigen::Vector3d::Zero();      ///< mm
        int tag_id = -1;
        int hamming = -1;
    };

    explicit AprilTagExtractor(const Config& cfg) : cfg_(cfg) {}
    ~AprilTagExtractor();

    AprilTagExtractor(const AprilTagExtractor&) = delete;
    AprilTagExtractor& operator=(const AprilTagExtractor&) = delete;

    /// 启动 worker 子进程并握手; 失败返回 false (err_ 含原因)
    bool initialize();

    /**
     * @brief 在 ROI 上检测该 class 配置的 AprilTag 并解算位姿
     * @param roi      ROI 子图 (BGR 或灰度)
     * @param class_id 0 / 1 (决定用哪套标签模板)
     * @param cx_roi   ROI 局部主点 x = 全图 cx - roi.x (位姿解算用)
     * @param cy_roi   ROI 局部主点 y = 全图 cy - roi.y
     *
     * worker IO 失败时本帧失败并自动停 worker, 下次调用自动重启 (一次)。
     */
    Result detect(const cv::Mat& roi, int class_id,
                  double fx, double fy, double cx_roi, double cy_roi);

    const std::string& lastError() const { return err_; }

private:
    bool spawnWorker();
    void shutdownWorker();
    bool writeAll(int fd, const void* buf, size_t len);
    bool readReplyLine(std::string* line);

    Config cfg_;
    bool worker_alive_ = false;
    int fd_in_ = -1;      ///< 写往 worker stdin
    int fd_out_ = -1;     ///< 读 worker stdout
    long child_pid_ = -1;
    std::string err_;
};

} // namespace gpnp
