#pragma once

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace gpnp {

/**
 * @brief 内嵌 MJPEG HTTP 服务器 —— 实时预览 (output.live_view)
 *
 * 浏览器打开 http://<运行主机IP>:<port>/ 即可观看 (推荐索引页: 断流自动重连;
 * 直接开 /stream 为裸流, 断流需手动刷新):
 * 位姿成功帧为叠加三维轴的画面, 失败帧为原始输入图 (由调用方组装后 submitFrame)。
 *
 * - JPEG 编码在内部 encoder 线程: 主循环只付一次 Mat 拷贝, 不承担编码耗时
 * - take-latest 语义: 观看端慢则跳帧, 不积压、不阻塞主循环
 * - 无观看者时 submitFrame 直接返回 (零开销)
 * - 仅 Linux/POSIX (板端 SSH / Docker 工具链)
 */
class LiveMjpegServer {
public:
    struct Config {
        int port = 8080;          ///< 监听端口
        int jpeg_quality = 70;    ///< JPEG 质量 [1,100]
        double max_fps = 10.0;    ///< 推送帧率上限 (编码节流)
    };

    LiveMjpegServer() = default;
    ~LiveMjpegServer();

    LiveMjpegServer(const LiveMjpegServer&) = delete;
    LiveMjpegServer& operator=(const LiveMjpegServer&) = delete;

    /// 绑定+监听+启动 accept/encoder 线程; 失败返回 false (主流程可继续无预览运行)
    bool start(const Config& cfg);
    void stop();
    bool running() const { return running_.load(); }
    int clientCount() const;

    /// 提交最新一帧 (BGR); 有人观看才暂存 (clone) 并唤醒编码线程
    void submitFrame(const cv::Mat& bgr);

private:
    void acceptLoop();
    void encoderLoop();
    void handleConnection(int fd);
    void streamLoop(int fd);
    void removeClient(int fd);

    Config cfg_;
    std::atomic<bool> running_{false};
    int listen_fd_{-1};
    std::thread accept_thread_;
    std::thread encoder_thread_;

    mutable std::mutex clients_mtx_;
    std::vector<int> client_fds_;
    std::vector<std::thread> conn_threads_;

    // 编码流水线: submitFrame → pending_bgr_ --encoder线程--> latest_jpeg_ → 各发送线程
    std::mutex frame_mtx_;
    std::condition_variable frame_cv_;
    cv::Mat pending_bgr_;
    std::uint64_t pending_seq_ = 0;
    std::vector<std::uint8_t> latest_jpeg_;
    std::uint64_t frame_seq_ = 0;
};

} // namespace gpnp
