#include "utils/LiveMjpegServer.hpp"

#include <opencv2/imgcodecs.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace gpnp {

namespace {

/// 完整发送; 失败 (对端断开 / SO_SNDTIMEO 超时) 返回 false。
/// MSG_NOSIGNAL: 对端已关闭时不触发 SIGPIPE (默认行为会杀死进程)
bool sendAll(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

/// 索引页: <img> 嵌 MJPEG 流, onerror 1s 后自动重连
/// (浏览器对断掉的 multipart 流不会自行重连, 裸开 /stream 断流即永久卡在最后一帧)
constexpr char kIndexPage[] =
    "<!doctype html><html><head><meta charset='utf-8'><title>GPNP LiveView</title>"
    "<style>body{margin:0;background:#111;display:flex;align-items:center;"
    "justify-content:center;height:100vh}img{max-width:100%;max-height:100vh}</style>"
    "</head><body><img id='v' src='/stream' alt='stream'>"
    "<script>var v=document.getElementById('v');"
    "v.onerror=function(){setTimeout(function(){v.src='/stream?r='+Date.now()},1000)};"
    "</script></body></html>";

} // namespace

LiveMjpegServer::~LiveMjpegServer() {
    stop();
}

bool LiveMjpegServer::start(const Config& cfg) {
    if (running_.exchange(true)) return true;   // 已在运行
    cfg_ = cfg;

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        running_ = false;
        return false;
    }
    int yes = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<std::uint16_t>(cfg_.port));
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0
        || ::listen(listen_fd_, 4) < 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        running_ = false;
        return false;
    }

    accept_thread_ = std::thread(&LiveMjpegServer::acceptLoop, this);
    encoder_thread_ = std::thread(&LiveMjpegServer::encoderLoop, this);
    return true;
}

void LiveMjpegServer::stop() {
    if (!running_.exchange(false)) return;

    // 1) 关监听 fd 唤醒 accept 线程并 join —— 之后不再有新连接
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (accept_thread_.joinable()) accept_thread_.join();

    // 2) 唤醒全部连接线程并 join; fd 只由所属线程 close (shutdown 仅解除阻塞)
    std::vector<int> fds;
    std::vector<std::thread> threads;
    {
        std::lock_guard<std::mutex> lk(clients_mtx_);
        fds = client_fds_;
        threads = std::move(conn_threads_);
        client_fds_.clear();
    }
    for (int fd : fds) ::shutdown(fd, SHUT_RDWR);
    frame_cv_.notify_all();
    for (auto& th : threads)
        if (th.joinable()) th.join();
    if (encoder_thread_.joinable()) encoder_thread_.join();
}

int LiveMjpegServer::clientCount() const {
    std::lock_guard<std::mutex> lk(clients_mtx_);
    return static_cast<int>(client_fds_.size());
}

void LiveMjpegServer::submitFrame(const cv::Mat& bgr) {
    if (!running_.load() || bgr.empty() || clientCount() == 0) return;
    {
        std::lock_guard<std::mutex> lk(frame_mtx_);
        pending_bgr_ = bgr.clone();   // 主线程只付一次 memcpy, JPEG 编码在 encoder 线程
        ++pending_seq_;
    }
    frame_cv_.notify_all();
}

void LiveMjpegServer::acceptLoop() {
    while (running_.load()) {
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) {
            if (!running_.load()) break;   // stop() 关闭了监听 fd
            continue;
        }
        int yes = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        // 发送超时 3s + 加大发送缓冲: 板端瞬时高负载 (NPU/GStreamer 抢 CPU/带宽)
        // 造成的短阻塞不再立即踢客户端; 真断连 (RST) 仍会立刻失败退出
        timeval tv{};
        tv.tv_sec = 3;
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        // 请求头读取超时: 恶意/无声连接不长期占线程
        timeval rtv{};
        rtv.tv_sec = 2;
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));
        int sndbuf = 512 * 1024;
        ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

        std::lock_guard<std::mutex> lk(clients_mtx_);
        client_fds_.push_back(fd);
        conn_threads_.emplace_back(&LiveMjpegServer::handleConnection, this, fd);
    }
}

void LiveMjpegServer::encoderLoop() {
    const double fps = cfg_.max_fps > 0.1 ? cfg_.max_fps : 0.1;
    const auto min_interval = std::chrono::milliseconds(
        static_cast<long long>(1000.0 / fps));
    auto last_encode = std::chrono::steady_clock::now() - min_interval;
    std::uint64_t consumed = 0;

    while (running_.load()) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lk(frame_mtx_);
            frame_cv_.wait_for(lk, std::chrono::milliseconds(250), [&] {
                return !running_.load() || pending_seq_ != consumed;
            });
            if (!running_.load()) break;
            if (pending_seq_ == consumed) continue;
            frame = pending_bgr_;        // pending_bgr_ 是独立 clone, 锁外编码安全
            consumed = pending_seq_;
        }
        if (clientCount() == 0) continue;   // 观看者已全断, 不编码

        // 按 max_fps 节流; 慢消费靠 take-latest 天然跳帧 (只编码最新, 不积压)
        const auto elapsed = std::chrono::steady_clock::now() - last_encode;
        if (elapsed < min_interval)
            std::this_thread::sleep_for(min_interval - elapsed);
        last_encode = std::chrono::steady_clock::now();

        std::vector<std::uint8_t> jpeg;
        const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, cfg_.jpeg_quality};
        if (!cv::imencode(".jpg", frame, jpeg, params)) continue;

        {
            std::lock_guard<std::mutex> lk(frame_mtx_);
            latest_jpeg_ = std::move(jpeg);
            ++frame_seq_;
        }
        frame_cv_.notify_all();
    }
}

void LiveMjpegServer::handleConnection(int fd) {
    // 读 HTTP 请求头, 只看首行路径 (浏览器请求头 ≤4KB)
    std::string req;
    char buf[1024];
    while (req.find("\r\n\r\n") == std::string::npos && req.size() < 4096) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            removeClient(fd);
            ::close(fd);
            return;
        }
        req.append(buf, static_cast<size_t>(n));
    }

    if (req.rfind("GET /stream", 0) != 0) {
        // 索引页 (任何非 /stream 路径一律给索引页)
        char head[192];
        int n = std::snprintf(head, sizeof(head),
                              "HTTP/1.1 200 OK\r\n"
                              "Content-Type: text/html; charset=utf-8\r\n"
                              "Content-Length: %zu\r\n"
                              "Connection: close\r\n\r\n",
                              sizeof(kIndexPage) - 1);
        if (n > 0) {
            sendAll(fd, head, static_cast<size_t>(n));
            sendAll(fd, kIndexPage, sizeof(kIndexPage) - 1);
        }
        removeClient(fd);
        ::close(fd);
        return;
    }

    streamLoop(fd);
}

void LiveMjpegServer::streamLoop(int fd) {
    static constexpr char kStreamHeader[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=gpnpframe\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n"
        "\r\n";
    if (!sendAll(fd, kStreamHeader, sizeof(kStreamHeader) - 1)) {
        removeClient(fd);
        ::close(fd);
        return;
    }

    std::uint64_t sent_seq = 0;
    while (running_.load()) {
        std::vector<std::uint8_t> jpeg;
        {
            std::unique_lock<std::mutex> lk(frame_mtx_);
            frame_cv_.wait_for(lk, std::chrono::milliseconds(500), [&] {
                return !running_.load() || frame_seq_ != sent_seq;
            });
            if (!running_.load()) break;
            if (frame_seq_ == sent_seq) continue;   // 暂无新帧
            jpeg = latest_jpeg_;                    // 拷贝出来, 锁外发送
            sent_seq = frame_seq_;
        }

        char part[128];
        int n = std::snprintf(part, sizeof(part),
                              "--gpnpframe\r\n"
                              "Content-Type: image/jpeg\r\n"
                              "Content-Length: %zu\r\n\r\n",
                              jpeg.size());
        if (n <= 0
            || !sendAll(fd, part, static_cast<size_t>(n))
            || !sendAll(fd, reinterpret_cast<const char*>(jpeg.data()), jpeg.size())
            || !sendAll(fd, "\r\n", 2)) {
            break;
        }
    }

    removeClient(fd);
    ::close(fd);
}

void LiveMjpegServer::removeClient(int fd) {
    // 先移除后 close: 避免 fd 号被新连接复用后误删他人
    std::lock_guard<std::mutex> lk(clients_mtx_);
    client_fds_.erase(std::remove(client_fds_.begin(), client_fds_.end(), fd),
                      client_fds_.end());
}

} // namespace gpnp
