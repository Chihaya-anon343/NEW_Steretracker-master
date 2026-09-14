/**
 * @file AprilTagExtractor.cpp
 * @brief AprilTag Python sidecar 的 C++ 端: 子进程管理 + 管道协议 + 回包解析。
 *
 * 协议 (与 scripts/apriltag_worker.py 严格对应):
 *   握手: 配置 JSON 行 → "READY"
 *   请求: JSON 头行 {"class","w","h","fx","fy","cx","cy","img_len"} + 灰度原始字节
 *   回复: 一行 JSON {"ok","n_tags","tag":{id,hamming,corners,R,t_mm,...}|null}
 */

#include "feature/AprilTagExtractor.hpp"

#include <opencv2/imgproc.hpp>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace gpnp {

namespace {

/// 与 MonoPnPSolver 一致的有效位姿判据 (mm)
constexpr double kMinDepthMm = 10.0;
constexpr double kMaxDepthMm = 100000.0;
/// worker 回包读超时 (ms) — 含检测耗时
constexpr int kReplyTimeoutMs = 10000;

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

} // namespace

AprilTagExtractor::~AprilTagExtractor() {
    shutdownWorker();
}

bool AprilTagExtractor::initialize() {
    if (worker_alive_) return true;
    return spawnWorker();
}

bool AprilTagExtractor::spawnWorker() {
    err_.clear();
    shutdownWorker();

    int in_pipe[2], out_pipe[2];   // in_pipe: 父→子 stdin; out_pipe: 子 stdout→父
    if (::pipe(in_pipe) != 0 || ::pipe(out_pipe) != 0) {
        err_ = "pipe() 失败: " + std::string(std::strerror(errno));
        return false;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        err_ = "fork() 失败: " + std::string(std::strerror(errno));
        for (int fd : {in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1]}) ::close(fd);
        return false;
    }
    if (pid == 0) {
        // 子进程: stderr 继承 (worker 日志直达终端), stdin/stdout 接管道
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        for (int fd : {in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1]})
            ::close(fd);
        ::signal(SIGPIPE, SIG_DFL);
        std::string script = cfg_.worker_script;
        std::string cmd = cfg_.python_cmd;
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(cmd.c_str()));
        argv.push_back(const_cast<char*>(script.c_str()));
        argv.push_back(nullptr);
        ::execvp(argv[0], argv.data());
        std::cerr << "[AprilTagExtractor] execvp 失败: " << argv[0] << " "
                  << argv[1] << ": " << std::strerror(errno) << std::endl;
        ::_exit(127);
    }

    ::close(in_pipe[0]);
    ::close(out_pipe[1]);
    fd_in_ = in_pipe[1];
    fd_out_ = out_pipe[0];
    child_pid_ = pid;

    // --- 握手: 发配置, 等 READY ---
    std::ostringstream cfg_json;
    cfg_json << "{"
             << "\"nthreads\":" << cfg_.nthreads;
    for (int cls = 0; cls < 2; ++cls) {
        const ClassTagConfig& c = cls == 0 ? cfg_.class0 : cfg_.class1;
        cfg_json << ",\"class" << cls << "\":{"
                 << "\"family\":\"" << jsonEscape(c.family) << "\""
                 << ",\"tag_ids\":[";
        for (size_t i = 0; i < c.tag_ids.size(); ++i) {
            if (i) cfg_json << ",";
            cfg_json << c.tag_ids[i];
        }
        cfg_json << "]"
                 << ",\"tag_size_mm\":" << c.tag_size_mm
                 << "}";
    }
    cfg_json << "}\n";
    const std::string payload = cfg_json.str();
    if (!writeAll(fd_in_, payload.data(), payload.size())) {
        err_ = "握手写入失败: " + err_;
        shutdownWorker();
        return false;
    }
    std::string line;
    if (!readReplyLine(&line)) {
        err_ = "握手无响应: " + err_;
        shutdownWorker();
        return false;
    }
    if (line != "READY") {
        err_ = "握手异常响应: " + line.substr(0, 200);
        shutdownWorker();
        return false;
    }
    worker_alive_ = true;
    return true;
}

void AprilTagExtractor::shutdownWorker() {
    if (fd_in_ >= 0) { ::close(fd_in_); fd_in_ = -1; }
    if (fd_out_ >= 0) { ::close(fd_out_); fd_out_ = -1; }
    if (child_pid_ > 0) {
        // 关 stdin 后 worker 读到 EOF 自然退出; 宽限后 SIGKILL 兜底
        for (int i = 0; i < 10; ++i) {
            int status = 0;
            const pid_t r = ::waitpid(child_pid_, &status, WNOHANG);
            if (r == child_pid_) { child_pid_ = -1; break; }
            if (r < 0) { child_pid_ = -1; break; }
            ::usleep(50 * 1000);
        }
        if (child_pid_ > 0) {
            ::kill(child_pid_, SIGKILL);
            ::waitpid(child_pid_, nullptr, 0);
            child_pid_ = -1;
        }
    }
    worker_alive_ = false;
}

bool AprilTagExtractor::writeAll(int fd, const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    size_t off = 0;
    while (off < len) {
        const ssize_t n = ::write(fd, p + off, len - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            err_ = std::string(std::strerror(errno));
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}

bool AprilTagExtractor::readReplyLine(std::string* line) {
    line->clear();
    ::pollfd pfd{fd_out_, POLLIN, 0};
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        const int remain_ms = kReplyTimeoutMs - static_cast<int>(elapsed);
        if (remain_ms <= 0) {
            err_ = "读超时";
            return false;
        }
        const int pr = ::poll(&pfd, 1, remain_ms);
        if (pr < 0) {
            if (errno == EINTR) continue;
            err_ = std::string(std::strerror(errno));
            return false;
        }
        if (pr == 0) continue;
        char ch = 0;
        const ssize_t n = ::read(fd_out_, &ch, 1);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            err_ = n == 0 ? "worker 已退出 (EOF)" : std::string(std::strerror(errno));
            return false;
        }
        if (ch == '\n') return true;
        if (ch != '\r') line->push_back(ch);
        if (line->size() > 4 * 1024 * 1024) {
            err_ = "回包超长";
            return false;
        }
    }
}

AprilTagExtractor::Result AprilTagExtractor::detect(const cv::Mat& roi, int class_id,
                                                    double fx, double fy,
                                                    double cx_roi, double cy_roi) {
    Result result;
    if (class_id != 0 && class_id != 1) {
        result.message = "非法 class_id: " + std::to_string(class_id);
        return result;
    }
    if (!worker_alive_ && !spawnWorker()) {
        result.message = "worker 不可用: " + err_;
        return result;
    }

    // --- 灰度化 + 连续存储 ---
    cv::Mat gray;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = roi;
    }
    if (gray.type() != CV_8UC1) {
        result.message = "ROI 类型异常 (需 8UC1/BGR)";
        return result;
    }
    if (!gray.isContinuous()) gray = gray.clone();

    // --- 请求: JSON 头 + 原始像素 ---
    std::ostringstream hd;
    hd << "{\"class\":" << class_id
       << ",\"w\":" << gray.cols
       << ",\"h\":" << gray.rows
       << ",\"fx\":" << fx
       << ",\"fy\":" << fy
       << ",\"cx\":" << cx_roi
       << ",\"cy\":" << cy_roi
       << ",\"img_len\":" << (gray.rows * gray.cols)
       << "}\n";
    std::string header = hd.str();
    if (!writeAll(fd_in_, header.data(), header.size()) ||
        !writeAll(fd_in_, gray.data, gray.total())) {
        result.message = "请求写入失败: " + err_ + " (本帧丢弃, 下帧重启 worker)";
        shutdownWorker();
        return result;
    }

    // --- 回复 ---
    std::string reply;
    if (!readReplyLine(&reply)) {
        result.message = "回复读取失败: " + err_ + " (本帧丢弃, 下帧重启 worker)";
        shutdownWorker();
        return result;
    }

    cv::FileStorage fs(reply, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if (!fs.isOpened()) {
        result.message = "回包 JSON 解析失败: " + reply.substr(0, 120);
        return result;
    }
    if (static_cast<int>(fs["ok"]) == 0) {
        std::string e = static_cast<std::string>(fs["error"]);
        result.message = "worker 报错: " + e;
        return result;
    }
    const cv::FileNode tag = fs["tag"];
    if (tag.empty()) {
        result.message = "未检测到标签 (n_tags=0)";
        return result;
    }

    // --- 解析角点 / R / t ---
    const cv::FileNode corners_node = tag["corners"];
    if (!corners_node.isSeq() || corners_node.size() != 4) {
        result.message = "角点数量异常";
        return result;
    }
    for (const auto& c : corners_node)
        result.pts_2d.emplace_back(static_cast<float>(c[0]),
                                   static_cast<float>(c[1]));

    const cv::FileNode R_node = tag["R"];
    const cv::FileNode t_node = tag["t_mm"];
    if (R_node.empty() || t_node.empty()) {
        result.message = "worker 未返回位姿 (pose 估计失败)";
        return result;
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            result.R(i, j) = static_cast<double>(R_node[i][j]);
    for (int i = 0; i < 3; ++i)
        result.t(i) = static_cast<double>(t_node[i]);

    result.tag_id = static_cast<int>(tag["id"]);
    result.hamming = static_cast<int>(tag["hamming"]);

    // --- 有效性判据 (与 MonoPnPSolver 对齐) ---
    if (!result.R.allFinite() || !result.t.allFinite()) {
        result.message = "位姿含非有限值";
        result.pts_2d.clear();
        return result;
    }
    if (result.t(2) <= 0.0) {
        result.message = "t.z = " + std::to_string(result.t(2)) + " ≤ 0";
        result.pts_2d.clear();
        return result;
    }
    const double tn = result.t.norm();
    if (tn < kMinDepthMm || tn > kMaxDepthMm) {
        result.message = "|t| = " + std::to_string(tn) + "mm 超出 [" +
                         std::to_string(kMinDepthMm) + "," +
                         std::to_string(kMaxDepthMm) + "]";
        result.pts_2d.clear();
        return result;
    }

    result.success = true;
    return result;
}

} // namespace gpnp
