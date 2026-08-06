#pragma once

/**
 * @file AsyncImageSaver.hpp
 * @brief 异步图像落盘 (M4): 后台线程执行 PNG 编码 + 磁盘 IO,
 *        将写盘成本移出主线程关键路径。
 *
 * 用法:
 *   utils::AsyncImageSaver::write(path, img);   // 入队, 立即返回 (内部深拷贝)
 *   utils::AsyncImageSaver::flush();            // 阻塞直到队列清空 (程序结束前)
 *
 * 线程安全: 单例 + 互斥队列, 可多线程并发 write()。
 * 生命周期: 单例析构时排空队列并 join 写线程, 程序退出不丢图
 *           (main 中显式 flush() 用于确定落盘时序)。
 */

#include <opencv2/imgcodecs.hpp>

#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace gpnp {
namespace utils {

class AsyncImageSaver {
public:
    static AsyncImageSaver& instance() {
        static AsyncImageSaver saver;
        return saver;
    }

    /// 异步保存图像 (与 cv::imwrite 签名兼容)。
    /// 内部深拷贝: 调用方入队后可立即复用/修改原 Mat, 无数据竞争;
    /// 拷贝成本远低于 PNG 编码, 对关键路径影响可忽略。
    static void write(const std::string& path, const cv::Mat& img,
                      const std::vector<int>& params = {}) {
        if (img.empty()) return;
        instance().enqueue(path, img.clone(), params);
    }

    /// 阻塞直到所有已入队图像写完 (帧循环结束/退出前调用)。
    static void flush() { instance().flushInternal(); }

private:
    struct Job {
        std::string path;
        cv::Mat img;
        std::vector<int> params;
    };

    AsyncImageSaver() = default;
    ~AsyncImageSaver() { stop(); }

    AsyncImageSaver(const AsyncImageSaver&) = delete;
    AsyncImageSaver& operator=(const AsyncImageSaver&) = delete;

    void enqueue(std::string path, cv::Mat img, std::vector<int> params) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push_back(Job{std::move(path), std::move(img), std::move(params)});
            ++pending_;
        }
        ensureWorker();
        cv_.notify_one();
    }

    void ensureWorker() {
        std::lock_guard<std::mutex> lock(worker_mtx_);
        if (!worker_ || !worker_->joinable()) {
            worker_ = std::make_unique<std::thread>([this] { workerLoop(); });
        }
    }

    void workerLoop() {
        while (true) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&] { return !queue_.empty() || stop_; });
                if (queue_.empty()) break;   // stop_ 且队列排空 → 退出
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            try {
                cv::imwrite(job.path, job.img, job.params);
            } catch (const std::exception& e) {
                std::cerr << "[AsyncImageSaver] 写图失败: " << job.path
                          << ": " << e.what() << std::endl;
            }
            {
                std::lock_guard<std::mutex> lock(mtx_);
                --pending_;
            }
            cv_.notify_all();   // 唤醒 flush 等待者
        }
    }

    void flushInternal() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return pending_ == 0; });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        std::lock_guard<std::mutex> worker_lock(worker_mtx_);
        if (worker_ && worker_->joinable()) worker_->join();
    }

    std::mutex mtx_;            ///< 保护 queue_ / pending_ / stop_
    std::condition_variable cv_;
    std::deque<Job> queue_;
    int64_t pending_ = 0;
    bool stop_ = false;

    std::mutex worker_mtx_;     ///< 保护 worker_ 创建/join (锁序: mtx_ → worker_mtx_)
    std::unique_ptr<std::thread> worker_;
};

} // namespace utils
} // namespace gpnp
