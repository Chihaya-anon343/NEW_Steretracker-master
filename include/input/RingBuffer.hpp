#pragma once

/**
 * @file RingBuffer.hpp
 * @brief 线程安全的环形缓冲区模板（header-only）。
 *
 * 设计模式（来自 ARCHITECTURE.md）：
 *   - 生产者用 unique_lock（独占写），消费者用 shared_lock（共享读）
 *   - 容量上限防止内存无限增长，满时丢弃最老数据
 *   - 维护 dropped_ 计数器用于监控
 */

#include <deque>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace gpnp {
namespace input {

template<typename T>
class RingBuffer {
public:
    /// @param capacity 最大容量，0 表示无限制（不推荐）
    explicit RingBuffer(size_t capacity = 64)
        : capacity_(capacity) {}

    // ---- 写操作（生产者） ----

    /// 写入一个元素。若缓冲区已满，丢弃最老元素。
    void push(T&& item) {
        std::unique_lock lock(mtx_);
        buffer_.push_back(std::move(item));
        if (capacity_ > 0 && buffer_.size() > capacity_) {
            buffer_.pop_front();
            ++dropped_;
        }
    }

    /// 写入一个元素（拷贝版本）。
    void push(const T& item) {
        std::unique_lock lock(mtx_);
        buffer_.push_back(item);
        if (capacity_ > 0 && buffer_.size() > capacity_) {
            buffer_.pop_front();
            ++dropped_;
        }
    }

    // ---- 读操作（消费者） ----

    /// 获取最新元素（不删除）。
    /// @return 若缓冲区非空返回 true。
    bool peekLatest(T& out) const {
        std::shared_lock lock(mtx_);
        if (buffer_.empty()) return false;
        out = buffer_.back();
        return true;
    }

    /// 获取并移除最新元素。
    /// @return 若缓冲区非空返回 true。
    bool popLatest(T& out) {
        std::unique_lock lock(mtx_);
        if (buffer_.empty()) return false;
        out = std::move(buffer_.back());
        buffer_.pop_back();
        return true;
    }

    /// 获取并移除最旧元素。
    /// @return 若缓冲区非空返回 true。
    bool popOldest(T& out) {
        std::unique_lock lock(mtx_);
        if (buffer_.empty()) return false;
        out = std::move(buffer_.front());
        buffer_.pop_front();
        return true;
    }

    /// 排空所有元素（用于批量消费，如 ESKF 预测）。
    std::vector<T> drainAll() {
        std::unique_lock lock(mtx_);
        std::vector<T> result;
        result.reserve(buffer_.size());
        while (!buffer_.empty()) {
            result.push_back(std::move(buffer_.front()));
            buffer_.pop_front();
        }
        return result;
    }

    /// 获取所有元素的快照（不删除）。
    std::vector<T> snapshot() const {
        std::shared_lock lock(mtx_);
        return std::vector<T>(buffer_.begin(), buffer_.end());
    }

    // ---- 状态查询 ----

    size_t size() const {
        std::shared_lock lock(mtx_);
        return buffer_.size();
    }

    bool empty() const {
        std::shared_lock lock(mtx_);
        return buffer_.empty();
    }

    /// 自上次 clear() 以来因满溢而丢弃的元素数。
    size_t dropped() const {
        std::shared_lock lock(mtx_);
        return dropped_;
    }

    void clear() {
        std::unique_lock lock(mtx_);
        buffer_.clear();
        dropped_ = 0;
    }

private:
    std::deque<T> buffer_;
    size_t capacity_;
    size_t dropped_ = 0;
    mutable std::shared_mutex mtx_;
};

} // namespace input
} // namespace gpnp
