#pragma once

/**
 * @file TestAssert.hpp
 * @brief 自研轻量测试断言框架（无第三方依赖）。
 *
 * 通过静态注册器收集测试用例，main() 中统一执行并汇总结果。
 * 用法：
 *   void test_xxx() { TEST_ASSERT(x > 0); }
 *   REGISTER_TEST(test_xxx);
 */

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace gpnp_test {

// ============================================================================
// 断言失败异常
// ============================================================================
struct AssertionFailure {
    std::string message;
    std::string file;
    int line;
};

// ============================================================================
// 测试注册器（静态单例）
// ============================================================================
struct TestCase {
    std::string name;
    std::function<void()> fn;
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    void add(const std::string& name, std::function<void()> fn) {
        cases_.push_back({name, std::move(fn)});
    }

    const std::vector<TestCase>& cases() const { return cases_; }

    // 运行所有测试，返回失败的测试数
    int runAll() {
        int passed = 0;
        int failed = 0;
        std::vector<std::string> failures;

        for (const auto& tc : cases_) {
            try {
                tc.fn();
                ++passed;
                std::cout << "[PASS] " << tc.name << "\n";
            } catch (const AssertionFailure& af) {
                ++failed;
                failures.push_back(tc.name);
                std::cout << "[FAIL] " << tc.name << "\n"
                          << "       " << af.file << ":" << af.line << "\n"
                          << "       " << af.message << "\n";
            } catch (const std::exception& e) {
                ++failed;
                failures.push_back(tc.name);
                std::cout << "[FAIL] " << tc.name << " (exception: " << e.what() << ")\n";
            } catch (...) {
                ++failed;
                failures.push_back(tc.name);
                std::cout << "[FAIL] " << tc.name << " (unknown exception)\n";
            }
        }

        std::cout << "\n=== 测试汇总 ===\n"
                  << "  总计: " << (passed + failed) << "\n"
                  << "  通过: " << passed << "\n"
                  << "  失败: " << failed << "\n";
        if (!failures.empty()) {
            std::cout << "  失败用例:\n";
            for (const auto& f : failures) {
                std::cout << "    - " << f << "\n";
            }
        }
        return failed == 0 ? 0 : 1;
    }

private:
    std::vector<TestCase> cases_;
};

// ============================================================================
// 断言宏
// ============================================================================

#define TEST_ASSERT(cond)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::ostringstream oss_;                                               \
            oss_ << "断言失败: " << #cond;                                         \
            throw gpnp_test::AssertionFailure{oss_.str(), __FILE__, __LINE__};     \
        }                                                                          \
    } while (0)

#define TEST_ASSERT_MSG(cond, msg)                                                 \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::ostringstream oss_;                                               \
            oss_ << "断言失败: " << #cond << " — " << msg;                         \
            throw gpnp_test::AssertionFailure{oss_.str(), __FILE__, __LINE__};     \
        }                                                                          \
    } while (0)

#define TEST_ASSERT_THROWS(expr, ex_type)                                          \
    do {                                                                           \
        bool threw_ = false;                                                       \
        try {                                                                      \
            expr;                                                                  \
        } catch (const ex_type&) {                                                 \
            threw_ = true;                                                         \
        } catch (...) {                                                            \
        }                                                                          \
        if (!threw_) {                                                             \
            std::ostringstream oss_;                                               \
            oss_ << "期望抛出 " << #ex_type << ": " << #expr;                      \
            throw gpnp_test::AssertionFailure{oss_.str(), __FILE__, __LINE__};     \
        }                                                                          \
    } while (0)

// 整型/同类型精确相等
#define TEST_ASSERT_EQ(a, b)                                                       \
    do {                                                                           \
        auto va_ = (a);                                                            \
        auto vb_ = (b);                                                            \
        if (!(va_ == vb_)) {                                                       \
            std::ostringstream oss_;                                               \
            oss_ << "相等断言失败: " << #a << " = " << va_ << ", " << #b           \
                 << " = " << vb_;                                                  \
            throw gpnp_test::AssertionFailure{oss_.str(), __FILE__, __LINE__};     \
        }                                                                          \
    } while (0)

// 浮点近似相等（相对容差）
#define TEST_ASSERT_NEAR(a, b, tol)                                                \
    do {                                                                           \
        double va_ = static_cast<double>(a);                                       \
        double vb_ = static_cast<double>(b);                                       \
        double diff_ = std::fabs(va_ - vb_);                                       \
        double denom_ = std::max(std::fabs(va_), std::fabs(vb_));                  \
        if (!(diff_ <= (tol) * (denom_ > 0.0 ? denom_ : 1.0))) {                   \
            std::ostringstream oss_;                                               \
            oss_ << "数值不等: " << #a << " = " << va_ << ", " << #b               \
                 << " = " << vb_ << ", 容差 = " << (tol);                          \
            throw gpnp_test::AssertionFailure{oss_.str(), __FILE__, __LINE__};     \
        }                                                                          \
    } while (0)

#define REGISTER_TEST(fn)                                                          \
    static bool gpnp_test_reg_##fn = [] {                                          \
        gpnp_test::TestRegistry::instance().add(#fn, &fn);                         \
        return true;                                                               \
    }()

} // namespace gpnp_test