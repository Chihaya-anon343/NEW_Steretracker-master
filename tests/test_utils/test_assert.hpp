#pragma once
#ifndef STEREO_TEST_ASSERT_HPP
#define STEREO_TEST_ASSERT_HPP

/**
 * @file test_assert.hpp
 * @brief Steretracker 单元测试自定义断言宏
 *
 * 提供浮点数容差比较、Eigen 矩阵比较、OpenCV Mat 比较等断言。
 * 不依赖任何测试框架（Google Test / Catch2），可在裸 C++ 环境中使用。
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace test_utils {

// ============================================================
// 模板值转字符串
// ============================================================

template <typename T>
std::string to_string(const T& val) {
    if constexpr (std::is_same_v<T, std::string>) {
        return val;
    } else if constexpr (std::is_arithmetic_v<T>) {
        return std::to_string(val);
    } else {
        return "[unknown_type]";
    }
}

inline std::string to_string(const Eigen::Matrix3d& m) {
    // 格式化为简短形式
    char buf[256];
    snprintf(buf, sizeof(buf),
             "[[%.4f, %.4f, %.4f], [%.4f, %.4f, %.4f], [%.4f, %.4f, %.4f]]",
             m(0,0), m(0,1), m(0,2),
             m(1,0), m(1,1), m(1,2),
             m(2,0), m(2,1), m(2,2));
    return std::string(buf);
}

inline std::string to_string(const Eigen::Vector3d& v) {
    char buf[128];
    snprintf(buf, sizeof(buf), "[%.4f, %.4f, %.4f]", v.x(), v.y(), v.z());
    return std::string(buf);
}

inline std::string to_string(const cv::Point2f& p) {
    char buf[64];
    snprintf(buf, sizeof(buf), "(%f, %f)", p.x, p.y);
    return std::string(buf);
}

// ============================================================
// 断言计数
// ============================================================
inline int g_passed = 0;
inline int g_failed = 0;
inline std::string g_current_test = "";

inline void set_test_name(const std::string& name) {
    g_current_test = name;
    g_passed = 0;
    g_failed = 0;
}

inline void print_test_summary() {
    std::cout << "\n  [" << g_current_test << "] ";
    if (g_failed == 0) {
        std::cout << "\033[1;32mPASS\033[0m (" << g_passed << " assertions)";
    } else {
        std::cout << "\033[1;31mFAIL\033[0m (" << g_failed << "/" << (g_passed + g_failed) << " failed)";
    }
    std::cout << std::endl;
}

inline void print_global_summary(int total_passed, int total_failed) {
    std::cout << "\n========================================\n";
    std::cout << "  Total: " << total_passed << " passed, " << total_failed << " failed\n";
    std::cout << "========================================\n";
}

// ============================================================
// 核心断言
// ============================================================

inline void assert_true(bool condition, const std::string& file, int line, const std::string& expr) {
    if (condition) {
        g_passed++;
    } else {
        g_failed++;
        std::cerr << "  \033[1;31mFAIL\033[0m " << file << ":" << line
                  << "  ASSERT_TRUE(" << expr << ")\n";
    }
}

inline void assert_false(bool condition, const std::string& file, int line, const std::string& expr) {
    assert_true(!condition, file, line, "! (" + expr + ")");
}

template <typename T>
void assert_equal(const T& expected, const T& actual,
                  const std::string& file, int line,
                  const std::string& expr_expected, const std::string& expr_actual) {
    if (expected == actual) {
        g_passed++;
    } else {
        g_failed++;
        std::cerr << "  \033[1;31mFAIL\033[0m " << file << ":" << line
                  << "  ASSERT_EQ(" << expr_expected << ", " << expr_actual << ")\n"
                  << "    expected: " << to_string(expected) << "\n"
                  << "    actual:   " << to_string(actual) << "\n";
    }
}

inline void assert_near(double expected, double actual, double tolerance,
                        const std::string& file, int line,
                        const std::string& expr_expected, const std::string& expr_actual) {
    if (std::abs(expected - actual) <= tolerance) {
        g_passed++;
    } else {
        g_failed++;
        std::cerr << "  \033[1;31mFAIL\033[0m " << file << ":" << line
                  << "  ASSERT_NEAR(" << expr_expected << ", " << expr_actual << ", tol=" << tolerance << ")\n"
                  << "    expected: " << expected << "\n"
                  << "    actual:   " << actual << "\n"
                  << "    diff:     " << std::abs(expected - actual) << "\n";
    }
}

inline void assert_throws(void (*fn)(), const std::string& file, int line,
                          const std::string& expr) {
    bool threw = false;
    try {
        fn();
    } catch (...) {
        threw = true;
    }
    if (threw) {
        g_passed++;
    } else {
        g_failed++;
        std::cerr << "  \033[1;31mFAIL\033[0m " << file << ":" << line
                  << "  ASSERT_THROWS(" << expr << ") — no exception thrown\n";
    }
}

// ============================================================
// Eigen 专用断言
// ============================================================

inline void assert_eigen_mat_near(const Eigen::Matrix3d& expected, const Eigen::Matrix3d& actual,
                                   double tolerance, const std::string& file, int line,
                                   const std::string& name_exp, const std::string& name_act) {
    double diff = (expected - actual).norm();
    if (diff <= tolerance) {
        g_passed++;
    } else {
        g_failed++;
        std::cerr << "  \033[1;31mFAIL\033[0m " << file << ":" << line
                  << "  ASSERT_EIGEN_NEAR(" << name_exp << ", " << name_act << ", tol=" << tolerance << ")\n"
                  << "    diff norm: " << diff << "\n"
                  << "    expected:\n" << expected << "\n"
                  << "    actual:\n" << actual << "\n";
    }
}

inline void assert_eigen_vec_near(const Eigen::Vector3d& expected, const Eigen::Vector3d& actual,
                                   double tolerance, const std::string& file, int line,
                                   const std::string& name_exp, const std::string& name_act) {
    double diff = (expected - actual).norm();
    if (diff <= tolerance) {
        g_passed++;
    } else {
        g_failed++;
        std::cerr << "  \033[1;31mFAIL\033[0m " << file << ":" << line
                  << "  ASSERT_EIGEN_NEAR(" << name_exp << ", " << name_act << ", tol=" << tolerance << ")\n"
                  << "    diff norm: " << diff << "\n"
                  << "    expected: " << to_string(expected) << "\n"
                  << "    actual:   " << to_string(actual) << "\n";
    }
}

// ============================================================
// OpenCV 专用断言
// ============================================================

inline void assert_cv_mat_not_empty(const cv::Mat& mat,
                                     const std::string& file, int line,
                                     const std::string& expr) {
    if (!mat.empty()) {
        g_passed++;
    } else {
        g_failed++;
        std::cerr << "  \033[1;31mFAIL\033[0m " << file << ":" << line
                  << "  ASSERT_MAT_NOT_EMPTY(" << expr << ") — mat is empty\n";
    }
}

inline void assert_cv_size_eq(const cv::Mat& mat, int expected_w, int expected_h,
                               const std::string& file, int line,
                               const std::string& expr) {
    if (mat.cols == expected_w && mat.rows == expected_h) {
        g_passed++;
    } else {
        g_failed++;
        std::cerr << "  \033[1;31mFAIL\033[0m " << file << ":" << line
                  << "  ASSERT_MAT_SIZE(" << expr << ", " << expected_w << "x" << expected_h << ")\n"
                  << "    actual: " << mat.cols << "x" << mat.rows << "\n";
    }
}

inline void assert_cv_point_near(const cv::Point2f& expected, const cv::Point2f& actual,
                                  double tolerance, const std::string& file, int line,
                                  const std::string& name_exp, const std::string& name_act) {
    double dist = cv::norm(expected - actual);
    if (dist <= tolerance) {
        g_passed++;
    } else {
        g_failed++;
        std::cerr << "  \033[1;31mFAIL\033[0m " << file << ":" << line
                  << "  ASSERT_POINT_NEAR(" << name_exp << ", " << name_act << ", tol=" << tolerance << ")\n"
                  << "    distance: " << dist << "\n"
                  << "    expected: " << to_string(expected) << "\n"
                  << "    actual:   " << to_string(actual) << "\n";
    }
}

} // namespace test_utils

// ============================================================
// 宏定义
// ============================================================

#define TEST(name)                                                              \
    static void test_##name();                                                  \
    struct TestRunner_##name {                                                  \
        TestRunner_##name() {                                                   \
            ::test_utils::set_test_name(#name);                                 \
            test_##name();                                                      \
            ::test_utils::print_test_summary();                                 \
        }                                                                       \
    } _runner_##name;                                                           \
    static void test_##name()

#define ASSERT_TRUE(expr)    ::test_utils::assert_true((expr), __FILE__, __LINE__, #expr)
#define ASSERT_FALSE(expr)   ::test_utils::assert_false((expr), __FILE__, __LINE__, #expr)
#define ASSERT_EQ(a, b)      ::test_utils::assert_equal((a), (b), __FILE__, __LINE__, #a, #b)
#define ASSERT_NEAR(a, b, tol) ::test_utils::assert_near((a), (b), (tol), __FILE__, __LINE__, #a, #b)
#define ASSERT_THROWS(expr)  ::test_utils::assert_throws([](){ expr; }, __FILE__, __LINE__, #expr)
#define ASSERT_EIGEN_MAT_NEAR(exp, act, tol) \
    ::test_utils::assert_eigen_mat_near((exp), (act), (tol), __FILE__, __LINE__, #exp, #act)
#define ASSERT_EIGEN_VEC_NEAR(exp, act, tol) \
    ::test_utils::assert_eigen_vec_near((exp), (act), (tol), __FILE__, __LINE__, #exp, #act)
#define ASSERT_MAT_NOT_EMPTY(m) ::test_utils::assert_cv_mat_not_empty((m), __FILE__, __LINE__, #m)
#define ASSERT_MAT_SIZE(m, w, h) ::test_utils::assert_cv_size_eq((m), (w), (h), __FILE__, __LINE__, #m)
#define ASSERT_POINT_NEAR(exp, act, tol) \
    ::test_utils::assert_cv_point_near((exp), (act), (tol), __FILE__, __LINE__, #exp, #act)

#endif // STEREO_TEST_ASSERT_HPP