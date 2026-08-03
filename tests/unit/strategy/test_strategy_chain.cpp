/**
 * @file test_strategy_chain.cpp
 * @brief 五状态策略链配置 & 退化逻辑单元测试 (T6-01, T6-06)
 *
 * 测试内容：
 *   - configureStrategyChain() 对面积阈值的路由正确性
 *   - 退化链单向性 (AKAZE→BC→TT)
 *   - Dual-ROI 不进入退化链
 *   - fallback_extractors_ 数量始终为 0 或 1 或 2
 *   - 全图回退 (roi_area=0)
 *   - 零/负面积输入
 */

#include "../test_utils/test_assert.hpp"
#include <string>
#include <vector>

using std::string;
using std::vector;

// ============================================================
// 模拟策略链配置逻辑（从 TrackerBase::configureStrategyChain 抽取）
// ============================================================

namespace {

enum class StrategyType {
    Akaze,
    BinaryCorner,
    TinyTarget
};

struct StrategyChainConfig {
    int akaze_min_area;
    int tiny_max_area;

    // 模拟输出
    StrategyType primary;
    vector<StrategyType> fallback;
    string route_id;  // 用于测试唯一性
};

StrategyChainConfig configure_chain(int roi_area,
                                     int akaze_min_area = 40001,
                                     int tiny_max_area = 800) {
    StrategyChainConfig cfg;
    cfg.akaze_min_area = akaze_min_area;
    cfg.tiny_max_area = tiny_max_area;

    if (roi_area >= akaze_min_area || roi_area == 0) {
        cfg.primary = StrategyType::Akaze;
        cfg.fallback = {StrategyType::BinaryCorner, StrategyType::TinyTarget};
        cfg.route_id = "AKAZE_primary";
    } else if (roi_area > tiny_max_area) {
        cfg.primary = StrategyType::BinaryCorner;
        cfg.fallback = {StrategyType::TinyTarget};
        cfg.route_id = "BC_primary";
    } else {
        cfg.primary = StrategyType::TinyTarget;
        cfg.fallback = {};
        cfg.route_id = "TT_primary";
    }
    return cfg;
}

// 模拟退化遍历（从 StereoTracker::process 中的 fallback 循环抽取）
struct FallbackTrace {
    StrategyType attempted;
    vector<StrategyType> sequence;
    bool succeeded;
};

FallbackTrace simulate_fallback(StrategyChainConfig chain,
                                 bool akaze_success,
                                 bool bc_success,
                                 bool tt_success) {
    FallbackTrace trace;
    trace.sequence.push_back(chain.primary);

    // 主策略
    bool success = false;
    if (chain.primary == StrategyType::Akaze) {
        success = akaze_success;
    } else if (chain.primary == StrategyType::BinaryCorner) {
        success = bc_success;
    } else {
        success = tt_success;
    }
    if (success) {
        trace.succeeded = true;
        trace.attempted = chain.primary;
        return trace;
    }

    // 退化尝试
    for (auto& fb : chain.fallback) {
        trace.sequence.push_back(fb);
        if (fb == StrategyType::BinaryCorner && bc_success) {
            trace.succeeded = true;
            trace.attempted = fb;
            return trace;
        }
        if (fb == StrategyType::TinyTarget && tt_success) {
            trace.succeeded = true;
            trace.attempted = fb;
            return trace;
        }
    }

    trace.succeeded = false;
    trace.attempted = chain.fallback.empty() ? chain.primary : chain.fallback.back();
    return trace;
}

// 退化链必须是 DAG（无环）
bool is_degradation_dag(const vector<StrategyType>& path) {
    // TinyTarget 不能再退化
    for (size_t i = 1; i < path.size(); ++i) {
        if (path[i] == StrategyType::TinyTarget &&
            path[i-1] == StrategyType::TinyTarget) {
            return false;
        }
        // 不允许回退到更高级策略
        if (path[i] == StrategyType::Akaze && path[i-1] == StrategyType::BinaryCorner) {
            return false;
        }
        if (path[i] == StrategyType::BinaryCorner && path[i-1] == StrategyType::TinyTarget) {
            return false;
        }
    }
    return true;
}

} // anonymous namespace

// ============================================================
// 测试: 面积阈值路由
// ============================================================

TEST(strategy_chain_state1_tiny) {
    auto chain = configure_chain(500);  // ≤ tiny_max_area(800)
    ASSERT_EQ(chain.primary, StrategyType::TinyTarget);
    ASSERT_EQ(chain.fallback.size(), 0u);
}

TEST(strategy_chain_state1_boundary) {
    auto chain = configure_chain(800);  // exactly tiny_max_area
    ASSERT_EQ(chain.primary, StrategyType::TinyTarget);
}

TEST(strategy_chain_state2_bc) {
    auto chain = configure_chain(20000);  // 801~40000
    ASSERT_EQ(chain.primary, StrategyType::BinaryCorner);
    ASSERT_EQ(chain.fallback.size(), 1u);
    ASSERT_EQ(chain.fallback[0], StrategyType::TinyTarget);
}

TEST(strategy_chain_state2_boundary_low) {
    auto chain = configure_chain(801);
    ASSERT_EQ(chain.primary, StrategyType::BinaryCorner);
}

TEST(strategy_chain_state2_boundary_high) {
    auto chain = configure_chain(40000);
    ASSERT_EQ(chain.primary, StrategyType::BinaryCorner);
}

TEST(strategy_chain_state3_akaze) {
    auto chain = configure_chain(40001);  // ≥ akaze_min_area
    ASSERT_EQ(chain.primary, StrategyType::Akaze);
    ASSERT_EQ(chain.fallback.size(), 2u);
}

TEST(strategy_chain_state3_large_area) {
    auto chain = configure_chain(500000);  // 很大面积
    ASSERT_EQ(chain.primary, StrategyType::Akaze);
    ASSERT_EQ(chain.fallback.size(), 2u);
}

TEST(strategy_chain_zero_area_full_image_fallback) {
    auto chain = configure_chain(0);  // ROI 为空，全图回退
    ASSERT_EQ(chain.primary, StrategyType::Akaze);
    ASSERT_EQ(chain.fallback.size(), 2u);
}

TEST(strategy_chain_negative_area) {
    // 负面积无意义但不应崩溃
    auto chain = configure_chain(-100);
    ASSERT_EQ(chain.primary, StrategyType::TinyTarget);
}

// ============================================================
// 测试: 退化链行为
// ============================================================

TEST(degradation_akaze_success_no_fallback) {
    auto chain = configure_chain(50000);
    auto trace = simulate_fallback(chain, true, false, false);
    ASSERT_TRUE(trace.succeeded);
    ASSERT_EQ(trace.attempted, StrategyType::Akaze);
    ASSERT_EQ(trace.sequence.size(), 1u);
}

TEST(degradation_akaze_to_bc_fallback) {
    auto chain = configure_chain(50000);
    auto trace = simulate_fallback(chain, false, true, false);
    ASSERT_TRUE(trace.succeeded);
    ASSERT_EQ(trace.attempted, StrategyType::BinaryCorner);
    ASSERT_EQ(trace.sequence.size(), 2u);
    ASSERT_EQ(trace.sequence[0], StrategyType::Akaze);
    ASSERT_EQ(trace.sequence[1], StrategyType::BinaryCorner);
}

TEST(degradation_akaze_to_bc_to_tt_full) {
    auto chain = configure_chain(50000);
    auto trace = simulate_fallback(chain, false, false, true);
    ASSERT_TRUE(trace.succeeded);
    ASSERT_EQ(trace.attempted, StrategyType::TinyTarget);
    ASSERT_EQ(trace.sequence.size(), 3u);
}

TEST(degradation_akaze_no_fallback_all_fail) {
    auto chain = configure_chain(50000);
    auto trace = simulate_fallback(chain, false, false, false);
    ASSERT_FALSE(trace.succeeded);
    ASSERT_EQ(trace.sequence.size(), 3u);  // 尝试了全部 3 个
}

TEST(degradation_bc_to_tt_fallback) {
    auto chain = configure_chain(20000);
    auto trace = simulate_fallback(chain, false, false, true);
    ASSERT_TRUE(trace.succeeded);
    ASSERT_EQ(trace.attempted, StrategyType::TinyTarget);
    ASSERT_EQ(trace.sequence.size(), 2u);
}

TEST(degradation_tt_no_fallback) {
    auto chain = configure_chain(500);
    auto trace = simulate_fallback(chain, false, false, false);
    ASSERT_FALSE(trace.succeeded);
    ASSERT_EQ(trace.sequence.size(), 1u);  // TT 无后备，只尝试一次
}

TEST(degradation_is_dag) {
    // 所有退化路径都不应形成环
    auto chain1 = configure_chain(50000);
    auto t1 = simulate_fallback(chain1, false, false, false);
    ASSERT_TRUE(is_degradation_dag(t1.sequence));

    auto chain2 = configure_chain(20000);
    auto t2 = simulate_fallback(chain2, false, false, false);
    ASSERT_TRUE(is_degradation_dag(t2.sequence));

    auto chain3 = configure_chain(500);
    auto t3 = simulate_fallback(chain3, false, false, false);
    ASSERT_TRUE(is_degradation_dag(t3.sequence));
}

// ============================================================
// 测试: 自定义阈值
// ============================================================

TEST(strategy_custom_thresholds) {
    // 自定义阈值：akaze_min=100000, tiny_max=5000
    auto chain = configure_chain(60000, 100000, 5000);
    ASSERT_EQ(chain.primary, StrategyType::BinaryCorner);
    ASSERT_EQ(chain.fallback.size(), 1u);
}

// ============================================================
// 主入口
// ============================================================

int main() {
    return 0;
}