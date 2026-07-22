#pragma once

/**
 * @file LogConfig.hpp
 * @brief 全局静默标志 —— 所有模块的 std::cout 打印前检查此值。
 *
 * 由 StereoTracker::setVerboseConsole() 统一设置。
 */

namespace gpnp {

/// 全局终端详细输出开关。false 时仅 main.cpp 输出 [Frame xx] 行。
inline bool g_verbose_console = true;

} // namespace gpnp
