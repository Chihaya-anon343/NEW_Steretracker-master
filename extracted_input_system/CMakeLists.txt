cmake_minimum_required(VERSION 3.16)
project(sensor_input_system LANGUAGES CXX)

# ============================================================
# 可复用传感器输入系统 — 提取自 VNAL 项目 (NUAA, 2026)
# ============================================================
# 包含两个独立模块：
#   1. CanSocket   — Linux SocketCAN C++ RAII 封装
#   2. TimeSyncUnit — IMU 插值 + 雷达高度加权融合时间同步
#
# 依赖：
#   - CanSocket: Linux 内核 SocketCAN 支持 (CONFIG_CAN=y)
#   - TimeSyncUnit: Eigen3
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CanSocket 库 ----
add_library(can_socket STATIC
    can_socket/can_socket.cpp
)
target_include_directories(can_socket PUBLIC can_socket)
target_link_libraries(can_socket PUBLIC pthread)

# ---- TimeSyncUnit 库 ----
find_package(Eigen3 REQUIRED)

add_library(time_alignment STATIC
    time_alignment/timeAlignment.cpp
)
target_include_directories(time_alignment PUBLIC time_alignment)
target_link_libraries(time_alignment PUBLIC Eigen3::Eigen)
