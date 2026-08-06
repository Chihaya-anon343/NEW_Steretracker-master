# =============================================================================
# tests/CMakeLists.txt — Steretracker 自动测试构建
#
# 用法 (独立建 test 目录, 推荐):
#   mkdir build-tests && cd build-tests
#   cmake .. -DGPNP_BUILD_TESTS=ON
#   cmake --build . --config Release
#
# 或在根项目生成时启用 (根 CMakeLists.txt 末尾):
#   cmake .. -DGPNP_BUILD_TESTS=ON
#
# 依赖策略 (比主程序更宽松, 便于 CI):
#   - OpenCV / Eigen3   : REQUIRED (核心算法必需)
#   - ONNX Runtime      : 可选
#       * 找到 → 编译含 YOLO 的用例 (test_input_system)
#       * 未找到 → 跳过该用例并打印警告
#
# 数据策略:
#   - 模板目录路径通过 -D 缓存变量注入, 默认指向仓库 data/ 下
#   - test_integration 在模板目录缺失时自动 SKIP, 不视为失败
# =============================================================================

cmake_minimum_required(VERSION 3.22)

# ============================================================================
# C++ Standard (与根项目一致)
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# add_subdirectory 包含时的保护: 根项目已设置, 跳过重复设置
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# ============================================================================
# 依赖查找
# ============================================================================

# --- OpenCV (REQUIRED; 组件子集 -- 单元测试不需要 GUI) ---
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    features2d
    calib3d
    video
    imgcodecs
)

# --- Eigen3 (REQUIRED; 兼容 target 与模块两种格式) ---
find_package(Eigen3 REQUIRED)

# --- ONNX Runtime (可选) ---
# 优先使用根项目传入的 ONNXRUNTIME_ROOT; 未传入时尝试自动发现
set(GPNP_ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root")
find_library(GPNP_ONNXRUNTIME_LIB
    NAMES onnxruntime onnxruntime.so
    HINTS ${GPNP_ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64
)
find_path(GPNP_ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${GPNP_ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include
)
set(GPNP_ONNXRUNTIME_FOUND FALSE)
if(GPNP_ONNXRUNTIME_LIB AND GPNP_ONNXRUNTIME_INCLUDE_DIR)
    set(GPNP_ONNXRUNTIME_FOUND TRUE)
endif()

# ============================================================================
# 测试源码
# ============================================================================

set(TEST_PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../
                             ${CMAKE_CURRENT_SOURCE_DIR}/../include
                             ${CMAKE_CURRENT_SOURCE_DIR}/framework
)

# 与主程序共享的实现源码 (不含 main.cpp / YoloRoiProvider.cpp, ONNX 可选)
set(TEST_SHARED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tracker/TrackerBase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tracker/MonoTracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tracker/StereoTracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/feature/AkazeExtractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/feature/AkazeGpnpExtractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/feature/BinaryCornerExtractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/feature/TinyTargetExtractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/feature/MadDisparityFilter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/feature/OpticalFlowTracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/matching/TemplateMatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stereo/StereoProjector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/pose/InitialPnPSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/pose/GPnPSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/pose/MonoPnPSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/PoseUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/visualization/Visualizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/detection/RoiGenerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/input/FileStereoSource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/input/DirectoryStereoSource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/input/SequenceSource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/input/CameraSource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/input/InputProvider.cpp
)

if(GPNP_ONNXRUNTIME_FOUND)
    list(APPEND TEST_SHARED_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/detection/YoloRoiProvider.cpp
    )
endif()

# ============================================================================
# 测试辅助函数: 构建最小链接环境 (复刻根项目 include/link/flag)
# ============================================================================
function(gpnp_add_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${ARG_SOURCES} ${TEST_SHARED_SOURCES})

    target_include_directories(${name} PRIVATE
        ${TEST_PROJECT_INCLUDE_DIR}
    )
    if(OpenCV_FOUND)
        target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
    endif()
    target_link_libraries(${name} PRIVATE ${OpenCV_LIBS})

    if(TARGET Eigen3::Eigen)
        target_link_libraries(${name} PRIVATE Eigen3::Eigen)
    elseif(DEFINED EIGEN3_INCLUDE_DIR)
        target_include_directories(${name} PRIVATE ${EIGEN3_INCLUDE_DIR})
    elseif(DEFINED EIGEN3_INCLUDE_DIRS)
        target_include_directories(${name} PRIVATE ${EIGEN3_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "Eigen3 头文件路径未找到 (设置 EIGEN3_INCLUDE_DIR)")
    endif()

    if(MSVC)
        target_compile_options(${name} PRIVATE /W4 /EHsc /bigobj)
        target_compile_definitions(${name} PRIVATE
            _CRT_SECURE_NO_WARNINGS
            NOMINMAX
        )
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # ONNX 可选链接 (YoloRoiProvider.cpp 属于共享源码, 所有测试目标均需链接)
    if(GPNP_ONNXRUNTIME_FOUND)
        target_include_directories(${name} PRIVATE ${GPNP_ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(${name} PRIVATE ${GPNP_ONNXRUNTIME_LIB})
    endif()

    # 测试数据路径注入 (绝对路径, 保证任意工作目录可运行)
    if(TEST_TEMPLATE_DIR)
        target_compile_definitions(${name} PRIVATE
            GPNP_TEST_TEMPLATE_DIR="${TEST_TEMPLATE_DIR}"
        )
    endif()
    if(TEST_FIXTURES_DIR)
        target_compile_definitions(${name} PRIVATE
            GPNP_TEST_FIXTURES_DIR="${TEST_FIXTURES_DIR}"
        )
    endif()

    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ============================================================================
# 测试目标
# ============================================================================

# 配置工厂函数 (无第三方重依赖)
gpnp_add_test(test_config
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_config.cpp
)

# YOLO ROS 五状态判定 (无 ONNX 依赖)
gpnp_add_test(test_roi_generator
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_roi_generator.cpp
)

# PnP 求解器 (需要 Eigen + OpenCV calib3d)
gpnp_add_test(test_pose_solvers
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_pose_solvers.cpp
)

# 特征提取器 (BinaryCorner / TinyTarget / AKAZE 合成图, 模板依赖自动跳过)
gpnp_add_test(test_extractors
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_extractors.cpp
)

# 端到端冒烟 (合成双眼/单目图 + 手动 ROI; 无模板时 SKIP)
gpnp_add_test(test_integration
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_integration.cpp
)

# 输入系统 (Phase1) — 需要 ONNX, 缺失时跳过
if(GPNP_ONNXRUNTIME_FOUND)
    gpnp_add_test(test_input_system
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_input_system.cpp
    )
    target_link_libraries(test_input_system PRIVATE ${GPNP_ONNXRUNTIME_LIB})
else()
    message(WARNING
        "ONNX Runtime 未找到 (设置 GPNP_ONNXRUNTIME_ROOT) — "
        "跳过 test_input_system")
endif()

# ============================================================================
# 辅助目标
# ============================================================================

# 一键运行全部测试 (ctest 封装)
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_config test_roi_generator test_pose_solvers
            test_extractors test_integration
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "运行 Steretracker 单元与集成测试"
)

# ============================================================================
# 摘要
# ============================================================================
set(GPNP_TEST_LIST "test_config, test_roi_generator, test_pose_solvers, test_extractors, test_integration")
if(GPNP_ONNXRUNTIME_FOUND)
    set(GPNP_TEST_LIST "${GPNP_TEST_LIST}, test_input_system")
endif()
message(STATUS "GPNP 测试配置:")
message(STATUS "  测试可执行: ${GPNP_TEST_LIST}")
message(STATUS "  OpenCV:   ${OpenCV_VERSION}")
if(DEFINED EIGEN3_INCLUDE_DIR)
    message(STATUS "  Eigen3:   ${EIGEN3_INCLUDE_DIR}")
elseif(DEFINED EIGEN3_INCLUDE_DIRS)
    message(STATUS "  Eigen3:   ${EIGEN3_INCLUDE_DIRS}")
else()
    message(STATUS "  Eigen3:   (via Eigen3::Eigen target)")
endif()
if(GPNP_ONNXRUNTIME_FOUND)
    message(STATUS "  ONNX Runtime: found (${GPNP_ONNXRUNTIME_ROOT})")
else()
    message(STATUS "  ONNX Runtime: NOT FOUND (test_input_system 已跳过)")
endif()
