# ============================================================================
# Steretracker 单元测试 CMake 配置
#
# 仅依赖 OpenCV + Eigen3，不依赖 ONNX Runtime
# 支持 Windows / Linux 编译
# ============================================================================
cmake_minimum_required(VERSION 3.22)
project(test_stereotracker VERSION 1.0.0 LANGUAGES CXX)

# --- C++ Standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Dependencies (仅 OpenCV + Eigen3，无 ONNX Runtime)
# ============================================================================

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    features2d
    calib3d
    highgui
    imgcodecs
)

find_package(Eigen3 REQUIRED)

# ============================================================================
# 主项目源码路径 (tests/unit/ → 项目根目录)
# ============================================================================
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

set(PROJECT_INCLUDE_DIR "${PROJECT_ROOT}/include")
set(PROJECT_SRC_DIR "${PROJECT_ROOT}/src")

# ============================================================================
# 待测源码 (从主项目编译)
# ============================================================================
set(TEST_SOURCES
    # 共同工具
    "${PROJECT_SRC_DIR}/utils/PoseUtils.cpp"

    # 检测层
    "${PROJECT_SRC_DIR}/detection/RoiGenerator.cpp"
    "${PROJECT_SRC_DIR}/detection/YoloRoiProvider.cpp"

    # 特征提取层
    "${PROJECT_SRC_DIR}/feature/AkazeExtractor.cpp"
    "${PROJECT_SRC_DIR}/feature/AkazeGpnpExtractor.cpp"
    "${PROJECT_SRC_DIR}/feature/BinaryCornerExtractor.cpp"
    "${PROJECT_SRC_DIR}/feature/TinyTargetExtractor.cpp"
    "${PROJECT_SRC_DIR}/feature/OpticalFlowTracker.cpp"
    "${PROJECT_SRC_DIR}/feature/MadDisparityFilter.cpp"

    # 立体视觉
    "${PROJECT_SRC_DIR}/stereo/StereoProjector.cpp"

    # 模板匹配
    "${PROJECT_SRC_DIR}/matching/TemplateMatcher.cpp"

    # 位姿解算
    "${PROJECT_SRC_DIR}/pose/GPnPSolver.cpp"
    "${PROJECT_SRC_DIR}/pose/InitialPnPSolver.cpp"
    "${PROJECT_SRC_DIR}/pose/MonoPnPSolver.cpp"

    # Tracker
    "${PROJECT_SRC_DIR}/tracker/TrackerBase.cpp"
    "${PROJECT_SRC_DIR}/tracker/StereoTracker.cpp"

    # 输入系统
    "${PROJECT_SRC_DIR}/input/FileStereoSource.cpp"
    "${PROJECT_SRC_DIR}/input/DirectoryStereoSource.cpp"
    "${PROJECT_SRC_DIR}/input/SequenceSource.cpp"
    "${PROJECT_SRC_DIR}/input/InputProvider.cpp"

    # 可视化 (Debug 模式输出)
    "${PROJECT_SRC_DIR}/visualization/Visualizer.cpp"
)

# ============================================================================
# 测试用例源文件
# ============================================================================
set(TEST_CASES
    "${CMAKE_CURRENT_SOURCE_DIR}/pose/test_pose_validity.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strategy/test_strategy_chain.cpp"
)

# ============================================================================
# 可执行文件
# ============================================================================
add_executable(${PROJECT_NAME} ${TEST_SOURCES} ${TEST_CASES})

# ============================================================================
# Include 目录
# ============================================================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/.."   # 测试工具目录 (tests/)
)

# ============================================================================
# 链接依赖
# ============================================================================
if(OpenCV_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

if(TARGET Eigen3::Eigen)
    target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# ============================================================================
# 编译选项
# ============================================================================
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4
        /EHsc
        /O2
        /bigobj
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O2
    )
endif()

# ============================================================================
# 测试驱动程序 (CTest 集成)
# ============================================================================
enable_testing()
add_test(NAME test_stereotracker_unit COMMAND ${PROJECT_NAME})

# ============================================================================
# Summary
# ============================================================================
message(STATUS "=== Steretracker Unit Tests ===")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV:       ${OpenCV_VERSION}")
message(STATUS "  Project Root: ${PROJECT_ROOT}")