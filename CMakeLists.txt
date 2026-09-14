cmake_minimum_required(VERSION 3.22)
project(GPNP VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Dependencies
# ============================================================================

# --- OpenCV ---
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    features2d
    calib3d
    video
    highgui
    imgcodecs
)

# --- Eigen3 ---
find_package(Eigen3 REQUIRED)

# --- Threads (Phase 4: ESKF 融合工作线程 std::thread) ---
find_package(Threads REQUIRED)

# --- ONNX Runtime (YOLO detection, required) ---
set(ONNXRUNTIME_ROOT
    "/opt/onnxruntime-linux-x64-gpu-1.20.1"
    CACHE PATH "ONNX Runtime installation path")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}")
    message(FATAL_ERROR
            "ONNX Runtime not found:\n"
            "  ${ONNXRUNTIME_ROOT}\n"
            "Please verify Docker image installation.")
endif()

message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")

# ============================================================================
# Source Files
# ============================================================================

set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

set(SOURCES
    main.cpp
    src/feature/AkazeExtractor.cpp
    src/feature/AkazeGpnpExtractor.cpp
    src/feature/BinaryCornerExtractor.cpp
    src/feature/MadDisparityFilter.cpp
    src/feature/OpticalFlowTracker.cpp
    src/feature/TinyTargetExtractor.cpp
    src/matching/TemplateMatcher.cpp
    src/stereo/StereoProjector.cpp
    src/pose/InitialPnPSolver.cpp
    src/pose/GPnPSolver.cpp
    src/pose/MonoPnPSolver.cpp
    src/utils/PoseUtils.cpp
    src/visualization/Visualizer.cpp
    src/tracker/TrackerBase.cpp
    src/tracker/MonoTracker.cpp
    src/tracker/StereoTracker.cpp
    src/detection/RoiGenerator.cpp
    src/detection/YoloRoiProvider.cpp
    # 输入系统 (Phase 1 + Phase 3 摄像头源)
    src/input/FileStereoSource.cpp
    src/input/DirectoryStereoSource.cpp
    src/input/SequenceSource.cpp
    src/input/CameraSource.cpp
    src/input/InputProvider.cpp
    src/input/SimulatedSensors.cpp
    # ESKF 多源融合 (适配层; eskf/eskf_vio.hpp 为 header-only)
    src/fusion/EskfFusionManager.cpp
)

# ============================================================================
# Executable
# ============================================================================

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}   # eskf/ (header-only 库)
)

# Link OpenCV (works with both modern targets and legacy lib list)
if(OpenCV_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Link Eigen3 (fallback for older CMake without Eigen3::Eigen target)
if(TARGET Eigen3::Eigen)
    target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# Link ONNX Runtime (YOLO detection)
target_link_libraries(${PROJECT_NAME} PRIVATE onnxruntime)

# Link Threads (Phase 4: 融合工作线程 std::thread)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# ============================================================================
# Executable: pose_compare —— 对比 pipeline (YOLO框 → SIFT特征匹配 → PnP)
#
# 不依赖 tracker 体系 (无策略链/退化链):
#   main_compare.cpp + SiftKeypointExtractor + TemplateMatcher + MonoPnPSolver
#   (+ NetworkKeypointExtractor 占位, method="network" 时启用)
# ============================================================================

add_executable(pose_compare
    main_compare.cpp
    src/feature/SiftKeypointExtractor.cpp
    src/feature/NetworkKeypointExtractor.cpp
    src/matching/TemplateMatcher.cpp
    src/pose/MonoPnPSolver.cpp
)

target_include_directories(pose_compare PRIVATE
    ${PROJECT_INCLUDE_DIR}
)

if(OpenCV_FOUND)
    target_include_directories(pose_compare PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(pose_compare PRIVATE ${OpenCV_LIBS})

if(TARGET Eigen3::Eigen)
    target_link_libraries(pose_compare PRIVATE Eigen3::Eigen)
else()
    target_include_directories(pose_compare PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

target_link_libraries(pose_compare PRIVATE onnxruntime Threads::Threads)

# ============================================================================
# Executable: pose_compare2 —— 对比 pipeline 2 (YOLO框 → 特征提取+PnP 占位)
#
# 输入输出与 pose_compare 一致; 特征提取+PnP = AprilTag Python sidecar
# (scripts/apriltag_worker.py, pupil-apriltags, 需容器内 pip 安装)
# ============================================================================

add_executable(pose_compare2
    main_compare2.cpp
    src/feature/AprilTagExtractor.cpp
)

target_include_directories(pose_compare2 PRIVATE
    ${PROJECT_INCLUDE_DIR}
)

if(OpenCV_FOUND)
    target_include_directories(pose_compare2 PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(pose_compare2 PRIVATE ${OpenCV_LIBS})

if(TARGET Eigen3::Eigen)
    target_link_libraries(pose_compare2 PRIVATE Eigen3::Eigen)
else()
    target_include_directories(pose_compare2 PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

target_link_libraries(pose_compare2 PRIVATE onnxruntime Threads::Threads)

# ============================================================================
# Compiler Options
# ============================================================================

if(MSVC)
    foreach(tgt ${PROJECT_NAME} pose_compare pose_compare2)
        target_compile_options(${tgt} PRIVATE
            /W4
            /EHsc
            /O2
            /bigobj
        )
        target_compile_definitions(${tgt} PRIVATE
            _CRT_SECURE_NO_WARNINGS
            NOMINMAX
        )
    endforeach()
else()
    foreach(tgt ${PROJECT_NAME} pose_compare pose_compare2)
        target_compile_options(${tgt} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -O3
            -march=native
        )
    endforeach()
endif()

# ============================================================================
# Install
# ============================================================================

install(TARGETS ${PROJECT_NAME} pose_compare pose_compare2 DESTINATION bin)

# ============================================================================
# Tests (可选, 默认关闭)
#
# 启用方式:
#   cmake .. -DGPNP_BUILD_TESTS=ON
#
# 注意:
#   - 测试目录对 ONNX Runtime 为"可选"依赖, 单元测试无需 YOLO 即可运行
#   - 若本机缺少 ONNX Runtime (如 Windows 开发环境), 建议改为独立构建:
#       cmake -S tests -B build-tests
#     (tests/ 自带 find_package 探测逻辑, 便于跨平台 CI)
# ============================================================================
option(GPNP_BUILD_TESTS "Build unit/integration tests under tests/" OFF)

if(GPNP_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "GPNP Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV:       ${OpenCV_VERSION}")
if(DEFINED EIGEN3_INCLUDE_DIR)
    message(STATUS "  Eigen3:       ${EIGEN3_INCLUDE_DIR}")
elseif(DEFINED EIGEN3_INCLUDE_DIRS)
    message(STATUS "  Eigen3:       ${EIGEN3_INCLUDE_DIRS}")
else()
    message(STATUS "  Eigen3:       (found via target)")
endif()
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_ROOT}")
