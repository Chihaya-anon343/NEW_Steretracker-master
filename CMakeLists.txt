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
# Resolution order: -DONNXRUNTIME_ROOT=<path> > auto-detect /opt/onnxruntime-*
# (tolerates variant installs: linux-x64-gpu in Docker, linux-aarch64 on ARM boards)
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime install prefix (empty = auto-detect /opt/onnxruntime-*)")

set(_ORT_ROOT "")
if(ONNXRUNTIME_ROOT)
    if(EXISTS "${ONNXRUNTIME_ROOT}")
        set(_ORT_ROOT "${ONNXRUNTIME_ROOT}")
    else()
        message(WARNING "ONNXRUNTIME_ROOT='${ONNXRUNTIME_ROOT}' does not exist, falling back to auto-detect")
    endif()
endif()

if(NOT _ORT_ROOT)
    file(GLOB _ORT_CANDIDATES "/opt/onnxruntime-*")
    foreach(_d IN LISTS _ORT_CANDIDATES)
        if(EXISTS "${_d}/lib/libonnxruntime.so" AND EXISTS "${_d}/include/onnxruntime_cxx_api.h")
            list(APPEND _ORT_VALID "${_d}")
        endif()
    endforeach()
    list(SORT _ORT_VALID)
    list(LENGTH _ORT_VALID _ORT_N)
    if(_ORT_N EQUAL 0)
        message(FATAL_ERROR
                "ONNX Runtime not found. Searched:\n"
                "  - ONNXRUNTIME_ROOT='${ONNXRUNTIME_ROOT}' (cache)\n"
                "  - /opt/onnxruntime-* (needs lib/libonnxruntime.so + include/onnxruntime_cxx_api.h)\n"
                "Fix: install ONNX Runtime under /opt or pass -DONNXRUNTIME_ROOT=<path>.")
    elseif(_ORT_N GREATER 1)
        list(GET _ORT_VALID -1 _ORT_ROOT)
        message(WARNING "Multiple ONNX Runtime installs under /opt: ${_ORT_VALID}; using '${_ORT_ROOT}'")
    else()
        list(GET _ORT_VALID 0 _ORT_ROOT)
    endif()
endif()

set(ONNXRUNTIME_ROOT "${_ORT_ROOT}")  # shadow cache so later refs use the resolved path
message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")

# Built binaries resolve libonnxruntime.so without LD_LIBRARY_PATH
set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")

# --- RKNN Runtime (Rockchip NPU, 可选: 仅 ARM 板上有, x86 Docker 无) ---
# 探测失败不报错——.rknn 后端禁用, ONNX Runtime 后端不受影响。
set(RKNN_ROOT "" CACHE PATH "RKNN runtime install prefix (containing lib/ and include/)")

find_library(RKNN_RT_LIB
        NAMES rknnrt
        PATHS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib ${RKNN_ROOT}/lib)
find_path(RKNN_INC_DIR
        NAMES rknn_api.h
        PATHS /usr/include /usr/local/include ${RKNN_ROOT}/include
              ${CMAKE_SOURCE_DIR}/third_party/rknpu2/include)

if(RKNN_RT_LIB AND RKNN_INC_DIR)
    add_library(rknnrt SHARED IMPORTED)
    set_target_properties(rknnrt PROPERTIES
            IMPORTED_LOCATION "${RKNN_RT_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${RKNN_INC_DIR}")
    set(GPNP_RKNN_AVAILABLE ON)
    message(STATUS "RKNN: enabled (${RKNN_RT_LIB}, ${RKNN_INC_DIR})")
else()
    set(GPNP_RKNN_AVAILABLE OFF)
    message(STATUS "RKNN: not found, .rknn backend disabled")
endif()

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
# Executable: obb_pipeline —— 最小端到端 OBB 流水线 (pipeline 分支)
#
# YOLO-OBB 检测 → OBB 四边形角点提取 → MonoPnP 位姿解算。
# 不依赖 tracker 体系 (无策略链/退化链):
#   main_obb_pipeline.cpp + ObbCornerExtractor + MonoPnPSolver + YoloDetector(header-only)
# ============================================================================

add_executable(obb_pipeline
    main_obb_pipeline.cpp
    src/feature/ObbCornerExtractor.cpp
    src/pose/MonoPnPSolver.cpp
)

target_include_directories(obb_pipeline PRIVATE
    ${PROJECT_INCLUDE_DIR}
)

if(OpenCV_FOUND)
    target_include_directories(obb_pipeline PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(obb_pipeline PRIVATE ${OpenCV_LIBS})

if(TARGET Eigen3::Eigen)
    target_link_libraries(obb_pipeline PRIVATE Eigen3::Eigen)
else()
    target_include_directories(obb_pipeline PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

target_link_libraries(obb_pipeline PRIVATE onnxruntime Threads::Threads)

# ============================================================================
# RKNN 后端接线 (所有 YOLO 检测消费方: GPNP/pose_compare/pose_compare2/obb_pipeline)
#
# YoloDetectorFactory.cpp 按 model_path 后缀 .rknn 路由后端;
# RknnYoloDetector.cpp 仅在探测到 librknnrt 时编译。
# ============================================================================

foreach(tgt ${PROJECT_NAME} pose_compare pose_compare2 obb_pipeline)
    target_sources(${tgt} PRIVATE src/detection/YoloDetectorFactory.cpp)
    if(GPNP_RKNN_AVAILABLE)
        target_sources(${tgt} PRIVATE src/detection/RknnYoloDetector.cpp)
        target_link_libraries(${tgt} PRIVATE rknnrt)
        target_compile_definitions(${tgt} PRIVATE GPNP_WITH_RKNN=1)
    endif()
endforeach()

# ============================================================================
# Compiler Options
# ============================================================================

if(MSVC)
    foreach(tgt ${PROJECT_NAME} pose_compare pose_compare2 obb_pipeline)
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
    foreach(tgt ${PROJECT_NAME} pose_compare pose_compare2 obb_pipeline)
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

install(TARGETS ${PROJECT_NAME} pose_compare pose_compare2 obb_pipeline DESTINATION bin)

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
