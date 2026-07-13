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
    src/utils/PoseUtils.cpp
    src/visualization/Visualizer.cpp
    src/tracker/StereoTracker.cpp
    src/detection/RoiGenerator.cpp
    src/detection/YoloRoiProvider.cpp
)

# ============================================================================
# Executable
# ============================================================================

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_INCLUDE_DIR}
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

# ============================================================================
# Compiler Options
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
        -O3
        -march=native
    )
endif()

# ============================================================================
# Install
# ============================================================================

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

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
