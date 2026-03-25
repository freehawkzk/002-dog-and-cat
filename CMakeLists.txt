cmake_minimum_required(VERSION 3.10)
project(CatDogClassifier CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加 UTF-8 编码支持 (解决 C4819 警告)
if(MSVC)
    add_compile_options(/utf-8)
endif()

# ==================== 查找依赖库 ====================

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# ONNX Runtime
# 方式1: 使用环境变量 ONNXRUNTIME_DIR
# 方式2: 手动设置 ONNXRUNTIME_DIR 路径
if(DEFINED ENV{ONNXRUNTIME_DIR})
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
else()
    # Windows 默认路径 (请根据实际安装路径修改)
    if(WIN32)
        set(ONNXRUNTIME_DIR "C:/Program Files/onnxruntime" CACHE PATH "ONNX Runtime installation directory")
    else()
        set(ONNXRUNTIME_DIR "/usr/local" CACHE PATH "ONNX Runtime installation directory")
    endif()
endif()

message(STATUS "ONNX Runtime directory: ${ONNXRUNTIME_DIR}")

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
if(WIN32)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
else()
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()

if(NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR})
    message(FATAL_ERROR "ONNX Runtime include directory not found: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

# ==================== 可执行文件 ====================

add_executable(inference inference.cpp)

target_include_directories(inference PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(inference PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)

# ==================== Windows DLL 复制 ====================

if(WIN32)
    # 复制 ONNX Runtime DLL 到输出目录
    add_custom_command(TARGET inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        "$<TARGET_FILE_DIR:inference>"
        COMMENT "Copying ONNX Runtime DLL to output directory"
    )
    
    # 如果使用 CUDA，还需要复制 CUDA DLL
    # find_package(CUDA)
    # if(CUDA_FOUND)
    #     add_custom_command(TARGET inference POST_BUILD
    #         COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #         "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_*.dll"
    #         "$<TARGET_FILE_DIR:inference>"
    #     )
    # endif()
endif()

# ==================== 安装 ====================

install(TARGETS inference DESTINATION bin)

# ==================== 使用说明 ====================

message(STATUS "")
message(STATUS "========== 编译说明 ==========")
message(STATUS "1. 确保已安装 OpenCV 和 ONNX Runtime")
message(STATUS "2. 设置环境变量 ONNXRUNTIME_DIR 或修改 CMakeLists.txt 中的路径")
message(STATUS "3. 编译命令:")
message(STATUS "   mkdir build && cd build")
message(STATUS "   cmake .. -DONNXRUNTIME_DIR=<your_onnxruntime_path>")
message(STATUS "   cmake --build . --config Release")
message(STATUS "")
message(STATUS "4. 运行命令:")
message(STATUS "   ./inference <image_path> [model_path]")
message(STATUS "==============================")
message(STATUS "")
