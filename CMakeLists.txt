cmake_minimum_required(VERSION 3.20)
project(QnVoiceCebuanoResearch LANGUAGES C CXX)

set(LLAMA_SOURCE_DIR "" CACHE PATH "Checkout of the pinned llama.cpp runtime")
if(NOT EXISTS "${LLAMA_SOURCE_DIR}/include/llama.h")
    message(FATAL_ERROR "Set LLAMA_SOURCE_DIR to the runtime checkout listed in README.md")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static runtime libraries")
set(GGML_OPENMP OFF CACHE BOOL "Enable OpenMP for ggml")

add_subdirectory("${LLAMA_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama" EXCLUDE_FROM_ALL)
add_executable(native_benchmark tools/native_benchmark.cpp)
target_compile_features(native_benchmark PRIVATE cxx_std_17)
target_link_libraries(native_benchmark PRIVATE llama)
