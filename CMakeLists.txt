cmake_minimum_required(VERSION 3.5)
project(fast_detect_gpt)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Avoid building unnecessary components of llama.cpp
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

option(CUDA "Build with CUDA support" OFF)
if (CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif ()

add_subdirectory(llama.cpp)

add_executable(fast_detect_gpt main.cpp)

target_link_libraries(fast_detect_gpt PRIVATE llama)
