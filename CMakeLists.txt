cmake_minimum_required(VERSION 3.5)
project(fast-detect-gpt)

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

add_executable(fast-detect-gpt main.cpp)

target_link_libraries(fast-detect-gpt PRIVATE llama)
