cmake_minimum_required(VERSION 3.15)
project(fast-detect-gpt)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------ llama.cpp ------
# Avoid building unnecessary components
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

include(FetchContent)

# ------ argparse ------
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v3.2
)

FetchContent_MakeAvailable(argparse)

# ------ arrow ------
# Avoid building unnecessary components
set(ARROW_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ARROW_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ARROW_PARQUET ON CACHE BOOL "" FORCE)
set(ARROW_WITH_SNAPPY ON CACHE BOOL "" FORCE)
set(ARROW_DEPENDENCY_SOURCE "BUNDLED" CACHE STRING "" FORCE)
set(ARROW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ARROW_BUILD_UTILITIES OFF CACHE BOOL "" FORCE)
set(ARROW_COMPUTE OFF CACHE BOOL "" FORCE)
set(ARROW_CSV OFF CACHE BOOL "" FORCE)
set(ARROW_JSON OFF CACHE BOOL "" FORCE)
set(ARROW_FILESYSTEM ON CACHE BOOL "" FORCE)
set(ARROW_JEMALLOC OFF CACHE BOOL "" FORCE)
set(ARROW_SIMD_LEVEL "NONE" CACHE STRING "" FORCE)

FetchContent_Declare(
        arrow
        GIT_REPOSITORY https://github.com/apache/arrow.git
        GIT_TAG        apache-arrow-22.0.0
        SOURCE_SUBDIR  cpp
)
FetchContent_MakeAvailable(arrow)

add_executable(fast-detect-gpt src/main.cpp
        src/detect.cpp
        src/utils.cpp
        include/detect.h
        include/utils.h
)

target_link_libraries(fast-detect-gpt PRIVATE llama argparse::argparse)
