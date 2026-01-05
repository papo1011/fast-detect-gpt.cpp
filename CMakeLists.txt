cmake_minimum_required(VERSION 3.15)
project(fast-detect-gpt)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Avoid building unnecessary components of llama.cpp
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

include(FetchContent)

FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v3.2
)

FetchContent_MakeAvailable(argparse)

add_executable(fast-detect-gpt src/main.cpp
        src/detect.cpp
        src/utils.cpp
        include/detect.h
        include/utils.h
)

target_link_libraries(fast-detect-gpt PRIVATE llama argparse::argparse)
