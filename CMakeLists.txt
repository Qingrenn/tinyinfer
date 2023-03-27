cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_C_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message("Setting default build type to Release")
endif()

project(tinyinfer VERSION 0.0.1 LANGUAGES C CXX)

option(TINYINFER_SHARED_LIB "shared library support" OFF)
option(TINYINFER_ENABLE_TEST "shared library support" OFF)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(./src)
add_subdirectory(./tools)

if(TINYINFER_ENABLE_TEST)
    enable_testing()
    add_subdirectory(./test)
endif()


# add_subdirectory(./example)