#cmake_minimum_required (VERSION 2.6)
#
#include_directories(src)
#
#set(CMAKE_CXX_STANDARD 11)
#find_package(CUDA QUIET REQUIRED)
#
#project (parallel)
#
#find_package(Boost COMPONENTS filesystem REQUIRED)
#include_directories(${Boost_INCLUDE_DIRS})
#target_link_libraries(foo
#        ${Boost_FILESYSTEM_LIBRARY}
#        )
#
## Pass options to NVCC
#set(
#        CUDA_NVCC_FLAGS
#        ${CUDA_NVCC_FLAGS};
#        -std=c++11
#)
#
#add_executable(parallel src/main.cu)