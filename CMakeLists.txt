cmake_minimum_required (VERSION 2.6)

include_directories(src)

set(CMAKE_CXX_STANDARD 11)

project (debug)
add_library (my_library STATIC src/debug/debug.cpp)
add_executable(debug src/debug/debug.cpp)