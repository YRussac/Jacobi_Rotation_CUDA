cmake_minimum_required (VERSION 2.6)

include_directories( jacobi)

set(CMAKE_CXX_STANDARD 11)

project (Jacobi)
add_library (my_library STATIC jacobi/JacobiData.cpp)
add_executable(Jacobi jacobi/main.cpp)