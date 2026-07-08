cmake_minimum_required(VERSION 3.12)
project(fast_symbolic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(../../pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11_build)
find_package(OpenMP REQUIRED)

pybind11_add_module(fast_symbolic brf.cpp)
target_link_libraries(fast_symbolic PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(fast_symbolic PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)