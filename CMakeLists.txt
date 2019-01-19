cmake_minimum_required(VERSION 3.0)
project(fastforest LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
find_package(pybind11 REQUIRED)
include_directories( "." )
pybind11_add_module(fastforest fastforest_py.cpp fastforest.cpp)

add_custom_target(run ALL
        COMMAND python test.py)
