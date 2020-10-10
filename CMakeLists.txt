cmake_minimum_required(VERSION 3.10)
project(create_gabor_filter)

find_package(pybind11)
pybind11_add_module(gabor src/gabor.cpp)
