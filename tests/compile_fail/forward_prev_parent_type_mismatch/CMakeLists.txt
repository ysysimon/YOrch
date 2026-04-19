cmake_minimum_required(VERSION 3.21)

project(forward_prev_parent_type_mismatch LANGUAGES CXX)

add_executable(forward_prev_parent_type_mismatch main.cpp)
target_compile_features(forward_prev_parent_type_mismatch PRIVATE cxx_std_20)

get_filename_component(YORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
target_include_directories(forward_prev_parent_type_mismatch PRIVATE "${YORCH_ROOT}/include")
