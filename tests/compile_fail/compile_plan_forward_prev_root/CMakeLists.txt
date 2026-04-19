cmake_minimum_required(VERSION 3.21)

project(compile_plan_forward_prev_root LANGUAGES CXX)

add_executable(compile_plan_forward_prev_root main.cpp)
target_compile_features(compile_plan_forward_prev_root PRIVATE cxx_std_20)

get_filename_component(YORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
target_include_directories(compile_plan_forward_prev_root PRIVATE "${YORCH_ROOT}/include")
