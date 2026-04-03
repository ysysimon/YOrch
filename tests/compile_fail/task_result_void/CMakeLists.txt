cmake_minimum_required(VERSION 3.21)

project(task_result_void LANGUAGES CXX)

add_executable(task_result_void main.cpp)
target_compile_features(task_result_void PRIVATE cxx_std_20)

get_filename_component(YORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
target_include_directories(task_result_void PRIVATE "${YORCH_ROOT}/include")

