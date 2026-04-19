cmake_minimum_required(VERSION 3.21)

project(compile_plan_retry_consume_prev LANGUAGES CXX)

add_executable(compile_plan_retry_consume_prev main.cpp)
target_compile_features(compile_plan_retry_consume_prev PRIVATE cxx_std_20)

get_filename_component(YORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
target_include_directories(compile_plan_retry_consume_prev PRIVATE "${YORCH_ROOT}/include")
