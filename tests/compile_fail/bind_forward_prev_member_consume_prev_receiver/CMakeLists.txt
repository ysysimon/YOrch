cmake_minimum_required(VERSION 3.21)

project(bind_forward_prev_member_consume_prev_receiver LANGUAGES CXX)

add_executable(bind_forward_prev_member_consume_prev_receiver main.cpp)
target_compile_features(bind_forward_prev_member_consume_prev_receiver PRIVATE cxx_std_20)

get_filename_component(YORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
target_include_directories(bind_forward_prev_member_consume_prev_receiver PRIVATE "${YORCH_ROOT}/include")
