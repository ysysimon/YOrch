cmake_minimum_required(VERSION 3.21)

project(bind_into_member_arity_mismatch LANGUAGES CXX)

add_executable(bind_into_member_arity_mismatch main.cpp)
target_compile_features(bind_into_member_arity_mismatch PRIVATE cxx_std_20)

get_filename_component(YORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
target_include_directories(bind_into_member_arity_mismatch PRIVATE "${YORCH_ROOT}/include")
