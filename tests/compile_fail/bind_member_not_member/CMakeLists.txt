cmake_minimum_required(VERSION 3.21)

project(bind_member_not_member LANGUAGES CXX)

add_executable(bind_member_not_member main.cpp)
target_compile_features(bind_member_not_member PRIVATE cxx_std_20)

get_filename_component(YORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
target_include_directories(bind_member_not_member PRIVATE "${YORCH_ROOT}/include")
