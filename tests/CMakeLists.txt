include(GoogleTest)

find_package(GTest QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)

    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE
    )

    FetchContent_MakeAvailable(googletest)
endif()

function(yorch_add_gtest target_name link_target source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} PRIVATE ${link_target} GTest::gtest_main)
  gtest_discover_tests(${target_name} TEST_PREFIX "yorch.")
endfunction()

yorch_add_gtest(test_result yorch::yorch test_result.cpp)
yorch_add_gtest(test_context yorch::yorch test_context.cpp)
yorch_add_gtest(test_specs yorch::yorch test_specs.cpp)
