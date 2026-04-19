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

function(yorch_add_gtest target_name link_target)
  add_executable(${target_name} ${ARGN})
  target_link_libraries(${target_name} PRIVATE ${link_target} GTest::gtest_main)
  gtest_discover_tests(${target_name}
    TEST_PREFIX "yorch."
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
  )
endfunction()

add_subdirectory(compile_fail)

yorch_add_gtest(test_result yorch::yorch test_result.cpp)
yorch_add_gtest(test_context yorch::yorch test_context.cpp)
yorch_add_gtest(test_specs yorch::yorch test_specs.cpp)
yorch_add_gtest(test_resolve yorch::yorch test_resolve.cpp)
yorch_add_gtest(
  test_bind
  yorch::yorch
  bind/test_traits.cpp
  bind/test_bind_free.cpp
  bind/test_bind_member.cpp
  bind/test_task_sugar.cpp
)
yorch_add_gtest(
  test_executor
  yorch::yorch
  executor/test_run_task.cpp
  executor/test_run_plan_basic.cpp
  executor/test_run_plan_policies.cpp
  executor/test_run_plan_direct_output.cpp
  executor/test_run_plan_prev_access.cpp
  executor/test_run_plan_fanout.cpp
)
yorch_add_gtest(test_task_adapters yorch::yorch test_task_adapters.cpp)
yorch_add_gtest(
  test_task_tree
  yorch::yorch
  task_tree/test_builder.cpp
  task_tree/test_callable_sugar.cpp
  task_tree/test_member_sugar.cpp
)
yorch_add_gtest(test_plan yorch::yorch test_plan.cpp)
yorch_add_gtest(test_slots yorch::yorch test_slots.cpp)
