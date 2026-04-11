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

try_compile(
  YORCH_TASK_RESULT_VOID_REJECTED
  "${CMAKE_CURRENT_BINARY_DIR}/compile_fail/task_result_void"
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/task_result_void"
  task_result_void
  OUTPUT_VARIABLE YORCH_TASK_RESULT_VOID_OUTPUT
)

if(YORCH_TASK_RESULT_VOID_REJECTED)
  message(FATAL_ERROR
    "Expected yorch::task_result<void> compile-fail test to be rejected, but it compiled successfully.")
endif()

string(FIND
  "${YORCH_TASK_RESULT_VOID_OUTPUT}"
  "yorch::task_result<void> is not supported"
  YORCH_TASK_RESULT_VOID_MESSAGE_POS)

if(YORCH_TASK_RESULT_VOID_MESSAGE_POS EQUAL -1)
  message(FATAL_ERROR
    "Expected yorch::task_result<void> compile-fail test to mention the migration guidance, but compiler output was:\n${YORCH_TASK_RESULT_VOID_OUTPUT}")
endif()

yorch_add_gtest(test_result yorch::yorch test_result.cpp)
yorch_add_gtest(test_context yorch::yorch test_context.cpp)
yorch_add_gtest(test_specs yorch::yorch test_specs.cpp)
yorch_add_gtest(test_resolve yorch::yorch test_resolve.cpp)
yorch_add_gtest(test_bind yorch::yorch test_bind.cpp)
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
yorch_add_gtest(test_task_tree yorch::yorch test_task_tree.cpp)
yorch_add_gtest(test_plan yorch::yorch test_plan.cpp)
yorch_add_gtest(test_slots yorch::yorch test_slots.cpp)
