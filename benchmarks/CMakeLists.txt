find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    include(FetchContent)

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5
        GIT_SHALLOW TRUE
    )

    FetchContent_MakeAvailable(benchmark)
endif()

function(yorch_add_benchmark target_name link_target)
  add_executable(${target_name} ${ARGN})
  target_link_libraries(${target_name} PRIVATE ${link_target} benchmark::benchmark_main)
endfunction()

yorch_add_benchmark(
  bench_run_plan
  yorch::yorch
  bench_run_plan_runtime.cpp
  bench_run_plan_optimized.cpp
)
