#include "run_plan_helpers.hpp"

#include <benchmark/benchmark.h>
#include <yorch/yorch.hpp>

#include <cstdint>
#include <string>

namespace {

template <typename Scenario, typename LayoutPolicy, typename ExecPolicy>
void run_plan_optimized_case(benchmark::State& state) {
    int sink = 0;
    auto tree = Scenario::make_tree(sink);
    auto plan = yorch::compile_plan(tree);

    for (auto _ : state) {
        const auto result = yorch::run_plan<LayoutPolicy, ExecPolicy>(plan);
        auto status = static_cast<int>(result.status);
        benchmark::DoNotOptimize(status);
        benchmark::DoNotOptimize(sink);

        if (!result.ok()) {
            state.SkipWithError("run_plan returned a non-success result");
            break;
        }
    }

    state.SetItemsProcessed(state.iterations() * Scenario::node_count);
}

[[nodiscard]] std::string make_case_name(
    const char* topology,
    const char* payload,
    std::int64_t nodes,
    const char* slot_layout,
    const char* exec_policy) {
    return std::string {"RunPlan/Optimized/"} + topology + "/" + payload + "/" +
           std::to_string(nodes) + "/" + slot_layout + "/" + exec_policy;
}

template <typename Scenario, typename LayoutPolicy, typename ExecPolicy>
void register_case(const char* slot_layout, const char* exec_policy) {
    benchmark::RegisterBenchmark(
        make_case_name(
            Scenario::topology,
            Scenario::payload,
            Scenario::node_count,
            slot_layout,
            exec_policy),
        &run_plan_optimized_case<Scenario, LayoutPolicy, ExecPolicy>);
}

template <typename Scenario>
void register_policy_cases() {
    register_case<
        Scenario,
        yorch::slot_layout_one_to_one_policy,
        yorch::exec_serial_dfs_recursive_policy>("OneToOne", "Recursive");
    register_case<
        Scenario,
        yorch::slot_layout_serial_dfs_compact_policy,
        yorch::exec_serial_dfs_recursive_policy>("Compact", "Recursive");
    register_case<
        Scenario,
        yorch::slot_layout_one_to_one_policy,
        yorch::exec_serial_dfs_explicit_heap_stack_policy>("OneToOne", "HeapStack");
    register_case<
        Scenario,
        yorch::slot_layout_serial_dfs_compact_policy,
        yorch::exec_serial_dfs_explicit_heap_stack_policy>("Compact", "HeapStack");
}

void register_run_plan_optimized_cases() {
    register_policy_cases<yorch_bench::chain8_scenario>();
    register_policy_cases<yorch_bench::chain32_scenario>();
    register_policy_cases<yorch_bench::wide8_scenario>();
    register_policy_cases<yorch_bench::balanced15_scenario>();
    register_policy_cases<yorch_bench::fanout_consume_copies3_scenario>();
}

const bool registered = [] {
    register_run_plan_optimized_cases();
    return true;
}();

} // namespace
