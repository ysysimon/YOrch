#include "support.hpp"

using namespace executor_test_support;

TEST(ExecutorTest, RunPlanSupportsDirectOutputTasksAndImmovablePayloads) {
    int seen_value = 0;

    auto tree = yorch::task_tree.root(yorch::bind_into<immovable_payload>(
            [](yorch::direct_out<immovable_payload> out) noexcept -> yorch::step_result {
                return out.success(11);
            }))
        .node<1>(yorch::bind(
            [&](const immovable_payload& value) noexcept -> yorch::step_result {
                seen_value = value.value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<immovable_payload>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, 11);
}

TEST(ExecutorTest, RunPlanSupportsCompactLayoutForDirectOutputTasksAndImmovablePayloads) {
    int seen_value = 0;

    auto tree = yorch::task_tree.root(yorch::bind_into<immovable_payload>(
            [](yorch::direct_out<immovable_payload> out) noexcept -> yorch::step_result {
                return out.success(19);
            }))
        .node<1>(yorch::bind(
            [&](const immovable_payload& value) noexcept -> yorch::step_result {
                seen_value = value.value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<immovable_payload>()));

    auto plan = yorch::compile_plan(tree);
    const auto result =
        yorch::run_plan<yorch::slot_layout_serial_dfs_compact_policy>(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, 19);
}

TEST(ExecutorTest, RunPlanExplicitHeapStackSupportsCompactLayoutForDirectOutputTasksAndImmovablePayloads) {
    int seen_value = 0;

    auto tree = yorch::task_tree.root(yorch::bind_into<immovable_payload>(
            [](yorch::direct_out<immovable_payload> out) noexcept -> yorch::step_result {
                return out.success(23);
            }))
        .node<1>(yorch::bind(
            [&](const immovable_payload& value) noexcept -> yorch::step_result {
                seen_value = value.value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<immovable_payload>()));

    auto plan = yorch::compile_plan(tree);
    const auto result =
        yorch::run_plan<
            yorch::slot_layout_serial_dfs_compact_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, 23);
}

TEST(ExecutorTest, RunPlanDestroysDirectOutputPayloadWhenNodeReturnsFailureAfterEmplace) {
    lifetime_tracker tracker;

    auto tree = yorch::task_tree.root(yorch::bind_into<lifetime_probe>(
            [&](yorch::direct_out<lifetime_probe> out) noexcept -> yorch::step_result {
                out.emplace(tracker, 13);
                return yorch::step_result::failure();
            }))
        .node<1>(yorch::bind(
            [&](const lifetime_probe&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev<lifetime_probe>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::failure);
    EXPECT_EQ(tracker.live_count, 0);
    EXPECT_EQ(tracker.destroyed_count, 1);
}

TEST(ExecutorTest, RunPlanSupportsRetryAdapterOnDirectOutputTasks) {
    int attempts = 0;
    int seen_value = 0;

    auto tree = yorch::task_tree.root(yorch::with_retry(
            yorch::bind_into<int>(
                [&](yorch::direct_out<int> out) noexcept -> yorch::step_result {
                    ++attempts;

                    if (attempts < 3) {
                        out.emplace(attempts);
                        return yorch::step_result::retry();
                    }

                    return out.success(21);
                }),
            yorch::retry_fixed_policy {4}))
        .node<1>(yorch::bind(
            [&](const int& value) noexcept -> yorch::step_result {
                seen_value = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(attempts, 3);
    EXPECT_EQ(seen_value, 21);
}

TEST(ExecutorTest, RunPlanSupportsCatchAdapterOnDirectOutputTasks) {
    std::vector<std::string> trace;

    auto tree = yorch::task_tree.root(yorch::catch_as_failure(
            yorch::bind_into<int>(
                [&](yorch::direct_out<int>) -> yorch::step_result {
                    trace.emplace_back("root");
                    throw std::runtime_error("boom");
                })))
        .node<1>(yorch::bind(
            [&](const int&) noexcept -> yorch::step_result {
                trace.emplace_back("child");
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::failure);
    EXPECT_EQ(trace, (std::vector<std::string> {"root"}));
}
