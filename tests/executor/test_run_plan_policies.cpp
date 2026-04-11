#include "support.hpp"

using namespace executor_test_support;

TEST(ExecutorTest, RunPlanSupportsExplicitLayoutPolicies) {
    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
        return 7;
    }));
    auto plan = yorch::compile_plan(tree);

    static_assert(can_run_plan<decltype(plan)>);
    static_assert(can_run_plan_with_layout<yorch::slot_layout_one_to_one_policy, decltype(plan)>);
    static_assert(can_run_plan_with_layout<yorch::slot_layout_serial_dfs_compact_policy, decltype(plan)>);
    static_assert(can_run_plan_with_layout_and_exec<
                  yorch::slot_layout_one_to_one_policy,
                  yorch::exec_serial_dfs_recursive_policy,
                  decltype(plan)>);
    static_assert(can_run_plan_with_layout_and_exec<
                  yorch::slot_layout_one_to_one_policy,
                  yorch::exec_serial_dfs_explicit_heap_stack_policy,
                  decltype(plan)>);
    static_assert(can_run_plan_with_layout_and_exec<
                  yorch::slot_layout_serial_dfs_compact_policy,
                  yorch::exec_serial_dfs_explicit_heap_stack_policy,
                  decltype(plan)>);

    const auto default_result = yorch::run_plan(plan);
    const auto explicit_result =
        yorch::run_plan<yorch::slot_layout_one_to_one_policy>(plan);
    const auto compact_result =
        yorch::run_plan<yorch::slot_layout_serial_dfs_compact_policy>(plan);
    const auto heap_result =
        yorch::run_plan<
            yorch::slot_layout_one_to_one_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(plan);
    const auto compact_heap_result =
        yorch::run_plan<
            yorch::slot_layout_serial_dfs_compact_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(plan);

    EXPECT_TRUE(default_result.ok());
    EXPECT_TRUE(explicit_result.ok());
    EXPECT_TRUE(compact_result.ok());
    EXPECT_TRUE(heap_result.ok());
    EXPECT_TRUE(compact_heap_result.ok());
}

TEST(ExecutorTest, RunPlanCompactLayoutMatchesOneToOneBehavior) {
    std::vector<std::string> default_trace;
    std::vector<std::string> compact_trace;
    int default_leaf = 0;
    int compact_leaf = 0;

    auto make_tree = [](std::vector<std::string>& trace, int& leaf) {
        return yorch::task_tree.root(yorch::bind([&]() noexcept -> int {
                trace.emplace_back("A");
                return 5;
            }))
            .node<1>(yorch::bind(
                [&](const int& value) noexcept -> yorch::task_result<std::string> {
                    trace.emplace_back("B");
                    return yorch::task_result<std::string>::success(std::to_string(value * 2));
                },
                yorch::borrow_prev<int>()))
                .node<2>(yorch::bind(
                    [&](const std::string& text) noexcept -> yorch::step_result {
                        trace.emplace_back("C");
                        leaf = static_cast<int>(text.size());
                        return yorch::step_result::success();
                    },
                    yorch::borrow_prev<std::string>()))
            .node<1>(yorch::bind(
                [&](const int& value) noexcept -> yorch::step_result {
                    trace.emplace_back("D");
                    leaf += value;
                    return yorch::step_result::success();
                },
                yorch::borrow_prev<int>()));
    };

    auto default_plan = yorch::compile_plan(make_tree(default_trace, default_leaf));
    auto compact_plan = yorch::compile_plan(make_tree(compact_trace, compact_leaf));

    const auto default_result = yorch::run_plan(default_plan);
    const auto compact_result =
        yorch::run_plan<yorch::slot_layout_serial_dfs_compact_policy>(compact_plan);

    EXPECT_EQ(default_result.status, compact_result.status);
    EXPECT_EQ(default_trace, compact_trace);
    EXPECT_EQ(default_leaf, compact_leaf);
}

TEST(ExecutorTest, RunPlanExplicitHeapStackMatchesRecursiveBehavior) {
    std::vector<std::string> recursive_trace;
    std::vector<std::string> explicit_trace;
    int recursive_leaf = 0;
    int explicit_leaf = 0;

    auto make_tree = [](std::vector<std::string>& trace, int& leaf) {
        return yorch::task_tree.root(yorch::bind([&]() noexcept -> int {
                trace.emplace_back("A");
                return 5;
            }))
            .node<1>(yorch::bind(
                [&](const int& value) noexcept -> yorch::task_result<std::string> {
                    trace.emplace_back("B");
                    return yorch::task_result<std::string>::success(std::to_string(value * 2));
                },
                yorch::borrow_prev<int>()))
                .node<2>(yorch::bind(
                    [&](const std::string& text) noexcept -> yorch::step_result {
                        trace.emplace_back("C");
                        leaf = static_cast<int>(text.size());
                        return yorch::step_result::success();
                    },
                    yorch::borrow_prev<std::string>()))
            .node<1>(yorch::bind(
                [&](const int& value) noexcept -> yorch::step_result {
                    trace.emplace_back("D");
                    leaf += value;
                    return yorch::step_result::success();
                },
                yorch::borrow_prev<int>()));
    };

    auto recursive_plan = yorch::compile_plan(make_tree(recursive_trace, recursive_leaf));
    auto explicit_plan = yorch::compile_plan(make_tree(explicit_trace, explicit_leaf));

    const auto recursive_result = yorch::run_plan(recursive_plan);
    const auto explicit_result =
        yorch::run_plan<
            yorch::slot_layout_one_to_one_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(explicit_plan);

    EXPECT_EQ(recursive_result.status, explicit_result.status);
    EXPECT_EQ(recursive_trace, explicit_trace);
    EXPECT_EQ(recursive_leaf, explicit_leaf);
}

TEST(ExecutorTest, RunPlanExplicitHeapStackConsumesAbortBranchLocallyAndContinuesSiblings) {
    std::vector<std::string> trace;

    auto tree = yorch::task_tree.root(yorch::bind([&]() noexcept -> int {
            trace.emplace_back("A");
            return 1;
        }))
        .node<1>(yorch::bind(
            [&](const int&) noexcept -> yorch::step_result {
                trace.emplace_back("B");
                return yorch::step_result::abort_branch();
            },
            yorch::borrow_prev<int>()))
            .node<2>(yorch::bind([&]() noexcept -> yorch::step_result {
                trace.emplace_back("C");
                return yorch::step_result::success();
            }))
        .node<1>(yorch::bind(
            [&](const int&) noexcept -> yorch::step_result {
                trace.emplace_back("D");
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()))
            .node<2>(yorch::bind([&]() noexcept -> yorch::step_result {
                trace.emplace_back("E");
                return yorch::step_result::success();
            }));

    auto plan = yorch::compile_plan(tree);
    const auto result =
        yorch::run_plan<
            yorch::slot_layout_one_to_one_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B", "D", "E"}));
}

TEST(ExecutorTest, RunPlanExplicitHeapStackPropagatesFailureFromChildAndStopsSiblings) {
    std::vector<std::string> trace;

    auto tree = yorch::task_tree.root(yorch::bind([&]() noexcept -> int {
            trace.emplace_back("A");
            return 1;
        }))
        .node<1>(yorch::bind(
            [&](const int&) noexcept -> yorch::step_result {
                trace.emplace_back("B");
                return yorch::step_result::failure();
            },
            yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [&](const int&) noexcept -> yorch::step_result {
                trace.emplace_back("C");
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result =
        yorch::run_plan<
            yorch::slot_layout_one_to_one_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(plan);

    EXPECT_EQ(result.status, yorch::step_status::failure);
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B"}));
}

TEST(ExecutorTest, RunPlanCompactLayoutPropagatesRetryAndCleansUpLiveSlots) {
    lifetime_tracker tracker;
    std::vector<std::string> trace;

    auto tree = yorch::task_tree.root(yorch::bind([&]() noexcept -> lifetime_probe {
            trace.emplace_back("A");
            return lifetime_probe {tracker, 9};
        }))
        .node<1>(yorch::bind(
            [&](const lifetime_probe& value) noexcept -> yorch::step_result {
                trace.emplace_back("B");
                EXPECT_EQ(value.value, 9);
                EXPECT_EQ(tracker.live_count, 1);
                return yorch::step_result::retry();
            },
            yorch::borrow_prev<lifetime_probe>()))
        .node<1>(yorch::bind(
            [&](const lifetime_probe&) noexcept -> yorch::step_result {
                trace.emplace_back("C");
                return yorch::step_result::success();
            },
            yorch::borrow_prev<lifetime_probe>()));

    auto plan = yorch::compile_plan(tree);
    const auto result =
        yorch::run_plan<yorch::slot_layout_serial_dfs_compact_policy>(plan);

    EXPECT_EQ(result.status, yorch::step_status::retry);
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B"}));
    EXPECT_EQ(tracker.live_count, 0);
    EXPECT_EQ(tracker.destroyed_count, 2);
}

TEST(ExecutorTest, RunPlanExplicitHeapStackPropagatesRetryAndCleansUpLiveSlots) {
    lifetime_tracker tracker;
    std::vector<std::string> trace;

    auto tree = yorch::task_tree.root(yorch::bind([&]() noexcept -> lifetime_probe {
            trace.emplace_back("A");
            return lifetime_probe {tracker, 9};
        }))
        .node<1>(yorch::bind(
            [&](const lifetime_probe& value) noexcept -> yorch::step_result {
                trace.emplace_back("B");
                EXPECT_EQ(value.value, 9);
                EXPECT_EQ(tracker.live_count, 1);
                return yorch::step_result::retry();
            },
            yorch::borrow_prev<lifetime_probe>()))
        .node<1>(yorch::bind(
            [&](const lifetime_probe&) noexcept -> yorch::step_result {
                trace.emplace_back("C");
                return yorch::step_result::success();
            },
            yorch::borrow_prev<lifetime_probe>()));

    auto plan = yorch::compile_plan(tree);
    const auto result =
        yorch::run_plan<
            yorch::slot_layout_one_to_one_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(plan);

    EXPECT_EQ(result.status, yorch::step_status::retry);
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B"}));
    EXPECT_EQ(tracker.live_count, 0);
    EXPECT_EQ(tracker.destroyed_count, 2);
}

TEST(ExecutorTest, RunPlanExplicitHeapStackSupportsContextAndTaskAdapters) {
    yorch::context<int> ctx(4);

    auto tree = yorch::task_tree.root(yorch::bind(
            [](int& value) noexcept -> int {
                value += 2;
                return value;
            },
            yorch::from_ctx<int>()))
        .node<1>(yorch::catch_as_failure(yorch::bind(
            [](const int&) -> yorch::step_result {
                throw std::runtime_error("boom");
            },
            yorch::borrow_prev<int>())));

    auto plan = yorch::compile_plan(tree);

    static_assert(can_run_plan_with_ctx_and_layout_and_exec<
                  yorch::slot_layout_one_to_one_policy,
                  yorch::exec_serial_dfs_explicit_heap_stack_policy,
                  decltype(plan),
                  decltype(ctx)>);

    const auto result =
        yorch::run_plan<
            yorch::slot_layout_one_to_one_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(plan, ctx);

    EXPECT_EQ(result.status, yorch::step_status::failure);
    EXPECT_EQ(ctx.get<int>(), 6);
}

TEST(ExecutorTest, RunPlanExplicitHeapStackTraversesMediumDepthLinearPlan) {
    std::vector<int> trace;

    auto tree = yorch::task_tree.root(yorch::bind([&]() noexcept -> int {
            trace.push_back(0);
            return 0;
        }))
        .node<1>(yorch::bind(
            [&](const int& value) noexcept -> int {
                trace.push_back(value + 1);
                return value + 1;
            },
            yorch::borrow_prev<int>()))
            .node<2>(yorch::bind(
                [&](const int& value) noexcept -> int {
                    trace.push_back(value + 1);
                    return value + 1;
                },
                yorch::borrow_prev<int>()))
                .node<3>(yorch::bind(
                    [&](const int& value) noexcept -> int {
                        trace.push_back(value + 1);
                        return value + 1;
                    },
                    yorch::borrow_prev<int>()))
                    .node<4>(yorch::bind(
                        [&](const int& value) noexcept -> int {
                            trace.push_back(value + 1);
                            return value + 1;
                        },
                        yorch::borrow_prev<int>()))
                        .node<5>(yorch::bind(
                            [&](const int& value) noexcept -> int {
                                trace.push_back(value + 1);
                                return value + 1;
                            },
                            yorch::borrow_prev<int>()))
                            .node<6>(yorch::bind(
                                [&](const int& value) noexcept -> int {
                                    trace.push_back(value + 1);
                                    return value + 1;
                                },
                                yorch::borrow_prev<int>()))
                                .node<7>(yorch::bind(
                                    [&](const int& value) noexcept -> yorch::step_result {
                                        trace.push_back(value + 1);
                                        return yorch::step_result::success();
                                    },
                                    yorch::borrow_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result =
        yorch::run_plan<
            yorch::slot_layout_one_to_one_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(trace, (std::vector<int> {0, 1, 2, 3, 4, 5, 6, 7}));
}
