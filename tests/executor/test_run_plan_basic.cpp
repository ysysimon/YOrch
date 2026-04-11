#include "support.hpp"

using namespace executor_test_support;

TEST(ExecutorTest, RunPlanExecutesRootOnlyPlan) {
    int executions = 0;

    auto tree = yorch::task_tree.root(yorch::bind([&]() noexcept -> int {
        ++executions;
        return 9;
    }));
    auto plan = yorch::compile_plan(tree);

    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(executions, 1);
}

TEST(ExecutorTest, RunPlanPropagatesDirectParentPayloadsOnly) {
    int seen_root = 0;
    std::string seen_child;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 5;
        }))
        .node<1>(yorch::bind(
            [&](const int& value) noexcept -> yorch::task_result<std::string> {
                seen_root = value;
                return yorch::task_result<std::string>::success(
                    std::to_string(value * 3));
            },
            yorch::borrow_prev<int>()))
            .node<2>(yorch::bind(
                [&](const std::string& value) noexcept -> yorch::step_result {
                    seen_child = value;
                    return yorch::step_result::success();
                },
                yorch::borrow_prev<std::string>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_root, 5);
    EXPECT_EQ(seen_child, "15");
}

TEST(ExecutorTest, RunPlanTraversesDepthFirstAndKeepsParentPayloadAlive) {
    lifetime_tracker tracker;

    std::vector<std::string> trace;

    auto tree = yorch::task_tree.root(yorch::bind([&]() noexcept -> lifetime_probe {
            trace.emplace_back("A");
            return lifetime_probe {tracker, 7};
        }))
        .node<1>(yorch::bind(
            [&](const lifetime_probe& value) noexcept -> int {
                trace.emplace_back("B");
                EXPECT_EQ(value.value, 7);
                EXPECT_EQ(tracker.live_count, 1);
                return value.value + 1;
            },
            yorch::borrow_prev<lifetime_probe>()))
            .node<2>(yorch::bind(
                [&](const int& value) noexcept -> yorch::step_result {
                    trace.emplace_back("C");
                    EXPECT_EQ(value, 8);
                    EXPECT_EQ(tracker.live_count, 1);
                    return yorch::step_result::success();
                },
                yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [&](const lifetime_probe& value) noexcept -> yorch::step_result {
                trace.emplace_back("D");
                EXPECT_EQ(value.value, 7);
                EXPECT_EQ(tracker.live_count, 1);
                return yorch::step_result::success();
            },
            yorch::borrow_prev<lifetime_probe>()))
            .node<2>(yorch::bind([&]() noexcept -> yorch::step_result {
                trace.emplace_back("E");
                EXPECT_EQ(tracker.live_count, 1);
                return yorch::step_result::success();
            }));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B", "C", "D", "E"}));
    EXPECT_EQ(tracker.live_count, 0);
    EXPECT_EQ(tracker.destroyed_count, 2);
}

TEST(ExecutorTest, RunPlanConsumesAbortBranchLocallyAndContinuesSiblings) {
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
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B", "D", "E"}));
}

TEST(ExecutorTest, RunPlanPropagatesAbortExecutionFromChild) {
    std::vector<std::string> trace;

    auto tree = yorch::task_tree.root(yorch::bind([&]() noexcept -> int {
            trace.emplace_back("A");
            return 1;
        }))
        .node<1>(yorch::bind(
            [&](const int&) noexcept -> yorch::step_result {
                trace.emplace_back("B");
                return yorch::step_result::abort_execution();
            },
            yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [&](const int&) noexcept -> yorch::step_result {
                trace.emplace_back("C");
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::abort_execution);
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B"}));
}

TEST(ExecutorTest, RunPlanPropagatesFailureFromChildAndStopsSiblings) {
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
    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::failure);
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B"}));
}

TEST(ExecutorTest, RunPlanPropagatesRetryAndCleansUpLiveSlots) {
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
    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::retry);
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B"}));
    EXPECT_EQ(tracker.live_count, 0);
    EXPECT_EQ(tracker.destroyed_count, 2);
}

TEST(ExecutorTest, RunPlanSupportsContextAndTaskAdapters) {
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
    const auto result = yorch::run_plan(plan, ctx);

    EXPECT_EQ(result.status, yorch::step_status::failure);
    EXPECT_EQ(ctx.get<int>(), 6);
}

TEST(ExecutorTest, RunPlanSupportsPerNodeRetryPolicies) {
    std::vector<std::string> trace;
    int fixed_attempts = 0;
    int forever_attempts = 0;

    auto tree = yorch::task_tree.root(yorch::bind([&]() noexcept -> int {
            trace.emplace_back("A");
            return 5;
        }))
        .node<1>(yorch::with_retry(
            yorch::bind(
                [&](const int& value) noexcept -> yorch::step_result {
                    trace.emplace_back("B");
                    EXPECT_EQ(value, 5);
                    ++fixed_attempts;
                    return fixed_attempts < 3
                        ? yorch::step_result::retry()
                        : yorch::step_result::success();
                },
                yorch::borrow_prev<int>()),
            yorch::retry_fixed_policy {4}))
            .node<2>(yorch::bind([&]() noexcept -> yorch::step_result {
                trace.emplace_back("C");
                return yorch::step_result::success();
            }))
        .node<1>(yorch::with_retry(
            yorch::bind(
                [&](const int& value) noexcept -> yorch::step_result {
                    trace.emplace_back("D");
                    EXPECT_EQ(value, 5);
                    ++forever_attempts;
                    return forever_attempts < 2
                        ? yorch::step_result::retry()
                        : yorch::step_result::success();
                },
                yorch::borrow_prev<int>()),
            yorch::retry_forever_policy {}));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(fixed_attempts, 3);
    EXPECT_EQ(forever_attempts, 2);
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B", "B", "B", "C", "D", "D"}));
}
