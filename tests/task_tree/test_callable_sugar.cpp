#include "support.hpp" // IWYU pragma: keep

using namespace task_tree_test_support;

TEST(TaskTreeTest, RootCallableAndNodeCallableBuildRunnablePlan) {
    int seen_root = 0;
    std::string seen_child;

    auto tree = yorch::task_tree
        .root([]() noexcept -> int {
            return 7;
        })()
        .node<1>(
            [&](const int& value) noexcept -> yorch::task_result<std::string> {
                seen_root = value;
                return yorch::task_result<std::string>::success(std::to_string(value * 2));
            })(yorch::borrow_prev<int>())
        .node<2>(
            [&](const std::string& value) noexcept -> yorch::step_result {
                seen_child = value;
                return yorch::step_result::success();
            })(yorch::borrow_prev<std::string>());

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_root, 7);
    EXPECT_EQ(seen_child, "14");
}

TEST(TaskTreeTest, RootCallableIntoAndNodeCallableIntoBuildRunnablePlan) {
    int seen_root = 0;
    int seen_child = 0;

    auto tree = yorch::task_tree
        .root_into([](yorch::direct_out<std::string> out) noexcept -> yorch::step_result {
            return out.success("7");
        })()
        .node_into<1>(
            [&](const std::string& value,
                yorch::direct_out<int> out) noexcept -> yorch::step_result {
                seen_root = std::stoi(value);
                return out.success(seen_root * 2);
            })(yorch::borrow_prev<std::string>())
        .node<2>(
            [&](const int& value) noexcept -> yorch::step_result {
                seen_child = value;
                return yorch::step_result::success();
            })(yorch::borrow_prev<int>());

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_root, 7);
    EXPECT_EQ(seen_child, 14);
}

TEST(TaskTreeTest, RootAndNodeForwardPrevCallableSugarBuildRunnablePlan) {
    const int* root_seen = nullptr;
    const int* child_seen = nullptr;
    int final_value = 0;

    auto tree = yorch::task_tree
        .root([]() noexcept -> int {
            return 5;
        })()
        .node_forward_prev<1>(
            [&](int& value) noexcept -> yorch::step_result {
                root_seen = &value;
                value += 4;
                return yorch::step_result::success();
            })(yorch::borrow_prev_mut<int>())
        .node<2>(
            [&](const int& value) noexcept -> yorch::step_result {
                child_seen = &value;
                final_value = value;
                return yorch::step_result::success();
            })(yorch::borrow_prev<int>());

    auto plan = yorch::compile_plan(tree);
    using plan_t = decltype(plan);
    static_assert(plan_t::template output_storage_mode_for<1> == yorch::detail::output_storage_mode::forwarded_prev);
    static_assert(yorch::plan_exec_slots<plan_t>::physical_slot_count == 1);

    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    ASSERT_NE(root_seen, nullptr);
    ASSERT_NE(child_seen, nullptr);
    EXPECT_EQ(root_seen, child_seen);
    EXPECT_EQ(final_value, 9);
}

TEST(TaskTreeTest, RootCallableWithDefaultCatchAdapterWrapsBoundTask) {
    auto tree = yorch::task_tree.root(
        []() -> yorch::step_result {
            throw std::runtime_error("boom");
        },
        yorch::adapters(yorch::adapt_catch_as_failure()))();
    auto plan = yorch::compile_plan(tree);

    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::failure);
}

TEST(TaskTreeTest, RootCallableIntoWithDefaultCatchAdapterWrapsDirectOutputTask) {
    auto tree = yorch::task_tree.root_into(
        [](yorch::direct_out<int>) -> yorch::step_result {
            throw std::runtime_error("boom");
        },
        yorch::adapters(yorch::adapt_catch_as_failure()))();
    auto plan = yorch::compile_plan(tree);

    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::failure);
}

TEST(TaskTreeTest, NodeCallableWithCustomCatchAdapterWrapsBoundTask) {
    auto tree = yorch::task_tree
        .root([]() noexcept -> int {
            return 5;
        })()
        .node<1>(
            [](const int&) -> yorch::task_result<int> {
                throw std::runtime_error("boom");
            },
            yorch::adapters(yorch::adapt_catch_as_failure(
                // NOLINTNEXTLINE(performance-unnecessary-value-param)
                [](std::exception_ptr) noexcept -> yorch::task_result<int> {
                    return yorch::task_result<int>::success(-9);
                })))(yorch::borrow_prev<int>());

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
}

TEST(TaskTreeTest, NodeCallableIntoWithCustomCatchAdapterWrapsDirectOutputTask) {
    auto tree = yorch::task_tree
        .root([]() noexcept -> int {
            return 5;
        })()
        .node_into<1>(
            [](const int&, yorch::direct_out<std::string>) -> yorch::step_result {
                throw std::runtime_error("boom");
            },
            yorch::adapters(yorch::adapt_catch_as_failure(
                // NOLINTNEXTLINE(performance-unnecessary-value-param)
                [](std::exception_ptr) noexcept -> yorch::step_result {
                    return yorch::step_result::abort_branch();
                })))(yorch::borrow_prev<int>());

    int parent_value = 5;
    yorch::exec_context<void, decltype(yorch::prev_slot(parent_value))> exec {
        yorch::prev_slot(parent_value)};
    yorch::detail::typed_slot<std::string> slot;

    const auto step = tree.entry<1>().task.invoke_into(
        exec,
        yorch::direct_out<std::string> {slot});

    EXPECT_EQ(step.status, yorch::step_status::abort_branch);
    EXPECT_FALSE(slot.has_value());
}

TEST(TaskTreeTest, RootCallableWithRetryAdapterWrapsBoundTask) {
    int attempts = 0;

    auto tree = yorch::task_tree.root(
        [&]() noexcept -> yorch::step_result {
            ++attempts;
            return attempts < 3
                ? yorch::step_result::retry()
                : yorch::step_result::success();
        },
        yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {3})))();
    auto plan = yorch::compile_plan(tree);

    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(attempts, 3);
}

TEST(TaskTreeTest, RootCallableIntoWithRetryAdapterWrapsDirectOutputTask) {
    int attempts = 0;
    int seen_value = 0;

    auto tree = yorch::task_tree
        .root_into(
            [&](yorch::direct_out<int> out) noexcept -> yorch::step_result {
                ++attempts;

                if (attempts < 3) {
                    out.emplace(attempts);
                    return yorch::step_result::retry();
                }

                return out.success(21);
            },
            yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {3})))()
        .node<1>(
            [&](const int& value) noexcept -> yorch::step_result {
                seen_value = value;
                return yorch::step_result::success();
            })(yorch::borrow_prev<int>());
    auto plan = yorch::compile_plan(tree);

    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(attempts, 3);
    EXPECT_EQ(seen_value, 21);
}

TEST(TaskTreeTest, NodeCallableWithRetryAdapterWrapsBoundTask) {
    int attempts = 0;

    auto tree = yorch::task_tree
        .root([]() noexcept -> int {
            return 3;
        })()
        .node<1>(
            [&](const int& value) noexcept -> yorch::step_result {
                ++attempts;
                return attempts < value
                    ? yorch::step_result::retry()
                    : yorch::step_result::success();
            },
            yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {2})))
        (yorch::borrow_prev<int>());
    auto plan = yorch::compile_plan(tree);

    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(attempts, 3);
}

TEST(TaskTreeTest, NodeCallableIntoWithRetryAdapterWrapsDirectOutputTask) {
    int attempts = 0;
    int seen_value = 0;

    auto tree = yorch::task_tree
        .root([]() noexcept -> int {
            return 3;
        })()
        .node_into<1>(
            [&](const int& value, yorch::direct_out<std::string> out) noexcept -> yorch::step_result {
                ++attempts;

                if (attempts < value) {
                    out.emplace(std::to_string(attempts));
                    return yorch::step_result::retry();
                }

                return out.success(std::to_string(value * 2));
            },
            yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {3})))(yorch::borrow_prev<int>())
        .node<2>(
            [&](const std::string& value) noexcept -> yorch::step_result {
                seen_value = std::stoi(value);
                return yorch::step_result::success();
            })(yorch::borrow_prev<std::string>());
    auto plan = yorch::compile_plan(tree);

    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(attempts, 3);
    EXPECT_EQ(seen_value, 6);
}
