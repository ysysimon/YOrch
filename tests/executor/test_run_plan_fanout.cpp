#include "support.hpp"

#include <algorithm>

using namespace executor_test_support;

namespace {

struct staging_event_log {
    std::vector<std::string> events;
};

struct staging_probe {
    staging_event_log* log = nullptr;
    int value = 0;

    explicit staging_probe(staging_event_log& event_log, int in) noexcept
        : log(&event_log),
          value(in) {}

    staging_probe(const staging_probe& other) noexcept
        : log(other.log),
          value(other.value) {
        log->events.emplace_back("copy");
    }

    staging_probe(staging_probe&& other) noexcept
        : log(other.log),
          value(other.value) {
        log->events.emplace_back("move");
        other.value = -1;
    }

    staging_probe& operator=(const staging_probe&) = default;
    staging_probe& operator=(staging_probe&&) noexcept = default;
    ~staging_probe() = default;
};

}  // namespace

TEST(ExecutorTest, RunPlanRejectsExclusivePrevAccessUnderSharedReadonlyFanoutPolicy) {
    auto invalid_tree = yorch::task_tree.root(
            yorch::bind([]() noexcept -> int {
                return 1;
            }),
            yorch::fanout_shared_readonly_policy {})
        .node<1>(yorch::bind(
            [](int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<int>()));
    using invalid_plan_t = yorch::compiled_plan_t<decltype(invalid_tree)>;

    static_assert(yorch::detail::validate_plan<invalid_plan_t>() ==
                  yorch::detail::plan_validation_error::fanout_policy_invalid);
    static_assert(!can_run_plan<invalid_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanSupportsConsumeWithCopiesFanoutAcrossPoliciesAndLayouts) {
    auto make_tree = [](std::vector<int>& seen) {
        return yorch::task_tree.root(
                yorch::bind([]() noexcept -> int {
                    return 11;
                }),
                yorch::fanout_consume_with_copies_policy {})
            .node<1>(yorch::bind(
                [&](int value) noexcept -> yorch::step_result {
                    seen.push_back(value);
                    return yorch::step_result::success();
                },
                yorch::consume_prev<int>()))
            .node<1>(yorch::bind(
                [&](int value) noexcept -> yorch::step_result {
                    seen.push_back(value);
                    return yorch::step_result::success();
                },
                yorch::copy_prev<int>()))
            .node<1>(yorch::bind(
                [&](int value) noexcept -> yorch::step_result {
                    seen.push_back(value);
                    return yorch::step_result::success();
                },
                yorch::copy_prev<int>()));
    };

    std::vector<int> default_seen;
    std::vector<int> compact_seen;
    std::vector<int> explicit_seen;
    std::vector<int> compact_explicit_seen;

    auto default_plan = yorch::compile_plan(make_tree(default_seen));
    auto compact_plan = yorch::compile_plan(make_tree(compact_seen));
    auto explicit_plan = yorch::compile_plan(make_tree(explicit_seen));
    auto compact_explicit_plan = yorch::compile_plan(make_tree(compact_explicit_seen));

    const auto default_result = yorch::run_plan(default_plan);
    const auto compact_result =
        yorch::run_plan<yorch::slot_layout_serial_dfs_compact_policy>(compact_plan);
    const auto explicit_result =
        yorch::run_plan<
            yorch::slot_layout_one_to_one_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(explicit_plan);
    const auto compact_explicit_result =
        yorch::run_plan<
            yorch::slot_layout_serial_dfs_compact_policy,
            yorch::exec_serial_dfs_explicit_heap_stack_policy>(compact_explicit_plan);

    EXPECT_TRUE(default_result.ok());
    EXPECT_TRUE(compact_result.ok());
    EXPECT_TRUE(explicit_result.ok());
    EXPECT_TRUE(compact_explicit_result.ok());
    EXPECT_EQ(default_seen, (std::vector<int> {11, 11, 11}));
    EXPECT_EQ(default_seen, compact_seen);
    EXPECT_EQ(default_seen, explicit_seen);
    EXPECT_EQ(default_seen, compact_explicit_seen);
}

TEST(ExecutorTest, RunPlanRejectsInvalidConsumeWithCopiesFanoutCombinations) {
    auto invalid_double_consume_tree = yorch::task_tree.root(
            yorch::bind([]() noexcept -> int {
                return 1;
            }),
            yorch::fanout_consume_with_copies_policy {})
        .node<1>(yorch::bind(
            [](int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::consume_prev<int>()))
        .node<1>(yorch::bind(
            [](int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::consume_prev<int>()));
    using invalid_double_consume_plan_t = yorch::compiled_plan_t<decltype(invalid_double_consume_tree)>;

    auto invalid_borrow_tree = yorch::task_tree.root(
            yorch::bind([]() noexcept -> int {
                return 1;
            }),
            yorch::fanout_consume_with_copies_policy {})
        .node<1>(yorch::bind(
            [](int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::consume_prev<int>()))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));
    using invalid_borrow_plan_t = yorch::compiled_plan_t<decltype(invalid_borrow_tree)>;

    auto invalid_mut_tree = yorch::task_tree.root(
            yorch::bind([]() noexcept -> int {
                return 1;
            }),
            yorch::fanout_consume_with_copies_policy {})
        .node<1>(yorch::bind(
            [](int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::consume_prev<int>()))
        .node<1>(yorch::bind(
            [](int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<int>()));
    using invalid_mut_plan_t = yorch::compiled_plan_t<decltype(invalid_mut_tree)>;

    static_assert(yorch::detail::validate_plan<invalid_double_consume_plan_t>() ==
                  yorch::detail::plan_validation_error::fanout_policy_invalid);
    static_assert(!can_run_plan<invalid_double_consume_plan_t>);
    static_assert(!can_run_plan<invalid_borrow_plan_t>);
    static_assert(!can_run_plan<invalid_mut_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanStagesCopiesBeforeConsumeChildRuns) {
    staging_event_log log;
    int seen_consume = 0;
    int seen_copy = 0;

    auto tree = yorch::task_tree.root(
            yorch::bind([&]() noexcept -> staging_probe {
                return staging_probe {log, 9};
            }),
            yorch::fanout_consume_with_copies_policy {})
        .node<1>(yorch::bind(
            // NOLINTNEXTLINE(performance-unnecessary-value-param)
            [&](staging_probe value) noexcept -> yorch::step_result {
                log.events.emplace_back("consume");
                seen_consume = value.value;
                return yorch::step_result::success();
            },
            yorch::consume_prev<staging_probe>()))
        .node<1>(yorch::bind(
            // NOLINTNEXTLINE(performance-unnecessary-value-param)
            [&](staging_probe value) noexcept -> yorch::step_result {
                log.events.emplace_back("copy-child");
                seen_copy = value.value;
                EXPECT_EQ(value.value, 9);
                return yorch::step_result::success();
            },
            yorch::copy_prev<staging_probe>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_consume, 9);
    EXPECT_EQ(seen_copy, 9);
    ASSERT_FALSE(log.events.empty());

    const auto consume_it = std::find(log.events.begin(), log.events.end(), "consume");
    const auto copy_it = std::find(log.events.begin(), log.events.end(), "copy");

    ASSERT_NE(consume_it, log.events.end());
    ASSERT_NE(copy_it, log.events.end());
    EXPECT_LT(copy_it, consume_it);
}

TEST(ExecutorTest, RunPlanUsesOneSharedStageForMultipleCopyChildren) {
    staging_event_log log;
    int seen_consume = 0;
    int seen_copy_a = 0;
    int seen_copy_b = 0;

    auto tree = yorch::task_tree.root(
            yorch::bind([&]() noexcept -> staging_probe {
                return staging_probe {log, 17};
            }),
            yorch::fanout_consume_with_copies_policy {})
        .node<1>(yorch::bind(
            // NOLINTNEXTLINE(performance-unnecessary-value-param)
            [&](staging_probe value) noexcept -> yorch::step_result {
                log.events.emplace_back("consume");
                seen_consume = value.value;
                return yorch::step_result::success();
            },
            yorch::consume_prev<staging_probe>()))
        .node<1>(yorch::bind(
            // NOLINTNEXTLINE(performance-unnecessary-value-param)
            [&](staging_probe value) noexcept -> yorch::step_result {
                log.events.emplace_back("copy-a");
                seen_copy_a = value.value;
                return yorch::step_result::success();
            },
            yorch::copy_prev<staging_probe>()))
        .node<1>(yorch::bind(
            // NOLINTNEXTLINE(performance-unnecessary-value-param)
            [&](staging_probe value) noexcept -> yorch::step_result {
                log.events.emplace_back("copy-b");
                seen_copy_b = value.value;
                return yorch::step_result::success();
            },
            yorch::copy_prev<staging_probe>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_consume, 17);
    EXPECT_EQ(seen_copy_a, 17);
    EXPECT_EQ(seen_copy_b, 17);
    EXPECT_EQ(std::count(log.events.begin(), log.events.end(), "copy"), 3);
}

TEST(ExecutorTest, RunPlanDestroysFanoutStagingOnAbortExecution) {
    lifetime_tracker tracker;
    bool copy_child_ran = false;

    auto tree = yorch::task_tree.root(
            yorch::bind([&]() noexcept -> lifetime_probe {
                return lifetime_probe {tracker, 5};
            }),
            yorch::fanout_consume_with_copies_policy {})
        .node<1>(yorch::bind(
            // NOLINTNEXTLINE(performance-unnecessary-value-param)
            [](lifetime_probe) noexcept -> yorch::step_result {
                return yorch::step_result::abort_execution();
            },
            yorch::consume_prev<lifetime_probe>()))
        .node<1>(yorch::bind(
            // NOLINTNEXTLINE(performance-unnecessary-value-param)
            [&](lifetime_probe) noexcept -> yorch::step_result {
                copy_child_ran = true;
                return yorch::step_result::success();
            },
            yorch::copy_prev<lifetime_probe>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::abort_execution);
    EXPECT_FALSE(copy_child_ran);
    EXPECT_EQ(tracker.live_count, 0);
    EXPECT_GT(tracker.destroyed_count, 0);
}
