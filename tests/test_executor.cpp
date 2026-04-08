#include "yorch/executor.hpp"

#include "yorch/bind.hpp"
#include "yorch/plan.hpp"

#include <gtest/gtest.h>

#include <string>
#include <stdexcept>
#include <vector>

namespace {

struct noexcept_task {
    static constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) noexcept {
        return yorch::step_result::success();
    }
};

struct throwing_spec_task {
    static constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) {
        return yorch::step_result::success();
    }
};

struct mismatched_declared_task {
    using raw_result_type = int;

    static constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) noexcept {
        return yorch::step_result::success();
    }
};

struct lifetime_tracker {
    int live_count = 0;
    int destroyed_count = 0;
};

struct lifetime_probe {
    lifetime_tracker* tracker = nullptr;

    int value = 0;

    explicit lifetime_probe(lifetime_tracker& tracker_ref)
        : tracker(&tracker_ref) {
        ++tracker->live_count;
    }

    lifetime_probe(lifetime_tracker& tracker_ref, int in)
        : tracker(&tracker_ref),
          value(in) {
        ++tracker->live_count;
    }

    lifetime_probe(const lifetime_probe& other)
        : tracker(other.tracker),
          value(other.value) {
        ++tracker->live_count;
    }

    lifetime_probe(lifetime_probe&& other) noexcept
        : tracker(other.tracker),
          value(other.value) {
        ++tracker->live_count;
        other.value = -1;
    }

    lifetime_probe& operator=(const lifetime_probe&) = default;
    lifetime_probe& operator=(lifetime_probe&&) noexcept = default;

    ~lifetime_probe() {
        --tracker->live_count;
        ++tracker->destroyed_count;
    }
};

struct immovable_payload {
    explicit immovable_payload(int in) : value(in) {}

    immovable_payload(const immovable_payload&) = delete;
    immovable_payload& operator=(const immovable_payload&) = delete;
    immovable_payload(immovable_payload&&) = delete;
    immovable_payload& operator=(immovable_payload&&) = delete;
    ~immovable_payload() = default;

    int value = 0;
};

template <typename Plan>
concept can_run_plan =
    requires(Plan& plan) {
        yorch::run_plan(plan);
    };

template <typename LayoutPolicy, typename Plan>
concept can_run_plan_with_layout =
    requires(Plan& plan) {
        yorch::run_plan<LayoutPolicy>(plan);
    };

template <typename Plan, typename Ctx>
concept can_run_plan_with_ctx =
    requires(Plan& plan, Ctx& ctx) {
        yorch::run_plan(plan, ctx);
    };

template <typename LayoutPolicy, typename Plan, typename Ctx>
concept can_run_plan_with_ctx_and_layout =
    requires(Plan& plan, Ctx& ctx) {
        yorch::run_plan<LayoutPolicy>(plan, ctx);
    };

}  // namespace

static_assert(yorch::executable_task<noexcept_task&, void>);
static_assert(!yorch::executable_task<throwing_spec_task&, void>);
static_assert(!yorch::executable_task<mismatched_declared_task&, void>);

using caught_throwing_task_t = decltype(yorch::catch_as_failure(throwing_spec_task {}));
static_assert(yorch::executable_task<caught_throwing_task_t&, void>);

using retrying_task_t = decltype(yorch::with_retry(
    yorch::bind([]() noexcept -> yorch::step_result {
        return yorch::step_result::retry();
    }),
    yorch::retry_fixed_policy {1}));
static_assert(yorch::executable_task<retrying_task_t&, void>);

using direct_output_task_t = decltype(yorch::bind_into<int>(
    [](yorch::result_out<int> out) noexcept -> yorch::step_result {
        return out.success(1);
    }));
static_assert(!yorch::executable_task<direct_output_task_t&, void>);
static_assert(yorch::executable_direct_output_task<direct_output_task_t&, void>);

using throwing_bound_task_t = decltype(yorch::bind([]() -> yorch::step_result {
    throw std::runtime_error("boom");
}));
static_assert(!yorch::executable_task<throwing_bound_task_t&, void>);

TEST(ExecutorTest, RunTaskExecutesBoundTaskAgainstContext) {
    yorch::context<int, long> ctx(3, 7L);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
           // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        [](int& value, long label, int delta) noexcept -> bool {
            value += delta;
            return label == 7L;
        },
        yorch::from_ctx<int>(),
        yorch::from_ctx<long>(),
        yorch::value(4));

    const auto result = yorch::run_task(task, exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(ExecutorTest, RunTaskTreatsBoolAsOrdinarySuccessfulPayload) {
    yorch::exec_context<void> exec;

    auto true_task = yorch::bind(
        [](int value) noexcept -> bool {
            return value == 42;
        },
        yorch::value(42));

    auto false_task = yorch::bind(
        []() noexcept -> bool {
            return false;
        });

    const auto true_result = yorch::run_task(true_task, exec);
    const auto false_result = yorch::run_task(false_task, exec);

    EXPECT_EQ(true_result.status, yorch::step_status::success);
    EXPECT_EQ(false_result.status, yorch::step_status::success);
}

TEST(ExecutorTest, RunTaskPropagatesNonSuccessStatuses) {
    yorch::context<int> ctx(5);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto retry_task = yorch::bind(
        [](int& value) noexcept -> yorch::step_result {
            value *= 2;
            return yorch::step_result::retry();
        },
        yorch::from_ctx<int>());

    const auto result = yorch::run_task(retry_task, exec);

    EXPECT_EQ(result.status, yorch::step_status::retry);
    EXPECT_EQ(ctx.get<int>(), 10);
}

TEST(ExecutorTest, RunTaskTreatsPlainValueReturnAsSuccess) {
    yorch::context<int> ctx(5);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
        [](int& value) noexcept -> int {
            value += 3;
            return value * 2;
        },
        yorch::from_ctx<int>());

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::success);
    EXPECT_EQ(ctx.get<int>(), 8);
}

TEST(ExecutorTest, RunTaskNormalizesTaskResultPayloadCarrier) {
    yorch::context<int> ctx(5);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
        [](int& value) noexcept -> yorch::task_result<int> {
            value += 2;
            return yorch::task_result<int>::success(value * 3);
        },
        yorch::from_ctx<int>());

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::success);
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(ExecutorTest, RunTaskNormalizesStatusOnlyStepResult) {
    yorch::exec_context<void> exec;

    auto task = yorch::bind(
        []() noexcept -> yorch::step_result {
            return yorch::step_result::abort_branch();
        });

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::abort_branch);
}

TEST(ExecutorTest, RunTaskIntoSupportsDirectOutputTasks) {
    yorch::context<int> ctx(4);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind_into<int>(
        [](int& value, yorch::result_out<int> out) noexcept -> yorch::step_result {
            value += 3;
            return out.success(value * 2);
        },
        yorch::from_ctx<int>());

    yorch::detail::typed_slot<int> slot;
    const auto result = yorch::run_task_into(task, exec, yorch::result_out<int> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(ctx.get<int>(), 7);
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), 14);
}

TEST(ExecutorTest, RunTaskExecutesCatchAsFailureAdapterEndToEnd) {
    yorch::exec_context<void> exec;

    auto task = yorch::catch_as_failure(
        yorch::bind(
            []() {
                throw std::runtime_error("boom");
            }));

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::failure);
}

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

TEST(ExecutorTest, RunPlanSupportsExplicitLayoutPolicies) {
    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
        return 7;
    }));
    auto plan = yorch::compile_plan(tree);

    static_assert(can_run_plan<decltype(plan)>);
    static_assert(can_run_plan_with_layout<yorch::slot_layout_one_to_one_policy, decltype(plan)>);
    static_assert(can_run_plan_with_layout<yorch::slot_layout_serial_dfs_compact_policy, decltype(plan)>);

    const auto default_result = yorch::run_plan(plan);
    const auto explicit_result =
        yorch::run_plan<yorch::slot_layout_one_to_one_policy>(plan);
    const auto compact_result =
        yorch::run_plan<yorch::slot_layout_serial_dfs_compact_policy>(plan);

    EXPECT_TRUE(default_result.ok());
    EXPECT_TRUE(explicit_result.ok());
    EXPECT_TRUE(compact_result.ok());
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
            yorch::from_prev<int>()))
            .node<2>(yorch::bind(
                [&](const std::string& value) noexcept -> yorch::step_result {
                    seen_child = value;
                    return yorch::step_result::success();
                },
                yorch::from_prev<std::string>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_root, 5);
    EXPECT_EQ(seen_child, "15");
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
                yorch::from_prev<int>()))
                .node<2>(yorch::bind(
                    [&](const std::string& text) noexcept -> yorch::step_result {
                        trace.emplace_back("C");
                        leaf = static_cast<int>(text.size());
                        return yorch::step_result::success();
                    },
                    yorch::from_prev<std::string>()))
            .node<1>(yorch::bind(
                [&](const int& value) noexcept -> yorch::step_result {
                    trace.emplace_back("D");
                    leaf += value;
                    return yorch::step_result::success();
                },
                yorch::from_prev<int>()));
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
            yorch::from_prev<lifetime_probe>()))
            .node<2>(yorch::bind(
                [&](const int& value) noexcept -> yorch::step_result {
                    trace.emplace_back("C");
                    EXPECT_EQ(value, 8);
                    EXPECT_EQ(tracker.live_count, 1);
                    return yorch::step_result::success();
                },
                yorch::from_prev<int>()))
        .node<1>(yorch::bind(
            [&](const lifetime_probe& value) noexcept -> yorch::step_result {
                trace.emplace_back("D");
                EXPECT_EQ(value.value, 7);
                EXPECT_EQ(tracker.live_count, 1);
                return yorch::step_result::success();
            },
            yorch::from_prev<lifetime_probe>()))
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
            yorch::from_prev<int>()))
            .node<2>(yorch::bind([&]() noexcept -> yorch::step_result {
                trace.emplace_back("C");
                return yorch::step_result::success();
            }))
        .node<1>(yorch::bind(
            [&](const int&) noexcept -> yorch::step_result {
                trace.emplace_back("D");
                return yorch::step_result::success();
            },
            yorch::from_prev<int>()))
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
            yorch::from_prev<int>()))
        .node<1>(yorch::bind(
            [&](const int&) noexcept -> yorch::step_result {
                trace.emplace_back("C");
                return yorch::step_result::success();
            },
            yorch::from_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::abort_execution);
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
            yorch::from_prev<lifetime_probe>()))
        .node<1>(yorch::bind(
            [&](const lifetime_probe&) noexcept -> yorch::step_result {
                trace.emplace_back("C");
                return yorch::step_result::success();
            },
            yorch::from_prev<lifetime_probe>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::retry);
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B"}));
    EXPECT_EQ(tracker.live_count, 0);
    EXPECT_EQ(tracker.destroyed_count, 2);
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
            yorch::from_prev<lifetime_probe>()))
        .node<1>(yorch::bind(
            [&](const lifetime_probe&) noexcept -> yorch::step_result {
                trace.emplace_back("C");
                return yorch::step_result::success();
            },
            yorch::from_prev<lifetime_probe>()));

    auto plan = yorch::compile_plan(tree);
    const auto result =
        yorch::run_plan<yorch::slot_layout_serial_dfs_compact_policy>(plan);

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
            yorch::from_prev<int>())));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan, ctx);

    EXPECT_EQ(result.status, yorch::step_status::failure);
    EXPECT_EQ(ctx.get<int>(), 6);
}

TEST(ExecutorTest, RunPlanSupportsDirectOutputTasksAndImmovablePayloads) {
    int seen_value = 0;

    auto tree = yorch::task_tree.root(yorch::bind_into<immovable_payload>(
            [](yorch::result_out<immovable_payload> out) noexcept -> yorch::step_result {
                return out.success(11);
            }))
        .node<1>(yorch::bind(
            [&](const immovable_payload& value) noexcept -> yorch::step_result {
                seen_value = value.value;
                return yorch::step_result::success();
            },
            yorch::from_prev<immovable_payload>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, 11);
}

TEST(ExecutorTest, RunPlanSupportsCompactLayoutForDirectOutputTasksAndImmovablePayloads) {
    int seen_value = 0;

    auto tree = yorch::task_tree.root(yorch::bind_into<immovable_payload>(
            [](yorch::result_out<immovable_payload> out) noexcept -> yorch::step_result {
                return out.success(19);
            }))
        .node<1>(yorch::bind(
            [&](const immovable_payload& value) noexcept -> yorch::step_result {
                seen_value = value.value;
                return yorch::step_result::success();
            },
            yorch::from_prev<immovable_payload>()));

    auto plan = yorch::compile_plan(tree);
    const auto result =
        yorch::run_plan<yorch::slot_layout_serial_dfs_compact_policy>(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, 19);
}

TEST(ExecutorTest, RunPlanDestroysDirectOutputPayloadWhenNodeReturnsFailureAfterEmplace) {
    lifetime_tracker tracker;

    auto tree = yorch::task_tree.root(yorch::bind_into<lifetime_probe>(
            [&](yorch::result_out<lifetime_probe> out) noexcept -> yorch::step_result {
                out.emplace(tracker, 13);
                return yorch::step_result::failure();
            }))
        .node<1>(yorch::bind(
            [&](const lifetime_probe&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::from_prev<lifetime_probe>()));

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
                [&](yorch::result_out<int> out) noexcept -> yorch::step_result {
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
            yorch::from_prev<int>()));

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
                [&](yorch::result_out<int>) -> yorch::step_result {
                    trace.emplace_back("root");
                    throw std::runtime_error("boom");
                })))
        .node<1>(yorch::bind(
            [&](const int&) noexcept -> yorch::step_result {
                trace.emplace_back("child");
                return yorch::step_result::success();
            },
            yorch::from_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_EQ(result.status, yorch::step_status::failure);
    EXPECT_EQ(trace, (std::vector<std::string> {"root"}));
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
                yorch::from_prev<int>()),
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
                yorch::from_prev<int>()),
            yorch::retry_forever_policy {}));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(fixed_attempts, 3);
    EXPECT_EQ(forever_attempts, 2);
    EXPECT_EQ(trace, (std::vector<std::string> {"A", "B", "B", "B", "C", "D", "D"}));
}

TEST(ExecutorTest, RunPlanAllowsConstPrevFanoutAndRejectsMutableOrValueFanout) {
    auto valid_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::from_prev<int>()))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::from_prev<int>()));
    auto valid_plan = yorch::compile_plan(valid_tree);

    auto invalid_ref_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::from_prev<int>()))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::from_prev<int>()));
    auto invalid_ref_plan = yorch::compile_plan(invalid_ref_tree);

    auto invalid_value_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::from_prev<int>()))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::from_prev<int>()));
    auto invalid_value_plan = yorch::compile_plan(invalid_value_tree);

    static_assert(can_run_plan<decltype(valid_plan)>);
    static_assert(!can_run_plan<decltype(invalid_ref_plan)>);
    static_assert(!can_run_plan<decltype(invalid_value_plan)>);

    const auto result = yorch::run_plan(valid_plan);
    EXPECT_TRUE(result.ok());
}

TEST(ExecutorTest, RunPlanRejectsFromPrevOnRootNode) {
    auto invalid_root_tree = yorch::task_tree.root(yorch::bind(
        [](const int&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::from_prev<int>()));
    auto invalid_root_plan = yorch::compile_plan(invalid_root_tree);

    static_assert(!can_run_plan<decltype(invalid_root_plan)>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanRejectsFromPrevWhenDirectParentOutputIsVoid) {
    auto invalid_void_parent_tree = yorch::task_tree.root(yorch::bind([]() noexcept {}))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::from_prev<int>()));
    auto invalid_void_parent_plan = yorch::compile_plan(invalid_void_parent_tree);

    static_assert(!can_run_plan<decltype(invalid_void_parent_plan)>);
    SUCCEED();
}
