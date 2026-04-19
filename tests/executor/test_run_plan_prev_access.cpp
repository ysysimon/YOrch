#include "support.hpp"
#include <gtest/gtest.h>

using namespace executor_test_support;

namespace {

struct member_prev_worker {
    int value = 0;

    int mutate(int delta) noexcept {
        value += delta;
        return value;
    }

    int mutate_from_view(const member_prev_worker& other) noexcept {
        value += other.value;
        return value;
    }

    int mutate_with_copy(member_prev_worker other) noexcept {
        value += other.value;
        return value;
    }

    [[nodiscard]] int read_plus(int delta) const noexcept {
        return value + delta;
    }

    [[nodiscard]] int read_with_copy(member_prev_worker other) const noexcept {
        return value + other.value;
    }
};

struct forwarded_payload {
    int value = 0;
};

struct move_only_forwarded_payload {
    explicit move_only_forwarded_payload(int initial) : value(initial) {}

    move_only_forwarded_payload(const move_only_forwarded_payload&) = delete;
    move_only_forwarded_payload& operator=(const move_only_forwarded_payload&) = delete;
    move_only_forwarded_payload(move_only_forwarded_payload&&) noexcept = default;
    move_only_forwarded_payload& operator=(move_only_forwarded_payload&&) noexcept = default;
    ~move_only_forwarded_payload() = default;

    int value = 0;
};

using borrow_prev_task_t = decltype(yorch::bind(
    [](const int&) noexcept -> yorch::step_result {
        return yorch::step_result::success();
    },
    yorch::borrow_prev<int>()));

using borrow_prev_mut_task_t = decltype(yorch::bind(
    [](int&) noexcept -> yorch::step_result {
        return yorch::step_result::success();
    },
    yorch::borrow_prev_mut<int>()));

using copy_prev_task_t = decltype(yorch::bind(
    [](int) noexcept -> yorch::step_result {
        return yorch::step_result::success();
    },
    yorch::copy_prev<int>()));

using consume_prev_value_task_t = decltype(yorch::bind(
    [](int) noexcept -> yorch::step_result {
        return yorch::step_result::success();
    },
    yorch::consume_prev<int>()));

using consume_prev_rvalue_task_t = decltype(yorch::bind(
    [](int&&) noexcept -> yorch::step_result {
        return yorch::step_result::success();
    },
    yorch::consume_prev<int>()));

using shared_prev_task_t = decltype(yorch::bind(
    [](const int&, int) noexcept -> yorch::step_result {
        return yorch::step_result::success();
    },
    yorch::borrow_prev<int>(),
    yorch::copy_prev<int>()));

using invalid_exclusive_mix_task_t = decltype(yorch::bind(
    [](const int&, int&) noexcept -> yorch::step_result {
        return yorch::step_result::success();
    },
    yorch::borrow_prev<int>(),
    yorch::borrow_prev_mut<int>()));

using invalid_double_exclusive_task_t = decltype(yorch::bind(
    [](int&, int) noexcept -> yorch::step_result {
        return yorch::step_result::success();
    },
    yorch::borrow_prev_mut<int>(),
    yorch::consume_prev<int>()));

using const_receiver_task_t = decltype(yorch::bind_member(
    &member_prev_worker::read_plus,
    yorch::borrow_prev<member_prev_worker>(),
    yorch::value(1)));

using invalid_mutable_receiver_task_t = decltype(yorch::bind_member(
    &member_prev_worker::mutate,
    yorch::borrow_prev<member_prev_worker>(),
    yorch::value(1)));

using catch_copy_prev_task_t = decltype(yorch::catch_as_failure(
    yorch::bind(
        [](int) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::copy_prev<int>())));

using retry_copy_prev_task_t = decltype(yorch::with_retry(
    yorch::bind(
        [](int) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::copy_prev<int>()),
    yorch::retry_fixed_policy {1}));

using retry_consume_prev_task_t = decltype(yorch::with_retry(
    yorch::bind(
        [](int) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::consume_prev<int>()),
    yorch::retry_fixed_policy {1}));

static_assert(yorch::detail::task_prev_access_valid_v<borrow_prev_task_t>);
static_assert(yorch::detail::task_uses_prev_access_v<borrow_prev_task_t>);
static_assert(yorch::detail::task_uses_borrow_prev_v<borrow_prev_task_t>);
static_assert(!yorch::detail::task_uses_copy_prev_v<borrow_prev_task_t>);

static_assert(yorch::detail::task_prev_access_valid_v<borrow_prev_mut_task_t>);
static_assert(yorch::detail::task_uses_borrow_prev_mut_v<borrow_prev_mut_task_t>);
static_assert(yorch::detail::task_uses_exclusive_prev_access_v<borrow_prev_mut_task_t>);

static_assert(yorch::detail::task_prev_access_valid_v<copy_prev_task_t>);
static_assert(yorch::detail::task_uses_copy_prev_v<copy_prev_task_t>);

static_assert(yorch::detail::task_prev_access_valid_v<consume_prev_value_task_t>);
static_assert(yorch::detail::task_prev_access_valid_v<consume_prev_rvalue_task_t>);
static_assert(yorch::detail::task_uses_consume_prev_v<consume_prev_rvalue_task_t>);
static_assert(yorch::detail::task_uses_exclusive_prev_access_v<consume_prev_rvalue_task_t>);

static_assert(yorch::detail::task_prev_access_valid_v<shared_prev_task_t>);
static_assert(!yorch::detail::task_prev_access_valid_v<invalid_exclusive_mix_task_t>);
static_assert(!yorch::detail::task_prev_access_valid_v<invalid_double_exclusive_task_t>);

static_assert(yorch::detail::task_prev_access_valid_v<const_receiver_task_t>);
static_assert(!yorch::detail::task_prev_access_valid_v<invalid_mutable_receiver_task_t>);

static_assert(yorch::detail::task_prev_access_valid_v<catch_copy_prev_task_t>);
static_assert(yorch::detail::task_uses_copy_prev_v<catch_copy_prev_task_t>);

static_assert(yorch::detail::task_prev_access_valid_v<retry_copy_prev_task_t>);
static_assert(yorch::detail::task_uses_copy_prev_v<retry_copy_prev_task_t>);
static_assert(!yorch::detail::task_prev_access_valid_v<retry_consume_prev_task_t>);

}  // namespace

TEST(ExecutorTest, ForwardPrevPlanValidationRejectsRootAndVoidParentSources) {
    auto invalid_root_tree = yorch::task_tree.root(yorch::bind_forward_prev<int>(
        [](int&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::borrow_prev_mut<int>()));
    using invalid_root_plan_t = yorch::compiled_plan_t<decltype(invalid_root_tree)>;

    auto invalid_void_parent_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> yorch::step_result {
            return yorch::step_result::success();
        }))
        .node<1>(yorch::bind_forward_prev<int>(
            [](int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<int>()));
    using invalid_void_parent_plan_t = yorch::compiled_plan_t<decltype(invalid_void_parent_tree)>;

    static_assert(yorch::detail::validate_plan<invalid_root_plan_t>() ==
                  yorch::detail::plan_validation_error::forward_prev_root_node);
    static_assert(yorch::detail::validate_plan<invalid_void_parent_plan_t>() ==
                  yorch::detail::plan_validation_error::forward_prev_void_parent);
    static_assert(!yorch::detail::plan_forward_prev_source_valid_v<invalid_root_plan_t>);
    static_assert(!yorch::detail::plan_forward_prev_source_valid_v<invalid_void_parent_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanSupportsStaticForwardPrevChainWithoutAllocatingExtraSlots) {
    const forwarded_payload* first_seen = nullptr;
    const forwarded_payload* second_seen = nullptr;
    const forwarded_payload* child_seen = nullptr;
    int child_value = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> forwarded_payload {
            return forwarded_payload {10};
        }))
        .node<1>(yorch::bind_forward_prev<forwarded_payload>(
            [&](forwarded_payload& value) noexcept -> yorch::step_result {
                first_seen = &value;
                value.value += 2;
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<forwarded_payload>()))
            .node<2>(yorch::bind_forward_prev<forwarded_payload>(
                [&](forwarded_payload& value) noexcept -> yorch::step_result {
                    second_seen = &value;
                    value.value += 3;
                    return yorch::step_result::success();
                },
                yorch::borrow_prev_mut<forwarded_payload>()))
                .node<3>(yorch::bind(
                    [&](const forwarded_payload& value) noexcept -> yorch::step_result {
                        child_seen = &value;
                        child_value = value.value;
                        return yorch::step_result::success();
                    },
                    yorch::borrow_prev<forwarded_payload>()));

    auto plan = yorch::compile_plan(tree);
    using plan_t = decltype(plan);

    static_assert(plan_t::template output_storage_mode_for<0> == yorch::detail::output_storage_mode::owned);
    static_assert(plan_t::template output_storage_mode_for<1> == yorch::detail::output_storage_mode::forwarded_prev);
    static_assert(plan_t::template output_storage_mode_for<2> == yorch::detail::output_storage_mode::forwarded_prev);
    static_assert(yorch::plan_exec_slots<plan_t>::physical_slot_count == 1);
    static_assert(yorch::plan_exec_slots<plan_t, yorch::slot_layout_serial_dfs_compact_policy>::physical_slot_count == 1);

    const auto default_result = yorch::run_plan(plan);

    EXPECT_TRUE(default_result.ok());
    ASSERT_NE(first_seen, nullptr);
    ASSERT_NE(second_seen, nullptr);
    ASSERT_NE(child_seen, nullptr);
    EXPECT_EQ(first_seen, second_seen);
    EXPECT_EQ(first_seen, child_seen);
    EXPECT_EQ(child_value, 15);
}

TEST(ExecutorTest, RunPlanSupportsStaticForwardPrevFromConsumePrevRvalueReference) {
    const move_only_forwarded_payload* seen_forward = nullptr;
    const move_only_forwarded_payload* seen_child = nullptr;
    int child_value = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> move_only_forwarded_payload {
            return move_only_forwarded_payload {17};
        }))
        .node<1>(yorch::bind_forward_prev<move_only_forwarded_payload>(
            // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
            [&](move_only_forwarded_payload&& value) noexcept -> yorch::step_result {
                seen_forward = &value;
                value.value += 4;
                return yorch::step_result::success();
            },
            yorch::consume_prev<move_only_forwarded_payload>()))
            .node<2>(yorch::bind(
                [&](const move_only_forwarded_payload& value) noexcept -> yorch::step_result {
                    seen_child = &value;
                    child_value = value.value;
                    return yorch::step_result::success();
                },
                yorch::borrow_prev<move_only_forwarded_payload>()));

    auto plan = yorch::compile_plan(tree);
    using plan_t = decltype(plan);

    static_assert(yorch::plan_exec_slots<plan_t>::physical_slot_count == 1);
    static_assert(yorch::plan_exec_slots<plan_t, yorch::slot_layout_serial_dfs_compact_policy>::physical_slot_count == 1);

    const auto default_result = yorch::run_plan(plan);
    const auto compact_result =
        yorch::run_plan<yorch::slot_layout_serial_dfs_compact_policy>(plan);

    EXPECT_TRUE(default_result.ok());
    EXPECT_TRUE(compact_result.ok());
    ASSERT_NE(seen_forward, nullptr);
    ASSERT_NE(seen_child, nullptr);
    EXPECT_EQ(seen_forward, seen_child);
    EXPECT_EQ(child_value, 21);
}

TEST(ExecutorTest, RunPlanSupportsExclusiveMutablePrevBorrowForSingleChild) {
    int seen_mutated = 0;
    int seen_grandchild = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 5;
        }))
        .node<1>(yorch::bind(
            [&](int& value) noexcept -> int {
                value += 4;
                seen_mutated = value;
                return value;
            },
            yorch::borrow_prev_mut<int>()))
            .node<2>(yorch::bind(
                [&](const int& value) noexcept -> yorch::step_result {
                    seen_grandchild = value;
                    return yorch::step_result::success();
                },
                yorch::borrow_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_mutated, 9);
    EXPECT_EQ(seen_grandchild, 9);
}

TEST(ExecutorTest, RunPlanSupportsBorrowPrevMutReceiverForMemberTask) {
    int seen_child = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> member_prev_worker {
            member_prev_worker worker;
            worker.value = 5;
            return worker;
        }))
        .node<1>(yorch::bind_member(
            &member_prev_worker::mutate,
            yorch::borrow_prev_mut<member_prev_worker>(),
            yorch::value(4)))
        .node<2>(yorch::bind(
            [&](const int& value) noexcept -> yorch::step_result {
                seen_child = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_child, 9);
}

TEST(ExecutorTest, RunPlanSupportsConsumePrevForSingleChildMoveOnlyPayload) {
    int seen_default = 0;
    int seen_compact = 0;

    auto make_tree = [](int& seen) {
        return yorch::task_tree.root(yorch::bind([]() noexcept -> move_only_payload {
                return move_only_payload {17};
            }))
            .node<1>(yorch::bind(
                [&](move_only_payload value) noexcept -> yorch::step_result {
                    seen = value.value;
                    return yorch::step_result::success();
                },
                yorch::consume_prev<move_only_payload>()));
    };

    auto default_plan = yorch::compile_plan(make_tree(seen_default));
    auto compact_plan = yorch::compile_plan(make_tree(seen_compact));

    const auto default_result = yorch::run_plan(default_plan);
    const auto compact_result =
        yorch::run_plan<yorch::slot_layout_serial_dfs_compact_policy>(compact_plan);

    EXPECT_TRUE(default_result.ok());
    EXPECT_TRUE(compact_result.ok());
    EXPECT_EQ(seen_default, 17);
    EXPECT_EQ(seen_compact, 17);
}

TEST(ExecutorTest, RunPlanAllowsReadonlyBorrowFanoutAndRejectsExclusiveAccessFanout) {
    auto valid_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));
    auto valid_plan = yorch::compile_plan(valid_tree);

    auto invalid_ref_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<int>()))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));
    using invalid_ref_plan_t = yorch::compiled_plan_t<decltype(invalid_ref_tree)>;

    auto invalid_value_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
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
    using invalid_value_plan_t = yorch::compiled_plan_t<decltype(invalid_value_tree)>;

    static_assert(can_run_plan<decltype(valid_plan)>);
    static_assert(!can_run_plan<invalid_ref_plan_t>);
    static_assert(!can_run_plan<invalid_value_plan_t>);

    const auto result = yorch::run_plan(valid_plan);
    EXPECT_TRUE(result.ok());
}

TEST(ExecutorTest, RunPlanRejectsReadonlyReceiverForMutableMemberFunction) {
    auto invalid_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> member_prev_worker {
            member_prev_worker worker;
            worker.value = 1;
            return worker;
        }))
        .node<1>(yorch::bind_member(
            &member_prev_worker::mutate,
            yorch::borrow_prev<member_prev_worker>(),
            yorch::value(3)));
    using invalid_plan_t = yorch::compiled_plan_t<decltype(invalid_tree)>;

    static_assert(!can_run_plan<invalid_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanSupportsConsumePrevReceiverForMemberFunction) {
    int seen_child = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> member_prev_worker {
            member_prev_worker worker;
            worker.value = 1;
            return worker;
        }))
        .node<1>(yorch::bind_member(
            &member_prev_worker::mutate,
            yorch::consume_prev<member_prev_worker>(),
            yorch::value(3)))
        .node<2>(yorch::bind(
            [&](const int& value) noexcept -> yorch::step_result {
                seen_child = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));
    auto plan = yorch::compile_plan(tree);

    static_assert(can_run_plan<decltype(plan)>);

    const auto result = yorch::run_plan(plan);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_child, 4);
}

TEST(ExecutorTest, RunPlanAllowsCopyPrevReceiverWithSharedPrevAccess) {
    int seen_total = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> member_prev_worker {
            member_prev_worker worker;
            worker.value = 2;
            return worker;
        }))
        .node<1>(yorch::bind_member(
            &member_prev_worker::read_with_copy,
            yorch::copy_prev<member_prev_worker>(),
            yorch::copy_prev<member_prev_worker>()))
        .node<2>(yorch::bind(
            [&](const int& value) noexcept -> yorch::step_result {
                seen_total = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));
    auto plan = yorch::compile_plan(tree);

    static_assert(can_run_plan<decltype(plan)>);

    const auto result = yorch::run_plan(plan);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_total, 4);
}

TEST(ExecutorTest, RunPlanRejectsConsumePrevReceiverMixedWithAdditionalPrevAccess) {
    auto invalid_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> member_prev_worker {
            member_prev_worker worker;
            worker.value = 1;
            return worker;
        }))
        .node<1>(yorch::bind_member(
            &member_prev_worker::mutate_with_copy,
            yorch::consume_prev<member_prev_worker>(),
            yorch::copy_prev<member_prev_worker>()));
    using invalid_plan_t = yorch::compiled_plan_t<decltype(invalid_tree)>;

    static_assert(!can_run_plan<invalid_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanRejectsBorrowPrevMutReceiverMixedWithSharedPrevAccess) {
    auto invalid_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> member_prev_worker {
            member_prev_worker worker;
            worker.value = 1;
            return worker;
        }))
        .node<1>(yorch::bind_member(
            &member_prev_worker::mutate_from_view,
            yorch::borrow_prev_mut<member_prev_worker>(),
            yorch::borrow_prev<member_prev_worker>()));
    using invalid_plan_t = yorch::compiled_plan_t<decltype(invalid_tree)>;

    static_assert(!can_run_plan<invalid_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanAllowsCopyPrevOnSingleChildAndFanoutPaths) {
    std::vector<int> seen;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 9;
        }))
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
    auto plan = yorch::compile_plan(tree);

    static_assert(can_run_plan<decltype(plan)>);

    const auto result = yorch::run_plan(plan);
    EXPECT_TRUE(result.ok());
    ASSERT_EQ(seen.size(), 2);
    EXPECT_EQ(seen[0], 9);
    EXPECT_EQ(seen[1], 9);
}

TEST(ExecutorTest, RunPlanAllowsMultipleReadonlyBorrowsWithinOneTask) {
    int seen_sum = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 4;
        }))
        .node<1>(yorch::bind(
            [&](const int& lhs, const int& rhs) noexcept -> yorch::step_result {
                seen_sum = lhs + rhs;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>(),
            yorch::borrow_prev<int>()));
    auto plan = yorch::compile_plan(tree);

    static_assert(can_run_plan<decltype(plan)>);

    const auto result = yorch::run_plan(plan);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_sum, 8);
}

TEST(ExecutorTest, RunPlanAllowsMixedSharedPrevAccessWithinOneTask) {
    int copied = 0;
    int borrowed = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 7;
        }))
        .node<1>(yorch::bind(
            [&](const int& ref, int value) noexcept -> yorch::step_result {
                borrowed = ref;
                copied = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>(),
            yorch::copy_prev<int>()));
    auto plan = yorch::compile_plan(tree);

    static_assert(can_run_plan<decltype(plan)>);

    const auto result = yorch::run_plan(plan);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(borrowed, 7);
    EXPECT_EQ(copied, 7);
}

TEST(ExecutorTest, RunPlanAllowsMultipleCopyPrevWithinOneTask) {
    int copied_sum = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 5;
        }))
        .node<1>(yorch::bind(
            [&](int lhs, int rhs) noexcept -> yorch::step_result {
                copied_sum = lhs + rhs;
                return yorch::step_result::success();
            },
            yorch::copy_prev<int>(),
            yorch::copy_prev<int>()));
    auto plan = yorch::compile_plan(tree);

    static_assert(can_run_plan<decltype(plan)>);

    const auto result = yorch::run_plan(plan);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(copied_sum, 10);
}

TEST(ExecutorTest, RunPlanRejectsMixedOrRepeatedExclusivePrevAccessWithinOneTask) {
    auto invalid_mixed_borrow_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](const int&, int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>(),
            yorch::borrow_prev_mut<int>()));
    using invalid_mixed_borrow_plan_t = yorch::compiled_plan_t<decltype(invalid_mixed_borrow_tree)>;

    auto invalid_double_mut_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int&, int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<int>(),
            yorch::borrow_prev_mut<int>()));
    using invalid_double_mut_plan_t = yorch::compiled_plan_t<decltype(invalid_double_mut_tree)>;

    auto invalid_mixed_consume_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int, const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::consume_prev<int>(),
            yorch::borrow_prev<int>()));
    using invalid_mixed_consume_plan_t = yorch::compiled_plan_t<decltype(invalid_mixed_consume_tree)>;

    auto invalid_copy_mut_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int, int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::copy_prev<int>(),
            yorch::borrow_prev_mut<int>()));
    using invalid_copy_mut_plan_t = yorch::compiled_plan_t<decltype(invalid_copy_mut_tree)>;

    auto invalid_copy_consume_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int, int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::copy_prev<int>(),
            yorch::consume_prev<int>()));
    using invalid_copy_consume_plan_t = yorch::compiled_plan_t<decltype(invalid_copy_consume_tree)>;

    static_assert(!can_run_plan<invalid_mixed_borrow_plan_t>);
    static_assert(!can_run_plan<invalid_double_mut_plan_t>);
    static_assert(!can_run_plan<invalid_mixed_consume_plan_t>);
    static_assert(!can_run_plan<invalid_copy_mut_plan_t>);
    static_assert(!can_run_plan<invalid_copy_consume_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanRejectsAnyPrevAccessOnRootNode) {
    auto invalid_root_tree = yorch::task_tree.root(yorch::bind(
        [](const int&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::borrow_prev<int>()));
    using invalid_root_plan_t = yorch::compiled_plan_t<decltype(invalid_root_tree)>;

    auto invalid_root_mut_tree = yorch::task_tree.root(yorch::bind(
        [](int&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::borrow_prev_mut<int>()));
    using invalid_root_mut_plan_t = yorch::compiled_plan_t<decltype(invalid_root_mut_tree)>;

    auto invalid_root_consume_tree = yorch::task_tree.root(yorch::bind(
        [](int) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::consume_prev<int>()));
    using invalid_root_consume_plan_t = yorch::compiled_plan_t<decltype(invalid_root_consume_tree)>;

    auto invalid_root_copy_tree = yorch::task_tree.root(yorch::bind(
        [](int) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::copy_prev<int>()));
    using invalid_root_copy_plan_t = yorch::compiled_plan_t<decltype(invalid_root_copy_tree)>;

    static_assert(yorch::detail::validate_plan<invalid_root_plan_t>() ==
                  yorch::detail::plan_validation_error::prev_access_root_node);
    static_assert(!can_run_plan<invalid_root_plan_t>);
    static_assert(!can_run_plan<invalid_root_mut_plan_t>);
    static_assert(!can_run_plan<invalid_root_consume_plan_t>);
    static_assert(!can_run_plan<invalid_root_copy_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanRejectsAnyPrevAccessWhenDirectParentOutputIsVoid) {
    auto invalid_void_parent_tree = yorch::task_tree.root(yorch::bind([]() noexcept {}))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));
    using invalid_void_parent_plan_t = yorch::compiled_plan_t<decltype(invalid_void_parent_tree)>;

    auto invalid_void_parent_mut_tree = yorch::task_tree.root(yorch::bind([]() noexcept {}))
        .node<1>(yorch::bind(
            [](int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<int>()));
    using invalid_void_parent_mut_plan_t = yorch::compiled_plan_t<decltype(invalid_void_parent_mut_tree)>;

    auto invalid_void_parent_consume_tree = yorch::task_tree.root(yorch::bind([]() noexcept {}))
        .node<1>(yorch::bind(
            [](int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::consume_prev<int>()));
    using invalid_void_parent_consume_plan_t = yorch::compiled_plan_t<decltype(invalid_void_parent_consume_tree)>;

    auto invalid_void_parent_copy_tree = yorch::task_tree.root(yorch::bind([]() noexcept {}))
        .node<1>(yorch::bind(
            [](int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::copy_prev<int>()));
    using invalid_void_parent_copy_plan_t = yorch::compiled_plan_t<decltype(invalid_void_parent_copy_tree)>;

    static_assert(yorch::detail::validate_plan<invalid_void_parent_plan_t>() ==
                  yorch::detail::plan_validation_error::prev_access_void_parent);
    static_assert(!can_run_plan<invalid_void_parent_plan_t>);
    static_assert(!can_run_plan<invalid_void_parent_mut_plan_t>);
    static_assert(!can_run_plan<invalid_void_parent_consume_plan_t>);
    static_assert(!can_run_plan<invalid_void_parent_copy_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanRejectsConsumePrevInsideRetryAdapter) {
    auto invalid_retry_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::with_retry(
            yorch::bind(
                [](int value) noexcept -> yorch::step_result {
                    return value > 0
                        ? yorch::step_result::success()
                        : yorch::step_result::failure();
                },
                yorch::consume_prev<int>()),
            yorch::retry_fixed_policy {1}));
    using invalid_retry_plan_t = yorch::compiled_plan_t<decltype(invalid_retry_tree)>;

    static_assert(yorch::detail::validate_plan<invalid_retry_plan_t>() ==
                  yorch::detail::plan_validation_error::prev_access_mode_invalid);
    static_assert(!can_run_plan<invalid_retry_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanRejectsConsumePrevReceiverInsideRetryAdapter) {
    auto invalid_retry_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> member_prev_worker {
            member_prev_worker worker;
            worker.value = 3;
            return worker;
        }))
        .node<1>(yorch::with_retry(
            yorch::bind_member(
                &member_prev_worker::mutate,
                yorch::consume_prev<member_prev_worker>(),
                yorch::value(2)),
            yorch::retry_fixed_policy {1}));
    using invalid_retry_plan_t = yorch::compiled_plan_t<decltype(invalid_retry_tree)>;

    static_assert(yorch::detail::validate_plan<invalid_retry_plan_t>() ==
                  yorch::detail::plan_validation_error::prev_access_mode_invalid);
    static_assert(!can_run_plan<invalid_retry_plan_t>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanAllowsCopyPrevInsideRetryAdapter) {
    int attempts = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 3;
        }))
        .node<1>(yorch::with_retry(
            yorch::bind(
                [&](int value) noexcept -> yorch::step_result {
                    ++attempts;
                    return attempts == 1 && value == 3
                        ? yorch::step_result::retry()
                        : yorch::step_result::success();
                },
                yorch::copy_prev<int>()),
            yorch::retry_fixed_policy {2}));
    auto plan = yorch::compile_plan(tree);

    static_assert(can_run_plan<decltype(plan)>);

    const auto result = yorch::run_plan(plan);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(attempts, 2);
}
