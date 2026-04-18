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

}  // namespace

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
    auto invalid_ref_plan = yorch::compile_plan(invalid_ref_tree);

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
    auto invalid_value_plan = yorch::compile_plan(invalid_value_tree);

    static_assert(can_run_plan<decltype(valid_plan)>);
    static_assert(!can_run_plan<decltype(invalid_ref_plan)>);
    static_assert(!can_run_plan<decltype(invalid_value_plan)>);

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
    auto invalid_plan = yorch::compile_plan(invalid_tree);

    static_assert(!can_run_plan<decltype(invalid_plan)>);
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
    auto invalid_plan = yorch::compile_plan(invalid_tree);

    static_assert(!can_run_plan<decltype(invalid_plan)>);
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
    auto invalid_plan = yorch::compile_plan(invalid_tree);

    static_assert(!can_run_plan<decltype(invalid_plan)>);
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
    auto invalid_mixed_borrow_plan = yorch::compile_plan(invalid_mixed_borrow_tree);

    auto invalid_double_mut_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int&, int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<int>(),
            yorch::borrow_prev_mut<int>()));
    auto invalid_double_mut_plan = yorch::compile_plan(invalid_double_mut_tree);

    auto invalid_mixed_consume_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int, const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::consume_prev<int>(),
            yorch::borrow_prev<int>()));
    auto invalid_mixed_consume_plan = yorch::compile_plan(invalid_mixed_consume_tree);

    auto invalid_copy_mut_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int, int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::copy_prev<int>(),
            yorch::borrow_prev_mut<int>()));
    auto invalid_copy_mut_plan = yorch::compile_plan(invalid_copy_mut_tree);

    auto invalid_copy_consume_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind(
            [](int, int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::copy_prev<int>(),
            yorch::consume_prev<int>()));
    auto invalid_copy_consume_plan = yorch::compile_plan(invalid_copy_consume_tree);

    static_assert(!can_run_plan<decltype(invalid_mixed_borrow_plan)>);
    static_assert(!can_run_plan<decltype(invalid_double_mut_plan)>);
    static_assert(!can_run_plan<decltype(invalid_mixed_consume_plan)>);
    static_assert(!can_run_plan<decltype(invalid_copy_mut_plan)>);
    static_assert(!can_run_plan<decltype(invalid_copy_consume_plan)>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanRejectsAnyPrevAccessOnRootNode) {
    auto invalid_root_tree = yorch::task_tree.root(yorch::bind(
        [](const int&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::borrow_prev<int>()));
    auto invalid_root_plan = yorch::compile_plan(invalid_root_tree);

    auto invalid_root_mut_tree = yorch::task_tree.root(yorch::bind(
        [](int&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::borrow_prev_mut<int>()));
    auto invalid_root_mut_plan = yorch::compile_plan(invalid_root_mut_tree);

    auto invalid_root_consume_tree = yorch::task_tree.root(yorch::bind(
        [](int) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::consume_prev<int>()));
    auto invalid_root_consume_plan = yorch::compile_plan(invalid_root_consume_tree);

    auto invalid_root_copy_tree = yorch::task_tree.root(yorch::bind(
        [](int) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::copy_prev<int>()));
    auto invalid_root_copy_plan = yorch::compile_plan(invalid_root_copy_tree);

    static_assert(!can_run_plan<decltype(invalid_root_plan)>);
    static_assert(!can_run_plan<decltype(invalid_root_mut_plan)>);
    static_assert(!can_run_plan<decltype(invalid_root_consume_plan)>);
    static_assert(!can_run_plan<decltype(invalid_root_copy_plan)>);
    SUCCEED();
}

TEST(ExecutorTest, RunPlanRejectsAnyPrevAccessWhenDirectParentOutputIsVoid) {
    auto invalid_void_parent_tree = yorch::task_tree.root(yorch::bind([]() noexcept {}))
        .node<1>(yorch::bind(
            [](const int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));
    auto invalid_void_parent_plan = yorch::compile_plan(invalid_void_parent_tree);

    auto invalid_void_parent_mut_tree = yorch::task_tree.root(yorch::bind([]() noexcept {}))
        .node<1>(yorch::bind(
            [](int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<int>()));
    auto invalid_void_parent_mut_plan = yorch::compile_plan(invalid_void_parent_mut_tree);

    auto invalid_void_parent_consume_tree = yorch::task_tree.root(yorch::bind([]() noexcept {}))
        .node<1>(yorch::bind(
            [](int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::consume_prev<int>()));
    auto invalid_void_parent_consume_plan = yorch::compile_plan(invalid_void_parent_consume_tree);

    auto invalid_void_parent_copy_tree = yorch::task_tree.root(yorch::bind([]() noexcept {}))
        .node<1>(yorch::bind(
            [](int) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::copy_prev<int>()));
    auto invalid_void_parent_copy_plan = yorch::compile_plan(invalid_void_parent_copy_tree);

    static_assert(!can_run_plan<decltype(invalid_void_parent_plan)>);
    static_assert(!can_run_plan<decltype(invalid_void_parent_mut_plan)>);
    static_assert(!can_run_plan<decltype(invalid_void_parent_consume_plan)>);
    static_assert(!can_run_plan<decltype(invalid_void_parent_copy_plan)>);
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
    auto invalid_retry_plan = yorch::compile_plan(invalid_retry_tree);

    static_assert(!can_run_plan<decltype(invalid_retry_plan)>);
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
    auto invalid_retry_plan = yorch::compile_plan(invalid_retry_tree);

    static_assert(!can_run_plan<decltype(invalid_retry_plan)>);
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
