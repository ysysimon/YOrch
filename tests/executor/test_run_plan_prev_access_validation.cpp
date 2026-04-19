#include "prev_access_support.hpp"

using namespace executor_test_support;

namespace {

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
