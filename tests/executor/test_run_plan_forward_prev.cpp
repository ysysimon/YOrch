#include "prev_access_support.hpp"

using namespace executor_test_support;

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

TEST(ExecutorTest, ForwardPrevSugarPlanValidationRejectsRootNodes) {
    auto invalid_root_tree = yorch::task_tree.root_forward_prev(
        [](int&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        })(
        yorch::borrow_prev_mut<int>());
    using invalid_root_plan_t = yorch::compiled_plan_t<decltype(invalid_root_tree)>;

    auto invalid_member_root_tree = yorch::task_tree.root_forward_prev_member(
        &member_forward_prev_worker::mutate_self,
        yorch::borrow_prev_mut<member_forward_prev_worker>())(
        yorch::value(1));
    using invalid_member_root_plan_t = yorch::compiled_plan_t<decltype(invalid_member_root_tree)>;

    static_assert(yorch::detail::validate_plan<invalid_root_plan_t>() ==
                  yorch::detail::plan_validation_error::forward_prev_root_node);
    static_assert(yorch::detail::validate_plan<invalid_member_root_plan_t>() ==
                  yorch::detail::plan_validation_error::forward_prev_root_node);
    SUCCEED();
}

TEST(ExecutorTest, MemberForwardPrevPlanValidationRejectsRootVoidParentAndTypeMismatch) {
    auto invalid_root_tree = yorch::task_tree.root(yorch::bind_forward_prev_member<int>(
        &member_forward_prev_service::bump_int,
        yorch::value(member_forward_prev_service {}),
        yorch::borrow_prev_mut<int>(),
        yorch::value(1)));
    using invalid_root_plan_t = yorch::compiled_plan_t<decltype(invalid_root_tree)>;

    auto invalid_void_parent_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> yorch::step_result {
            return yorch::step_result::success();
        }))
        .node<1>(yorch::bind_forward_prev_member<int>(
            &member_forward_prev_service::bump_int,
            yorch::value(member_forward_prev_service {}),
            yorch::borrow_prev_mut<int>(),
            yorch::value(1)));
    using invalid_void_parent_plan_t = yorch::compiled_plan_t<decltype(invalid_void_parent_tree)>;

    auto invalid_type_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> forwarded_payload {
            return forwarded_payload {4};
        }))
        .node<1>(yorch::bind_forward_prev_member<int>(
            &member_forward_prev_service::bump_int,
            yorch::value(member_forward_prev_service {}),
            yorch::borrow_prev_mut<int>(),
            yorch::value(1)));
    using invalid_type_plan_t = yorch::compiled_plan_t<decltype(invalid_type_tree)>;

    static_assert(yorch::detail::validate_plan<invalid_root_plan_t>() ==
                  yorch::detail::plan_validation_error::forward_prev_root_node);
    static_assert(yorch::detail::validate_plan<invalid_void_parent_plan_t>() ==
                  yorch::detail::plan_validation_error::forward_prev_void_parent);
    static_assert(yorch::detail::validate_plan<invalid_type_plan_t>() ==
                  yorch::detail::plan_validation_error::forward_prev_parent_type_mismatch);
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

TEST(ExecutorTest, RunPlanSupportsForwardPrevSugarWithoutAllocatingExtraSlots) {
    const forwarded_payload* forward_seen = nullptr;
    const forwarded_payload* child_seen = nullptr;
    int child_value = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> forwarded_payload {
            return forwarded_payload {10};
        }))
        .node_forward_prev<1>(
            [&](forwarded_payload& value) noexcept -> yorch::step_result {
                forward_seen = &value;
                value.value += 2;
                return yorch::step_result::success();
            })(
            yorch::borrow_prev_mut<forwarded_payload>())
        .node<2>(yorch::bind(
            [&](const forwarded_payload& value) noexcept -> yorch::step_result {
                child_seen = &value;
                child_value = value.value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<forwarded_payload>()));

    auto plan = yorch::compile_plan(tree);
    using plan_t = decltype(plan);

    static_assert(plan_t::template output_storage_mode_for<1> == yorch::detail::output_storage_mode::forwarded_prev);
    static_assert(yorch::plan_exec_slots<plan_t>::physical_slot_count == 1);

    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    ASSERT_NE(forward_seen, nullptr);
    ASSERT_NE(child_seen, nullptr);
    EXPECT_EQ(forward_seen, child_seen);
    EXPECT_EQ(child_value, 12);
}

TEST(ExecutorTest, RunPlanSupportsMemberForwardPrevSugarWithoutAllocatingExtraSlots) {
    const member_forward_prev_worker* child_seen = nullptr;
    int child_value = 0;

    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> member_forward_prev_worker {
            return member_forward_prev_worker {.state = 10};
        }))
        .node_forward_prev_member<1>(
            &member_forward_prev_worker::mutate_self,
            yorch::borrow_prev_mut<member_forward_prev_worker>())(
            yorch::value(2))
        .node<2>(yorch::bind(
            [&](const member_forward_prev_worker& value) noexcept -> yorch::step_result {
                child_seen = &value;
                child_value = value.state;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<member_forward_prev_worker>()));

    auto plan = yorch::compile_plan(tree);
    using plan_t = decltype(plan);

    static_assert(plan_t::template output_storage_mode_for<1> == yorch::detail::output_storage_mode::forwarded_prev);
    static_assert(yorch::plan_exec_slots<plan_t>::physical_slot_count == 1);

    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    ASSERT_NE(child_seen, nullptr);
    EXPECT_EQ(child_value, 12);
}

TEST(ExecutorTest, RunPlanSupportsMemberForwardPrevWithoutAllocatingExtraSlots) {
    const member_forward_prev_worker* child_seen = nullptr;
    int child_value = 0;

    auto runnable_tree = yorch::task_tree.root(yorch::bind([]() noexcept -> member_forward_prev_worker {
            return member_forward_prev_worker {.state = 10};
        }))
        .node<1>(yorch::bind_forward_prev_member<member_forward_prev_worker>(
            &member_forward_prev_worker::mutate_self,
            yorch::borrow_prev_mut<member_forward_prev_worker>(),
            yorch::value(2)))
            .node<2>(yorch::bind(
                [&](const member_forward_prev_worker& value) noexcept -> yorch::step_result {
                    child_seen = &value;
                    child_value = value.state;
                    return yorch::step_result::success();
                },
                yorch::borrow_prev<member_forward_prev_worker>()));

    auto plan = yorch::compile_plan(runnable_tree);
    using plan_t = decltype(plan);

    static_assert(plan_t::template output_storage_mode_for<1> == yorch::detail::output_storage_mode::forwarded_prev);
    static_assert(yorch::plan_exec_slots<plan_t>::physical_slot_count == 1);

    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    ASSERT_NE(child_seen, nullptr);
    EXPECT_EQ(child_value, 12);
}
