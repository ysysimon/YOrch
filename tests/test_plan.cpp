#include "yorch/plan.hpp"

#include "yorch/bind.hpp"
#include "yorch/task_adapters.hpp"

#include <gtest/gtest.h>

#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace {

template <typename TaskTree>
concept can_compile_plan =
    requires(TaskTree&& task_tree) {
        yorch::compile_plan(std::forward<TaskTree>(task_tree));
    };

constexpr auto make_void_task() {
    return yorch::bind([]() noexcept {});
}

struct rawless_task {
    void invoke_raw(yorch::exec_context<void>&) noexcept {}
};

} // namespace

TEST(PlanTest, CompilePlanRecoversParentChildStructureAndOutputMetadata) {
    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind([]() noexcept -> yorch::task_result<std::string> {
            return yorch::task_result<std::string>::success("child");
        }))
            .node<2>(yorch::bind([]() noexcept -> yorch::step_result {
                return yorch::step_result::abort_branch();
            }))
        .node<1>(yorch::bind([]() noexcept {}))
            .node<2>(yorch::bind([]() noexcept -> bool {
                return true;
            }));

    auto plan = yorch::compile_plan(tree);
    using plan_t = decltype(plan);

    static_assert(plan_t::node_count == 5);
    static_assert(plan_t::max_level == 2);
    static_assert(plan_t::no_parent == 5);
    static_assert(plan_t::slot_count == 5);

    static_assert(plan_t::template parent_index<0> == plan_t::no_parent);
    static_assert(plan_t::template parent_index<1> == 0);
    static_assert(plan_t::template parent_index<2> == 1);
    static_assert(plan_t::template parent_index<3> == 0);
    static_assert(plan_t::template parent_index<4> == 3);

    static_assert(plan_t::template child_count<0> == 2);
    static_assert(plan_t::template child_count<1> == 1);
    static_assert(plan_t::template child_count<2> == 0);
    static_assert(plan_t::template child_count<3> == 1);
    static_assert(plan_t::template child_count<4> == 0);

    static_assert(plan_t::template child_index<0, 0> == 1);
    static_assert(plan_t::template child_index<0, 1> == 3);
    static_assert(plan_t::template child_index<1, 0> == 2);
    static_assert(plan_t::template child_index<3, 0> == 4);

    static_assert(plan_t::template slot_index<0> == 0);
    static_assert(plan_t::template slot_index<4> == 4);

    static_assert(std::is_same_v<plan_t::template output_type<0>, int>);
    static_assert(std::is_same_v<plan_t::template output_type<1>, std::string>);
    static_assert(std::is_same_v<plan_t::template output_type<2>, void>);
    static_assert(std::is_same_v<plan_t::template output_type<3>, void>);
    static_assert(std::is_same_v<plan_t::template output_type<4>, bool>);
    static_assert(plan_t::template slot_logical_policy_for<0> == yorch::detail::slot_logical_policy::must_payload);
    static_assert(plan_t::template slot_logical_policy_for<1> == yorch::detail::slot_logical_policy::maybe_payload);
    static_assert(plan_t::template slot_logical_policy_for<2> == yorch::detail::slot_logical_policy::none);
    static_assert(plan_t::template slot_logical_policy_for<3> == yorch::detail::slot_logical_policy::none);
    static_assert(plan_t::template slot_logical_policy_for<4> == yorch::detail::slot_logical_policy::must_payload);

    EXPECT_EQ(plan_t::parent_indices[0], plan_t::no_parent);
    EXPECT_EQ(plan_t::parent_indices[1], 0);
    EXPECT_EQ(plan_t::parent_indices[2], 1);
    EXPECT_EQ(plan_t::parent_indices[3], 0);
    EXPECT_EQ(plan_t::parent_indices[4], 3);

    EXPECT_EQ(plan_t::child_offsets[0], 0);
    EXPECT_EQ(plan_t::child_offsets[1], 2);
    EXPECT_EQ(plan_t::child_offsets[2], 3);
    EXPECT_EQ(plan_t::child_offsets[3], 3);
    EXPECT_EQ(plan_t::child_offsets[4], 4);

    ASSERT_EQ(plan_t::child_indices.size(), 4U);
    EXPECT_EQ(plan_t::child_indices[0], 1);
    EXPECT_EQ(plan_t::child_indices[1], 3);
    EXPECT_EQ(plan_t::child_indices[2], 2);
    EXPECT_EQ(plan_t::child_indices[3], 4);
}

TEST(PlanTest, CompilePlanPreservesMoveOnlyTasksWhenConsumingBuilder) {
    auto tree = yorch::task_tree.root(yorch::bind(
            [](const std::unique_ptr<int>& value) noexcept -> int {
                return value ? *value : -1;
            },
            yorch::value(std::make_unique<int>(9))))
        .node<1>(make_void_task());

    auto plan = yorch::compile_plan(std::move(tree));

    static_assert(decltype(plan)::node_count == 2);

    auto& root_task = plan.template entry<0>().task;
    auto& spec = std::get<0>(root_task.specs);

    ASSERT_NE(spec.v, nullptr);
    EXPECT_EQ(*spec.v, 9);
}

TEST(PlanTest, CompiledPlanAliasMapsTaskTreeBuilderToPlanType) {
    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 7;
        }))
        .node<1>(make_void_task());

    using tree_t = decltype(tree);
    using plan_t = yorch::compiled_plan_t<tree_t>;

    static_assert(std::is_same_v<plan_t, decltype(yorch::compile_plan(tree))>);
    static_assert(plan_t::node_count == 2);

    SUCCEED();
}

TEST(PlanTest, CompilePlanRejectsEmptyTaskTree) {
    static_assert(!can_compile_plan<decltype(yorch::task_tree)&>,
                  "compile_plan should reject an empty task_tree_builder");

    SUCCEED();
}

TEST(PlanTest, CompilePlanRejectsTasksWithoutDeclaredRawResultType) {
    auto tree = yorch::task_tree.root(rawless_task {});

    static_assert(!can_compile_plan<decltype(tree)&>,
                  "compile_plan should reject tasks that do not expose raw_result_type");

    SUCCEED();
}

TEST(PlanTest, CompilePlanPrefersDeclaredTaskOutputTypeForDirectOutputTasks) {
    auto tree = yorch::task_tree.root(yorch::bind_into<std::string>(
            [](yorch::result_out<std::string> out) noexcept -> yorch::step_result {
                return out.success("root");
            }))
        .node<1>(yorch::bind_into<int>(
            [](const std::string& value, yorch::result_out<int> out) noexcept -> yorch::step_result {
                return out.success(static_cast<int>(value.size()));
            },
            yorch::from_prev<std::string>()));

    auto plan = yorch::compile_plan(tree);
    using plan_t = decltype(plan);

    static_assert(std::is_same_v<plan_t::template raw_result_type<0>, yorch::step_result>);
    static_assert(std::is_same_v<plan_t::template raw_result_type<1>, yorch::step_result>);
    static_assert(std::is_same_v<plan_t::template output_type<0>, std::string>);
    static_assert(std::is_same_v<plan_t::template output_type<1>, int>);
    static_assert(plan_t::template slot_logical_policy_for<0> == yorch::detail::slot_logical_policy::maybe_payload);
    static_assert(plan_t::template slot_logical_policy_for<1> == yorch::detail::slot_logical_policy::maybe_payload);

    SUCCEED();
}

TEST(PlanTest, CompilePlanInfersSlotPoliciesFromFinalTaskProtocols) {
    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind([]() noexcept -> yorch::task_result<std::string> {
            return yorch::task_result<std::string>::success("child");
        }))
        .node<1>(yorch::bind([]() noexcept -> yorch::step_result {
            return yorch::step_result::success();
        }))
        .node<1>(yorch::bind([]() noexcept {}))
        .node<1>(yorch::bind_into<int>(
            [](yorch::result_out<int> out) noexcept -> yorch::step_result {
                return out.success(3);
            }))
        .node<1>(yorch::with_retry(
            yorch::bind([]() noexcept -> bool {
                return true;
            }),
            yorch::retry_fixed_policy {1}))
        .node<1>(yorch::catch_as_failure(
            yorch::bind([]() -> int {
                throw std::runtime_error("boom");
            }),
            [](const std::exception_ptr&) noexcept -> int {
                return 7;
            }));

    auto plan = yorch::compile_plan(tree);
    using plan_t = decltype(plan);

    static_assert(plan_t::template slot_logical_policy_for<0> == yorch::detail::slot_logical_policy::must_payload);
    static_assert(plan_t::template slot_logical_policy_for<1> == yorch::detail::slot_logical_policy::maybe_payload);
    static_assert(plan_t::template slot_logical_policy_for<2> == yorch::detail::slot_logical_policy::none);
    static_assert(plan_t::template slot_logical_policy_for<3> == yorch::detail::slot_logical_policy::none);
    static_assert(plan_t::template slot_logical_policy_for<4> == yorch::detail::slot_logical_policy::maybe_payload);
    static_assert(plan_t::template slot_logical_policy_for<5> == yorch::detail::slot_logical_policy::must_payload);
    static_assert(plan_t::template slot_logical_policy_for<6> == yorch::detail::slot_logical_policy::must_payload);

    SUCCEED();
}
