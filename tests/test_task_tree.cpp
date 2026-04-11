#include "yorch/task_tree.hpp"

#include "yorch/bind.hpp"
#include "yorch/executor.hpp"
#include "yorch/plan.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr auto make_noop_task() {
    return yorch::bind([]() noexcept {});
}

struct ordinary_callable {
    void operator()() const noexcept {}
};

template <typename TaskTree, std::size_t Level>
concept can_append_noop =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).template node<Level>(make_noop_task());
    };

template <typename TaskTree, std::size_t Level>
concept can_append_noop_with_fanout =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).template node<Level>(
            make_noop_task(),
            yorch::fanout_shared_readonly_policy {});
    };

template <typename TaskTree>
concept can_append_root =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root(make_noop_task());
    };

template <typename TaskTree>
concept can_append_root_with_fanout =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root(
            make_noop_task(),
            yorch::fanout_shared_readonly_policy {});
    };

template <typename TaskTree>
concept can_append_root_callable =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root([]() noexcept {})();
    };

template <typename TaskTree>
concept can_append_root_callable_into =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree)
            .root_into([](yorch::direct_out<int>) noexcept {})();
    };

template <typename TaskTree, std::size_t Level>
concept can_append_node_callable_into =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree)
            .template node_into<Level>([](yorch::direct_out<int>) noexcept {})();
    };

} // namespace

TEST(TaskTreeTest, RootBuilderCapturesCompileTimeLevelSequence) {
    auto s = yorch::task_tree.root(make_noop_task())
        .node<1>(make_noop_task())
            .node<2>(make_noop_task())
        .node<1>(make_noop_task())
            .node<2>(make_noop_task());

    using task_tree_t = decltype(s);

    static_assert(task_tree_t::node_count == 5);
    static_assert(task_tree_t::max_level == 2);
    static_assert(task_tree_t::template node_type<0>::level == 0);
    static_assert(task_tree_t::template node_type<1>::level == 1);
    static_assert(task_tree_t::template node_type<2>::level == 2);
    static_assert(task_tree_t::template node_type<3>::level == 1);
    static_assert(task_tree_t::template node_type<4>::level == 2);

    EXPECT_EQ(task_tree_t::node_count, 5);
    EXPECT_EQ(task_tree_t::max_level, 2);
}

TEST(TaskTreeTest, BuilderCapturesFanoutPolicyMetadata) {
    auto s = yorch::task_tree
        .root(make_noop_task(), yorch::fanout_shared_readonly_policy {})
        .node<1>(make_noop_task())
        .node<1>(make_noop_task(), yorch::fanout_consume_with_copies_policy {});

    using task_tree_t = decltype(s);

    static_assert(std::is_same_v<
                  task_tree_t::template node_type<0>::fanout_policy_type,
                  yorch::fanout_shared_readonly_policy>);
    static_assert(std::is_same_v<
                  task_tree_t::template node_type<1>::fanout_policy_type,
                  yorch::fanout_auto_policy>);
    static_assert(std::is_same_v<
                  task_tree_t::template node_type<2>::fanout_policy_type,
                  yorch::fanout_consume_with_copies_policy>);

    SUCCEED();
}

TEST(TaskTreeTest, RootBuilderPreservesMoveOnlyTaskStorageAcrossRvalueChaining) {
    auto s = yorch::task_tree.root(yorch::bind(
            [](const std::unique_ptr<int>& value) noexcept -> int {
                return value ? *value : -1;
            },
            yorch::value(std::make_unique<int>(7))))
        .node<1>(make_noop_task());

    static_assert(decltype(s)::node_count == 2);

    auto& root_task = s.entry<0>().task;
    auto& spec = std::get<0>(root_task.specs);

    ASSERT_NE(spec.v, nullptr);
    EXPECT_EQ(*spec.v, 7);
}

TEST(TaskTreeTest, RootCallableIntoPreservesMoveOnlyTaskStorageAcrossRvalueChaining) {
    auto s = yorch::task_tree
        .root_into([](const std::unique_ptr<int>& value,
                 yorch::direct_out<int> out) noexcept -> yorch::step_result {
            return out.success(value ? *value : -1);
        })(yorch::value(std::make_unique<int>(7)))
        .node<1>(make_noop_task());

    static_assert(decltype(s)::node_count == 2);

    auto& root_task = s.entry<0>().task;
    auto& spec = std::get<0>(root_task.specs);

    ASSERT_NE(spec.v, nullptr);
    EXPECT_EQ(*spec.v, 7);
}

TEST(TaskTreeTest, RootBuilderRejectsInvalidLevelTransitions) {
    using root_task_tree_t =
        decltype(yorch::task_tree.root(make_noop_task()));
    using depth_one_task_tree_t =
        decltype(yorch::task_tree.root(make_noop_task()).node<1>(make_noop_task()));
    using root_callable_into_task_tree_t =
        decltype(yorch::task_tree.root_into([](yorch::direct_out<int>) noexcept {})());

    static_assert(can_append_root<decltype(yorch::task_tree)>,
                  "empty task_tree should allow appending a root node");
    static_assert(can_append_root_with_fanout<decltype(yorch::task_tree)>,
                  "empty task_tree should allow appending a root node with fanout policy");
    static_assert(can_append_root_callable<decltype(yorch::task_tree)>,
                  "empty task_tree should allow callable root sugar");
    static_assert(can_append_root_callable_into<decltype(yorch::task_tree)>,
                  "empty task_tree should allow callable root direct-output sugar");
    static_assert(!yorch::detail::direct_output_callable_task_argument<ordinary_callable>,
                  "root_into(...) should reject ordinary callables without direct_out<T>");
    static_assert(!can_append_root<root_task_tree_t&>,
                  "task_tree should reject adding a second root after the first node");
    static_assert(can_append_noop<decltype(yorch::task_tree), 0>,
                  "empty task_tree should still admit node<0>(...) as the root-equivalent entry");
    static_assert(can_append_noop_with_fanout<decltype(yorch::task_tree), 0>,
                  "empty task_tree should admit node<0>(..., fanout_policy) as the root-equivalent entry");
    static_assert(!can_append_noop<decltype(yorch::task_tree), 1>,
                  "empty task_tree should reject node<1>(...) because a root must come first");
    static_assert(!can_append_noop<root_task_tree_t&, 0>,
                  "non-empty task_tree should reject another node<0>(...) root");
    static_assert(can_append_noop<root_task_tree_t&, 1>,
                  "root task_tree should allow descending one level to node<1>(...)");
    static_assert(can_append_noop_with_fanout<root_task_tree_t&, 1>,
                  "root task_tree should allow descending one level to node<1>(..., fanout_policy)");
    static_assert(!can_append_noop<root_task_tree_t&, 2>,
                  "root task_tree should reject skipping directly from level 0 to level 2");
    static_assert(!can_append_node_callable_into<decltype(yorch::task_tree), 1>,
                  "empty task_tree should reject node_into<1>(callable) because a root must come first");
    static_assert(can_append_node_callable_into<root_callable_into_task_tree_t&, 1>,
                  "root direct-output task_tree should allow descending one level to callable node direct-output sugar");
    static_assert(!can_append_node_callable_into<root_callable_into_task_tree_t&, 2>,
                  "root direct-output task_tree should reject skipping directly from level 0 to level 2");
    static_assert(can_append_noop<depth_one_task_tree_t&, 1>,
                  "a depth-one task_tree should allow adding a sibling at the same level");
    static_assert(can_append_noop<depth_one_task_tree_t&, 2>,
                  "a depth-one task_tree should allow descending one additional level");
    static_assert(!can_append_noop<depth_one_task_tree_t&, 3>,
                  "a depth-one task_tree should reject descending by more than one level");

    SUCCEED();
}

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
