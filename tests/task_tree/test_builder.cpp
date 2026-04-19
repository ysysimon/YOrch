#include "support.hpp"

using namespace task_tree_test_support;

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
    static_assert(can_append_root_member_sugar<decltype(yorch::task_tree)>,
                  "empty task_tree should allow explicit member root sugar");
    static_assert(can_append_root_into_member_sugar<decltype(yorch::task_tree)>,
                  "empty task_tree should allow explicit member root direct-output sugar");
    static_assert(!can_append_root_member_without_receiver<decltype(yorch::task_tree)>,
                  "root_member(...) should reject missing receiver bindings");
    static_assert(!can_append_root_into_member_without_receiver<decltype(yorch::task_tree)>,
                  "root_into_member(...) should reject missing receiver bindings");
    static_assert(!can_append_root_member_callable<decltype(yorch::task_tree)>,
                  "empty task_tree should reject raw member function pointers in callable sugar");
    static_assert(!can_append_root_member_with_direct_output<decltype(yorch::task_tree)>,
                  "root_member(...) should reject direct-output member functions");
    static_assert(!can_append_root_into_member_with_ordinary<decltype(yorch::task_tree)>,
                  "root_into_member(...) should reject ordinary member functions");
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
    static_assert(can_append_node_member_sugar<root_task_tree_t&, 1>,
                  "root task_tree should allow descending one level to explicit member sugar");
    static_assert(can_append_node_into_member_sugar<root_task_tree_t&, 1>,
                  "root task_tree should allow descending one level to explicit member direct-output sugar");
    static_assert(!can_append_node_member_without_receiver<root_task_tree_t&, 1>,
                  "node_member<Level>(...) should reject missing receiver bindings");
    static_assert(!can_append_node_into_member_without_receiver<root_task_tree_t&, 1>,
                  "node_into_member<Level>(...) should reject missing receiver bindings");
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
