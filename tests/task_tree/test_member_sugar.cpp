#include "support.hpp"

using namespace task_tree_test_support;

TEST(TaskTreeTest, RootMemberAndNodeIntoMemberSugarBuildRunnablePlan) {
    member_tree_worker worker;
    std::string seen_child;

    auto tree = yorch::task_tree
        .root_member(
            &member_tree_worker::seed,
            yorch::value(std::ref(worker)))(
            yorch::value(3))
        .node_into_member<1>(
            &member_tree_worker::emit_text,
            yorch::value(std::ref(worker)))(
            yorch::value(3))
        .node<2>(yorch::bind(
            [&](const std::string& value) noexcept -> yorch::step_result {
                seen_child = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<std::string>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.base, 6);
    EXPECT_EQ(seen_child, "6");
}

TEST(TaskTreeTest, RootIntoMemberSugarBuildsRunnablePlanFromContextReceiver) {
    yorch::context<member_tree_worker> ctx(member_tree_worker {.base = 5});

    auto tree = yorch::task_tree
        .root_into_member(
            &member_tree_worker::emit_text,
            yorch::from_ctx<member_tree_worker>())(
            yorch::value(2))
        .node<1>(yorch::bind(
            [](const std::string& value) noexcept -> int {
                return std::stoi(value);
            },
            yorch::borrow_prev<std::string>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan, ctx);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(ctx.get<member_tree_worker>().base, 7);
}

TEST(TaskTreeTest, NodeMemberSugarSupportsBorrowPrevMutReceiver) {
    int seen_value = 0;

    auto tree = yorch::task_tree
        .root(yorch::bind([]() noexcept -> member_tree_worker {
            member_tree_worker worker;
            worker.base = 8;
            return worker;
        }))
        .node_member<1>(
            &member_tree_worker::seed,
            yorch::borrow_prev_mut<member_tree_worker>())(
            yorch::value(4))
        .node<2>(yorch::bind(
            [&](const int& value) noexcept -> yorch::step_result {
                seen_value = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, 12);
}

TEST(TaskTreeTest, NodeIntoMemberSugarSupportsBorrowPrevReceiver) {
    std::string seen_value;

    auto tree = yorch::task_tree
        .root(yorch::bind([]() noexcept -> move_only_tree_worker {
            return move_only_tree_worker {15};
        }))
        .node_into_member<1>(
            &move_only_tree_worker::emit_text,
            yorch::borrow_prev<move_only_tree_worker>())()
        .node<2>(yorch::bind(
            [&](const std::string& value) noexcept -> yorch::step_result {
                seen_value = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<std::string>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, "15");
}

TEST(TaskTreeTest, NodeForwardPrevMemberSugarSupportsBorrowPrevMutReceiver) {
    int seen_value = 0;

    auto tree = yorch::task_tree
        .root(yorch::bind([]() noexcept -> member_tree_worker {
            member_tree_worker worker;
            worker.base = 8;
            return worker;
        }))
        .node_forward_prev_member<1>(
            &member_tree_worker::mutate_self,
            yorch::borrow_prev_mut<member_tree_worker>())(
            yorch::value(4))
        .node<2>(yorch::bind(
            [&](const member_tree_worker& value) noexcept -> yorch::step_result {
                seen_value = value.base;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<member_tree_worker>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, 12);
}

TEST(TaskTreeTest, NodeForwardPrevMemberSugarSupportsContextReceiverAndPrevPayloadParameter) {
    int seen_value = 0;

    yorch::context<forward_prev_tree_service> ctx(forward_prev_tree_service {});

    auto tree = yorch::task_tree
        .root(yorch::bind([]() noexcept -> forward_prev_tree_payload {
            return forward_prev_tree_payload {6};
        }))
        .node_forward_prev_member<1>(
            &forward_prev_tree_service::bump,
            yorch::from_ctx<forward_prev_tree_service>())(
            yorch::borrow_prev_mut<forward_prev_tree_payload>(),
            yorch::value(5))
        .node<2>(yorch::bind(
            [&](const forward_prev_tree_payload& value) noexcept -> yorch::step_result {
                seen_value = value.value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<forward_prev_tree_payload>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan, ctx);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, 11);
    EXPECT_EQ(ctx.get<forward_prev_tree_service>().seen_value, 6);
}

TEST(TaskTreeTest, PreboundMemberTasksBuildRunnablePlan) {
    member_tree_worker worker;
    std::string seen_child;

    auto tree = yorch::task_tree
        .root(yorch::bind_member(
            &member_tree_worker::seed,
            yorch::value(std::ref(worker)),
            yorch::value(3)))
        .node_into<1>(yorch::bind_into_member<std::string>(
            &member_tree_worker::emit_text,
            yorch::value(std::ref(worker)),
            yorch::value(3)))
        .node<2>(yorch::bind(
            [&](const std::string& value) noexcept -> yorch::step_result {
                seen_child = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<std::string>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.base, 6);
    EXPECT_EQ(seen_child, "6");
}

TEST(TaskTreeTest, PreboundMemberTaskSupportsCopyPrevReceiver) {
    int seen_value = 0;

    auto tree = yorch::task_tree
        .root(yorch::bind([]() noexcept -> member_tree_worker {
            member_tree_worker worker;
            worker.base = 8;
            return worker;
        }))
        .node<1>(yorch::bind_member(
            &member_tree_worker::seed,
            yorch::copy_prev<member_tree_worker>(),
            yorch::value(4)))
        .node<2>(yorch::bind(
            [&](const int& value) noexcept -> yorch::step_result {
                seen_value = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<int>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, 12);
}

TEST(TaskTreeTest, PreboundMemberDirectOutputTaskSupportsConsumePrevReceiver) {
    std::string seen_value;

    auto tree = yorch::task_tree
        .root(yorch::bind([]() noexcept -> move_only_tree_worker {
            return move_only_tree_worker {15};
        }))
        .node_into<1>(yorch::bind_into_member<std::string>(
            &move_only_tree_worker::emit_text,
            yorch::consume_prev<move_only_tree_worker>()))
        .node<2>(yorch::bind(
            [&](const std::string& value) noexcept -> yorch::step_result {
                seen_value = value;
                return yorch::step_result::success();
            },
            yorch::borrow_prev<std::string>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(seen_value, "15");
}
