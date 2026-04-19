#include "support.hpp"

using namespace bind_test_support;

TEST(BindTest, TaskSugarBuildsBoundTaskFromCallableAndSpecs) {
    yorch::context<int, std::string> ctx(3, "job");
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::task(
        recorder {})(
        yorch::from_ctx<int>(),
        yorch::from_ctx<std::string>(),
        yorch::value(4));

    const auto result = task.invoke_raw(exec);

    EXPECT_EQ(task.func.seen_value, 3);
    ASSERT_NE(task.func.seen_label, nullptr);
    EXPECT_EQ(task.func.seen_label, &ctx.get<std::string>());
    EXPECT_EQ(ctx.get<int>(), 7);
    EXPECT_EQ(result.status, yorch::step_status::retry);
}

TEST(BindTest, TaskIntoSugarBuildsDirectOutputTaskFromCallableAndSpecs) {
    yorch::context<int> ctx(6);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::task_into(
        [](int& value, yorch::direct_out<std::string> out) noexcept -> yorch::step_result {
            value += 1;
            return out.success(std::to_string(value * 2));
        })(
        yorch::from_ctx<int>());

    yorch::detail::typed_slot<std::string> slot;
    const auto result = task.invoke_into(exec, yorch::direct_out<std::string> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), "14");
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(BindTest, TaskMemberSugarBuildsBoundMemberTaskFromReceiverAndSpecs) {
    yorch::context<member_worker, int, std::string> ctx(member_worker {}, 3, "job");
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::task_member(
        &member_worker::accumulate,
        yorch::from_ctx<member_worker>())(
        yorch::from_ctx<int>(),
        yorch::from_ctx<std::string>(),
        yorch::value(4));

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(ctx.get<member_worker>().seen_value, 3);
    ASSERT_NE(ctx.get<member_worker>().seen_label, nullptr);
    EXPECT_EQ(ctx.get<member_worker>().seen_label, &ctx.get<std::string>());
    EXPECT_EQ(ctx.get<member_worker>().state, 4);
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(BindTest, TaskIntoMemberSugarBuildsDirectOutputMemberTaskFromReceiverAndSpecs) {
    member_worker worker;
    worker.state = 2;

    yorch::context<member_worker, int> ctx(worker, 6);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::task_into_member(
        &member_worker::emit,
        yorch::from_ctx<member_worker>())(
        yorch::from_ctx<int>());

    yorch::detail::typed_slot<std::string> slot;
    const auto result = task.invoke_into(exec, yorch::direct_out<std::string> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), "8");
    EXPECT_EQ(ctx.get<int>(), 8);
    EXPECT_EQ(ctx.get<member_worker>().seen_value, 6);
}

TEST(BindTest, TaskMemberSugarAppliesAdaptersFromOutsideToInside) {
    member_worker worker;
    yorch::exec_context<void> exec;

    auto task = yorch::task_member(
        &member_worker::mutate_state,
        yorch::value(std::ref(worker)),
        yorch::adapters(yorch::adapt_catch_as_failure()))(
        yorch::value(5));

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.state, 5);
}

TEST(BindTest, TaskIntoMemberSugarAppliesAdaptersFromOutsideToInside) {
    move_only_member_worker worker {9};
    yorch::exec_context<void, decltype(yorch::prev_slot(worker))> exec {
        yorch::prev_slot(worker)};

    auto task = yorch::task_into_member(
        &move_only_member_worker::emit,
        yorch::consume_prev<move_only_member_worker>(),
        yorch::adapters(yorch::adapt_catch_as_failure()))(
        );

    yorch::detail::typed_slot<int> slot;
    const auto result = task.invoke_into(exec, yorch::direct_out<int> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), 9);
}

TEST(BindTest, TaskMemberSugarRejectsMissingReceiverBinding) {
    static_assert(!can_make_task_member_without_receiver<decltype(&member_worker::accumulate)>);
    SUCCEED();
}

TEST(BindTest, TaskMemberSugarRejectsDirectOutputMemberFunctions) {
    static_assert(!can_make_task_member_from_receiver_first<decltype(&member_worker::emit)>);
    SUCCEED();
}

TEST(BindTest, TaskIntoMemberSugarRejectsMissingReceiverBinding) {
    static_assert(!can_make_task_into_member_without_receiver<decltype(&member_worker::emit)>);
    SUCCEED();
}

TEST(BindTest, TaskIntoMemberSugarRejectsOrdinaryMemberFunctions) {
    static_assert(!can_make_task_into_member_from_receiver_first<decltype(&member_worker::accumulate)>);
    SUCCEED();
}

TEST(BindTest, TaskSugarWithCatchAdapterAppliesDefaultCatchAsFailure) {
    yorch::exec_context<void> exec;

    auto task = yorch::task(
        []() -> yorch::step_result {
            throw std::runtime_error("boom");
        },
        yorch::adapters(yorch::adapt_catch_as_failure()))();

    const auto result = task.invoke_raw(exec);

    EXPECT_EQ(result.status, yorch::step_status::failure);
}

TEST(BindTest, TaskSugarWithRetryAdapterAppliesRetryPolicy) {
    yorch::exec_context<void> exec;
    int attempts = 0;

    auto task = yorch::task(
        [&]() noexcept -> yorch::step_result {
            ++attempts;
            return attempts < 3
                ? yorch::step_result::retry()
                : yorch::step_result::success();
        },
        yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {3})))();

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(attempts, 3);
}

TEST(BindTest, TaskSugarAppliesAdaptersFromOutsideToInside) {
    yorch::exec_context<void> exec;
    int attempts = 0;

    auto task = yorch::task(
        [&]() -> yorch::step_result {
            ++attempts;
            if (attempts == 1) {
                throw std::runtime_error("boom");
            }
            return yorch::step_result::success();
        },
        yorch::adapters(
            yorch::adapt_catch_as_failure(),
            yorch::adapt_retry(yorch::retry_fixed_policy {2})))();

    const auto result = task.invoke_raw(exec);

    EXPECT_EQ(result.status, yorch::step_status::failure);
    EXPECT_EQ(attempts, 1);
}
