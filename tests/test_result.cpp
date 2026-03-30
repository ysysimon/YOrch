#include "yorch/result.hpp"

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

namespace {

struct move_only_payload {
    explicit move_only_payload(int in) : value(in) {}

    move_only_payload(const move_only_payload&) = delete;
    move_only_payload& operator=(const move_only_payload&) = delete;
    move_only_payload(move_only_payload&&) noexcept = default;
    move_only_payload& operator=(move_only_payload&&) noexcept = default;

    int value = 0;
};

} // namespace

TEST(ResultTest, StepAndTaskReportSuccess) {
    static_assert(std::is_enum_v<yorch::step_status>);
    static_assert(std::is_same_v<decltype(yorch::step_result {}.status), yorch::step_status>);
    static_assert(!std::is_aggregate_v<yorch::task_result<int>>);
    static_assert(!std::is_default_constructible_v<yorch::task_result<int>>);

    auto task = yorch::task_result<int>::success(42);

    EXPECT_TRUE(yorch::step_result {}.ok());
    EXPECT_TRUE(task.step.ok());
    EXPECT_TRUE(task.has_value());
    EXPECT_EQ(task.value(), 42);
}

TEST(ResultTest, NonSuccessTaskResultsDoNotMaterializePayload) {
    const auto failure = yorch::task_result<int>::failure();
    const auto skip = yorch::task_result<int>::skip();
    const auto retry = yorch::task_result<int>::retry();
    const auto abort_branch = yorch::task_result<int>::abort_branch();
    const auto abort_execution = yorch::task_result<int>::abort_execution();
    const auto from_step = yorch::task_result<int>::from_step(yorch::step_result::failure());

    EXPECT_EQ(failure.step.status, yorch::step_status::failure);
    EXPECT_EQ(skip.step.status, yorch::step_status::skip);
    EXPECT_EQ(retry.step.status, yorch::step_status::retry);
    EXPECT_EQ(abort_branch.step.status, yorch::step_status::abort_branch);
    EXPECT_EQ(abort_execution.step.status, yorch::step_status::abort_execution);
    EXPECT_EQ(from_step.step.status, yorch::step_status::failure);
    EXPECT_FALSE(failure.has_value());
    EXPECT_FALSE(skip.has_value());
    EXPECT_FALSE(retry.has_value());
    EXPECT_FALSE(abort_branch.has_value());
    EXPECT_FALSE(abort_execution.has_value());
    EXPECT_FALSE(from_step.has_value());
}

TEST(ResultTest, TaskResultSupportsMoveOnlyPayloads) {
    auto task = yorch::task_result<move_only_payload>::success(move_only_payload {9});

    static_assert(!std::is_copy_constructible_v<decltype(task)>);
    static_assert(std::is_move_constructible_v<decltype(task)>);

    ASSERT_TRUE(task.has_value());
    EXPECT_EQ(task.value().value, 9);

    auto moved = std::move(task);

    ASSERT_TRUE(moved.has_value());
    EXPECT_EQ(moved.value().value, 9);
}

TEST(ResultTest, TaskResultTraitsExposePayloadType) {
    static_assert(yorch::is_task_result_v<yorch::task_result<int>>);
    static_assert(yorch::is_task_result_v<const yorch::task_result<void>&>);
    static_assert(!yorch::is_task_result_v<yorch::step_result>);
    static_assert(std::is_same_v<yorch::task_result_value_t<yorch::task_result<int>>, int>);
    static_assert(std::is_same_v<yorch::task_result_value_t<const yorch::task_result<void>&>, void>);

    constexpr yorch::task_result<void> task {};
    EXPECT_TRUE(task.step.ok());
}
