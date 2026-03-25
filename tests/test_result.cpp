#include "yorch/result.hpp"

#include <gtest/gtest.h>

TEST(ResultTest, StepAndTaskReportSuccess) {
    static_assert(std::is_enum_v<yorch::step_status>);
    static_assert(std::is_same_v<decltype(yorch::step_result {}.status), yorch::step_status>);

    constexpr int test_value = 42;
    yorch::step_result step {};
    yorch::task_result<int> task {{}, test_value};

    EXPECT_TRUE(step.ok());
    EXPECT_TRUE(task.step.ok());
    EXPECT_EQ(task.value, test_value);
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
