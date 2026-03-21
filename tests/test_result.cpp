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
