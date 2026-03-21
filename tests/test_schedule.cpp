#include "yorch/yorch.hpp"

#include <gtest/gtest.h>

TEST(ScheduleTest, ChainThenAndStepComposePlan) {
    auto first = yorch::bind([](int) { return yorch::step_result {}; }, yorch::value(1));
    auto next = yorch::bind([]() {});

    auto plan = yorch::schedule.chain(first).then(next).step(next);

    EXPECT_NO_THROW(static_cast<void>(plan));
}
