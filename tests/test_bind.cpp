#include "yorch/yorch.hpp"

#include <gtest/gtest.h>

namespace {

struct demo_context {
    int scene = 0;
};

}  // namespace

TEST(BindTest, SupportsValueContextAndPreviousBindings) {
    auto bound = yorch::bind([](int) {}, yorch::value(42));
    auto ctx = yorch::from_ctx<&demo_context::scene>();
    auto prev = yorch::from_prev<int>();

    EXPECT_NO_THROW(static_cast<void>(bound));
    EXPECT_NO_THROW(static_cast<void>(ctx));
    EXPECT_NO_THROW(static_cast<void>(prev));
}
