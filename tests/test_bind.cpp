#include "yorch/bind.hpp"

#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <type_traits>

namespace {

bool add_and_check_positive(int& value, int delta) {
    value += delta;
    return value > 0;
}

struct recorder {
    int seen_value = 0;
    const std::string* seen_label = nullptr;

    yorch::step_result operator()(int& value, const std::string& label, long delta) {
        seen_value = value;
        seen_label = &label;
        value += static_cast<int>(delta);
        return label == "job"
                 ? yorch::step_result::retry()
                 : yorch::step_result::failure();
    }
};

}  // namespace

TEST(BindTest, FunctionTraitsSupportFreeFunctionsAndLambdas) {
    auto lambda =
        [](int&, const std::string&) -> yorch::step_result {
        return yorch::step_result::success();
    };

    using function_traits = yorch::detail::function_traits<decltype(&add_and_check_positive)>;
    using lambda_traits = yorch::detail::function_traits<decltype(lambda)>;

    static_assert(function_traits::arity == 2);
    static_assert(std::is_same_v<function_traits::result_type, bool>);
    static_assert(std::is_same_v<function_traits::arg<0>, int&>);
    static_assert(std::is_same_v<function_traits::arg<1>, int>);

    static_assert(lambda_traits::arity == 2);
    static_assert(std::is_same_v<lambda_traits::result_type, yorch::step_result>);
    static_assert(std::is_same_v<yorch::detail::nth_arg_t<0, decltype(lambda)>, int&>);
    static_assert(
        std::is_same_v<yorch::detail::nth_arg_t<1, decltype(lambda)>, const std::string&>);

    EXPECT_TRUE((std::is_same_v<yorch::detail::result_t<decltype(lambda)>, yorch::step_result>));
}

TEST(BindTest, BoundTaskResolvesSpecsUsingCallableSignature) {
    yorch::context<int, std::string> ctx(3, "job");
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
        recorder {},
        yorch::from_ctx<int>(),
        yorch::from_ctx<std::string>(),
        yorch::value(4));

    static_assert(std::is_same_v<
                  decltype(task.specs),
                  std::tuple<
                      yorch::from_ctx_t<int>,
                      yorch::from_ctx_t<std::string>,
                      yorch::value_t<int>>>);

    const auto result = task(exec);

    EXPECT_EQ(task.func.seen_value, 3);
    ASSERT_NE(task.func.seen_label, nullptr);
    EXPECT_EQ(task.func.seen_label, &ctx.get<std::string>());
    EXPECT_EQ(ctx.get<int>(), 7);
    EXPECT_EQ(result.status, yorch::step_status::retry);
}

TEST(BindTest, BoundTaskNormalizesBoolReturnValues) {
    std::string source = "payload";
    yorch::exec_context<void> exec;

    auto success_task = yorch::bind(
        // NOLINTNEXTLINE(performance-unnecessary-value-param)
        [](std::string value) -> bool {
            return value == "payload";
        },
        yorch::value(source));

    auto failure_task = yorch::bind(
        [](int value) -> bool {
            return value > 0;
        },
        yorch::value(-1));

    auto const_ref_task = yorch::bind(
        [](const std::string& value) -> bool {
            return value == "payload";
        },
        yorch::value(source));

    source = "updated";

    const auto success = success_task(exec);
    const auto failure = failure_task(exec);
    const auto const_ref_success = const_ref_task(exec);

    EXPECT_EQ(std::get<0>(success_task.specs).v, "payload");
    EXPECT_EQ(std::get<0>(const_ref_task.specs).v, "payload");
    EXPECT_EQ(success.status, yorch::step_status::success);
    EXPECT_EQ(failure.status, yorch::step_status::failure);
    EXPECT_EQ(const_ref_success.status, yorch::step_status::success);
}

TEST(BindTest, BoundTaskNormalizesVoidReturnAsSuccess) {
    yorch::context<int> ctx(3);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
        [](int& value) {
            value *= 2;
        },
        yorch::from_ctx<int>());

    const auto result = task(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(ctx.get<int>(), 6);
}
