#include "yorch/bind.hpp"
#include "yorch/result.hpp"

#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

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

struct ref_recorder {
    int* seen_value = nullptr;
    const std::string* seen_label = nullptr;

    yorch::step_result operator()(int& value, const std::string& label) {
        seen_value = &value;
        seen_label = &label;
        value += 2;
        return yorch::step_result::success();
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

    const auto result = task.invoke_raw(exec);

    EXPECT_EQ(task.func.seen_value, 3);
    ASSERT_NE(task.func.seen_label, nullptr);
    EXPECT_EQ(task.func.seen_label, &ctx.get<std::string>());
    EXPECT_EQ(ctx.get<int>(), 7);
    EXPECT_EQ(result.status, yorch::step_status::retry);
}

TEST(BindTest, BoundTaskBorrowsContextObjectsByReferenceWhenCallableExpectsReferences) {
    yorch::context<int, std::string> ctx(3, "job");
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
        ref_recorder {},
        yorch::from_ctx<int>(),
        yorch::from_ctx<std::string>());

    const auto result = task.invoke_raw(exec);

    ASSERT_NE(task.func.seen_value, nullptr);
    ASSERT_NE(task.func.seen_label, nullptr);
    EXPECT_EQ(task.func.seen_value, &ctx.get<int>());
    EXPECT_EQ(task.func.seen_label, &ctx.get<std::string>());
    EXPECT_EQ(ctx.get<int>(), 5);
    EXPECT_TRUE(result.ok());
}

TEST(BindTest, BoundTaskPreservesRawBoolReturnValues) {
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

    const auto success = success_task.invoke_raw(exec);
    const auto failure = failure_task.invoke_raw(exec);
    const auto const_ref_success = const_ref_task.invoke_raw(exec);

    EXPECT_EQ(std::get<0>(success_task.specs).v, "payload");
    EXPECT_EQ(std::get<0>(const_ref_task.specs).v, "payload");
    EXPECT_TRUE(success);
    EXPECT_FALSE(failure);
    EXPECT_TRUE(const_ref_success);
}

TEST(BindTest, BoundTaskPreservesVoidReturnType) {
    yorch::context<int> ctx(3);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
        [](int& value) {
            value *= 2;
        },
        yorch::from_ctx<int>());

    static_assert(std::is_void_v<decltype(task.invoke_raw(exec))>);
    task.invoke_raw(exec);

    EXPECT_EQ(ctx.get<int>(), 6);
}

TEST(BindTest, BoundTaskCanResolveFromDirectParentSlot) {
    int parent_value = 5;
    yorch::exec_context<void, decltype(yorch::prev_slot(parent_value))> exec {
        yorch::prev_slot(parent_value)};

    auto task = yorch::bind(
        [](int& value) -> yorch::step_result {
            value += 3;
            return yorch::step_result::success();
        },
        yorch::borrow_prev_mut<int>());

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(parent_value, 8);
}

TEST(BindTest, BoundTaskPreservesTaskResultPayload) {
    yorch::context<int> ctx(3);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
        [](int& value) -> yorch::task_result<int> {
            value += 4;
            return yorch::task_result<int>::success(value * 2);
        },
        yorch::from_ctx<int>());

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.step.ok());
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 14);
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(BindTest, BoundOutputTaskResolvesSpecsAndWritesToOutputSink) {
    yorch::context<int> ctx(6);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind_into<std::string>(
        [](int& value, yorch::result_out<std::string> out) noexcept -> yorch::step_result {
            value += 1;
            return out.success(std::to_string(value * 2));
        },
        yorch::from_ctx<int>());

    static_assert(std::is_same_v<
                  decltype(task.specs),
                  std::tuple<yorch::from_ctx_t<int>>>);

    yorch::detail::typed_slot<std::string> slot;
    const auto result = task.invoke_into(exec, yorch::result_out<std::string> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), "14");
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(BindTest, BoundOutputTaskCanConsumeParentPayloadAndForwardItToOutput) {
    std::string parent_value = "root";
    yorch::exec_context<void, decltype(yorch::prev_slot(parent_value))> exec {
        yorch::prev_slot(parent_value)};

    auto task = yorch::bind_into<std::string>(
        [](std::string&& value, yorch::result_out<std::string> out) noexcept -> yorch::step_result {
            value += "-child";
            return out.success(std::move(value));
        },
        yorch::consume_prev<std::string>());

    yorch::detail::typed_slot<std::string> slot;
    const auto result = task.invoke_into(exec, yorch::result_out<std::string> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), "root-child");
}

TEST(BindTest, BoundOutputTaskNormalizesVoidCallableReturnToSuccess) {
    yorch::exec_context<void> exec;

    auto task = yorch::bind_into<int>(
        [](yorch::result_out<int> out) noexcept {
            out.emplace(9);
        });

    yorch::detail::typed_slot<int> slot;
    const auto result = task.invoke_into(exec, yorch::result_out<int> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), 9);
}
