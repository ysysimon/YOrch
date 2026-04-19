#include "support.hpp"

using namespace bind_test_support;

namespace {

using forward_prev_borrow_mut_task_t = decltype(yorch::bind_forward_prev<int>(
    [](int& value) noexcept -> yorch::step_result {
        value += 1;
        return yorch::step_result::success();
    },
    yorch::borrow_prev_mut<int>()));

using forward_prev_consume_rvalue_task_t = decltype(yorch::bind_forward_prev<int>(
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    [](int&& value) noexcept -> yorch::step_result {
        value += 1;
        return yorch::step_result::success();
    },
    yorch::consume_prev<int>()));

static_assert(yorch::detail::task_prev_access_valid_v<forward_prev_borrow_mut_task_t>);
static_assert(yorch::detail::task_prev_access_valid_v<forward_prev_consume_rvalue_task_t>);
static_assert(yorch::detail::task_uses_forward_prev_output_protocol_v<forward_prev_borrow_mut_task_t>);
static_assert(yorch::detail::validate_bind_forward_prev<int, decltype([](int&) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::borrow_prev_mut_t<int>>() == yorch::detail::bind_forward_prev_error::ok);
static_assert(yorch::detail::validate_bind_forward_prev<void, decltype([](int&) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::borrow_prev_mut_t<int>>() == yorch::detail::bind_forward_prev_error::invalid_output_type);
static_assert(yorch::detail::bind_forward_prev_payload_matches_v<int, decltype([](int&) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::borrow_prev_mut_t<int>>);
static_assert(!yorch::detail::bind_forward_prev_bindings_supported_v<int, decltype([](const int&) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::borrow_prev_t<int>>);
static_assert(yorch::detail::validate_bind_forward_prev<int, decltype([](const int&) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::borrow_prev_t<int>>() == yorch::detail::bind_forward_prev_error::binding_mode_not_supported);
static_assert(!yorch::detail::bind_forward_prev_bindings_supported_v<int, decltype([](int) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::copy_prev_t<int>>);
static_assert(yorch::detail::validate_bind_forward_prev<int, decltype([](int) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::copy_prev_t<int>>() == yorch::detail::bind_forward_prev_error::binding_mode_not_supported);
static_assert(yorch::detail::bind_forward_prev_consume_by_value_requested_v<int, decltype([](int) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::consume_prev_t<int>>);
static_assert(yorch::detail::validate_bind_forward_prev<int, decltype([](int) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::consume_prev_t<int>>() == yorch::detail::bind_forward_prev_error::consume_by_value_not_supported);

}  // namespace

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
        [](int& value, yorch::direct_out<std::string> out) noexcept -> yorch::step_result {
            value += 1;
            return out.success(std::to_string(value * 2));
        },
        yorch::from_ctx<int>());

    static_assert(std::is_same_v<
                  decltype(task.specs),
                  std::tuple<yorch::from_ctx_t<int>>>);

    yorch::detail::typed_slot<std::string> slot;
    const auto result = task.invoke_into(exec, yorch::direct_out<std::string> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), "14");
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(BindTest, BoundForwardPrevTaskResolvesDirectParentAndExposesForwardPrevProtocol) {
    forward_prev_probe parent {7};
    yorch::exec_context<void, decltype(yorch::prev_slot(parent))> exec {
        yorch::prev_slot(parent)};

    auto task = yorch::bind_forward_prev<forward_prev_probe>(
        [](forward_prev_probe& value) noexcept -> yorch::step_result {
            value.value += 5;
            return yorch::step_result::success();
        },
        yorch::borrow_prev_mut<forward_prev_probe>());

    static_assert(std::is_same_v<
                  decltype(task.specs),
                  std::tuple<yorch::borrow_prev_mut_t<forward_prev_probe>>>);
    static_assert(yorch::detail::task_uses_forward_prev_output_protocol_v<decltype(task)>);
    static_assert(yorch::detail::task_prev_access_valid_v<decltype(task)>);
    static_assert(yorch::detail::task_uses_borrow_prev_mut_v<decltype(task)>);

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(parent.value, 12);
}

TEST(BindTest, TaskForwardPrevInfersOutputTypeFromUniquePrevAccessSpec) {
    auto task = yorch::task_forward_prev(
        [](forward_prev_probe& value) noexcept -> yorch::step_result {
            value.value += 1;
            return yorch::step_result::success();
        })(
        yorch::borrow_prev_mut<forward_prev_probe>());

    static_assert(std::is_same_v<
                  yorch::detail::declared_task_output_t<decltype(task)>,
                  forward_prev_probe>);
    static_assert(yorch::detail::task_uses_forward_prev_output_protocol_v<decltype(task)>);
}

TEST(BindTest, BoundOutputTaskCanConsumeParentPayloadAndForwardItToOutput) {
    std::string parent_value = "root";
    yorch::exec_context<void, decltype(yorch::prev_slot(parent_value))> exec {
        yorch::prev_slot(parent_value)};

    auto task = yorch::bind_into<std::string>(
        [](std::string&& value, yorch::direct_out<std::string> out) noexcept -> yorch::step_result {
            value += "-child";
            return out.success(std::move(value));
        },
        yorch::consume_prev<std::string>());

    yorch::detail::typed_slot<std::string> slot;
    const auto result = task.invoke_into(exec, yorch::direct_out<std::string> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), "root-child");
}

TEST(BindTest, BoundTaskCanCopyParentPayloadWithoutMutatingSource) {
    std::string parent_value = "root";
    yorch::exec_context<void, decltype(yorch::prev_slot(parent_value))> exec {
        yorch::prev_slot(parent_value)};

    auto task = yorch::bind(
        // NOLINTNEXTLINE(performance-unnecessary-value-param)
        [](std::string value) noexcept -> yorch::step_result {
            return value == "root"
                ? yorch::step_result::success()
                : yorch::step_result::failure();
        },
        yorch::copy_prev<std::string>());

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(parent_value, "root");
}

TEST(BindTest, BoundOutputTaskNormalizesVoidCallableReturnToSuccess) {
    yorch::exec_context<void> exec;

    auto task = yorch::bind_into<int>(
        [](yorch::direct_out<int> out) noexcept {
            out.emplace(9);
        });

    yorch::detail::typed_slot<int> slot;
    const auto result = task.invoke_into(exec, yorch::direct_out<int> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), 9);
}
