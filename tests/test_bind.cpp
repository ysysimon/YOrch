#include "yorch/bind.hpp"
#include "yorch/executor.hpp" // IWYU pragma: keep
#include "yorch/result.hpp"

#include <gtest/gtest.h>

#include <functional>
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

struct member_worker {
    int seen_value = 0;
    const std::string* seen_label = nullptr;
    int state = 0;

    yorch::step_result accumulate(int& value, const std::string& label, long delta) {
        seen_value = value;
        seen_label = &label;
        state += static_cast<int>(delta);
        value += static_cast<int>(delta);
        return label == "job"
                 ? yorch::step_result::success()
                 : yorch::step_result::failure();
    }

    bool note(int value) noexcept {
        seen_value = value;
        state += value;
        return true;
    }

    [[nodiscard]] int read_state(int delta) const noexcept {
        return state + delta;
    }

    yorch::step_result mutate_state(int delta) noexcept {
        state += delta;
        return yorch::step_result::success();
    }

    yorch::step_result advance_and_emit(int delta, yorch::direct_out<int> out) noexcept {
        state += delta;
        return out.success(state);
    }

    [[nodiscard]] yorch::step_result emit_state(yorch::direct_out<int> out) const noexcept {
        return out.success(state);
    }

    yorch::step_result emit(int& value, yorch::direct_out<std::string> out) noexcept {
        seen_value = value;
        value += state;
        return out.success(std::to_string(value));
    }
};

struct move_only_member_worker {
    explicit move_only_member_worker(int initial) : state(initial) {}

    move_only_member_worker(const move_only_member_worker&) = delete;
    move_only_member_worker& operator=(const move_only_member_worker&) = delete;
    move_only_member_worker(move_only_member_worker&&) noexcept = default;
    move_only_member_worker& operator=(move_only_member_worker&&) noexcept = default;
    ~move_only_member_worker() = default;

    int state = 0;

    int bump(int delta) noexcept {
        state += delta;
        return state;
    }

    [[nodiscard]] yorch::step_result emit(yorch::direct_out<int> out) const noexcept {
        return out.success(state);
    }
};

template <typename F>
concept can_make_task_member_without_receiver =
    requires(F&& func) {
        requires (!std::is_void_v<decltype(yorch::task_member(std::forward<F>(func)))>);
    };

template <typename F>
concept can_make_task_member_from_receiver_first =
    requires(F&& func) {
        yorch::task_member(
            std::forward<F>(func),
            yorch::value(member_worker {}))(
            yorch::value(1));
    };

template <typename F>
concept can_make_task_into_member_without_receiver =
    requires(F&& func) {
        requires (!std::is_void_v<decltype(yorch::task_into_member(std::forward<F>(func)))>);
    };

template <typename F>
concept can_make_task_into_member_from_receiver_first =
    requires(F&& func) {
        yorch::task_into_member(
            std::forward<F>(func),
            yorch::value(member_worker {}))(
            yorch::value(1));
    };

struct forward_prev_probe {
    int value = 0;
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

TEST(BindTest, MemberFunctionTraitsCaptureReceiverAndBusinessParameters) {
    using member_traits = yorch::detail::member_function_traits<decltype(&member_worker::accumulate)>;
    using const_member_traits = yorch::detail::member_function_traits<decltype(&member_worker::read_state)>;
    using direct_output_member_t = decltype(&member_worker::emit);

    static_assert(member_traits::arity == 3);
    static_assert(std::is_same_v<member_traits::class_type, member_worker>);
    static_assert(std::is_same_v<member_traits::receiver_arg_type, member_worker&>);
    static_assert(std::is_same_v<member_traits::result_type, yorch::step_result>);
    static_assert(std::is_same_v<member_traits::arg<0>, int&>);
    static_assert(std::is_same_v<member_traits::arg<1>, const std::string&>);
    static_assert(std::is_same_v<member_traits::arg<2>, long>);

    static_assert(std::is_same_v<const_member_traits::receiver_arg_type, const member_worker&>);
    static_assert(std::is_same_v<const_member_traits::arg<0>, int>);

    static_assert(yorch::detail::ordinary_member_bind_callable<decltype(&member_worker::accumulate)>);
    static_assert(!yorch::detail::ordinary_bind_callable<decltype(&member_worker::accumulate)>);
    static_assert(yorch::detail::inferable_direct_output_member_callable<direct_output_member_t>);
    static_assert(!yorch::detail::inferable_direct_output_callable<direct_output_member_t>);

    EXPECT_TRUE((std::is_same_v<
                 yorch::detail::member_result_t<decltype(&member_worker::read_state)>,
                 int>));
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
static_assert(yorch::detail::bind_forward_prev_payload_matches_v<int, decltype([](int&) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::borrow_prev_mut_t<int>>);
static_assert(!yorch::detail::bind_forward_prev_bindings_supported_v<int, decltype([](const int&) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::borrow_prev_t<int>>);
static_assert(!yorch::detail::bind_forward_prev_bindings_supported_v<int, decltype([](int) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::copy_prev_t<int>>);
static_assert(yorch::detail::bind_forward_prev_consume_by_value_requested_v<int, decltype([](int) noexcept -> yorch::step_result {
    return yorch::step_result::success();
}), yorch::consume_prev_t<int>>);

}  // namespace

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

TEST(BindTest, BoundMemberTaskResolvesReceiverFromContext) {
    yorch::context<member_worker, int, std::string> ctx(member_worker {}, 3, "job");
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind_member(
        &member_worker::accumulate,
        yorch::from_ctx<member_worker>(),
        yorch::from_ctx<int>(),
        yorch::from_ctx<std::string>(),
        yorch::value(4));

    static_assert(std::is_same_v<decltype(task.receiver_spec), yorch::from_ctx_t<member_worker>>);
    static_assert(std::is_same_v<
                  decltype(task.specs),
                  std::tuple<
                      yorch::from_ctx_t<int>,
                      yorch::from_ctx_t<std::string>,
                      yorch::value_t<int>>>);

    const auto result = task.invoke_raw(exec);
    auto& worker = ctx.get<member_worker>();

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.seen_value, 3);
    ASSERT_NE(worker.seen_label, nullptr);
    EXPECT_EQ(worker.seen_label, &ctx.get<std::string>());
    EXPECT_EQ(worker.state, 4);
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(BindTest, BoundMemberTaskSupportsReferenceWrapperReceiverStorage) {
    member_worker worker;
    yorch::exec_context<void> exec;

    auto task = yorch::bind_member(
        &member_worker::note,
        yorch::value(std::ref(worker)),
        yorch::value(7));

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result);
    EXPECT_EQ(worker.seen_value, 7);
    EXPECT_EQ(worker.state, 7);
}

TEST(BindTest, BoundMemberTaskCanBorrowConstReceiverFromDirectParentSlot) {
    member_worker worker;
    worker.state = 11;

    yorch::exec_context<void, decltype(yorch::prev_slot(worker))> exec {
        yorch::prev_slot(worker)};

    auto task = yorch::bind_member(
        &member_worker::read_state,
        yorch::borrow_prev<member_worker>(),
        yorch::value(4));

    const auto result = task.invoke_raw(exec);

    EXPECT_EQ(result, 15);
}

TEST(BindTest, BoundMemberTaskCanBorrowMutableReceiverFromDirectParentSlot) {
    member_worker worker;
    worker.state = 3;

    yorch::exec_context<void, decltype(yorch::prev_slot(worker))> exec {
        yorch::prev_slot(worker)};

    auto task = yorch::bind_member(
        &member_worker::mutate_state,
        yorch::borrow_prev_mut<member_worker>(),
        yorch::value(5));

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.state, 8);
}

TEST(BindTest, BoundMemberTaskCanCopyDirectParentReceiverWithoutMutatingSource) {
    member_worker worker;
    worker.state = 9;

    yorch::exec_context<void, decltype(yorch::prev_slot(worker))> exec {
        yorch::prev_slot(worker)};

    auto task = yorch::bind_member(
        &member_worker::mutate_state,
        yorch::copy_prev<member_worker>(),
        yorch::value(4));

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.state, 9);
}

TEST(BindTest, BoundMemberTaskCanConsumeDirectParentReceiver) {
    move_only_member_worker worker {7};

    yorch::exec_context<void, decltype(yorch::prev_slot(worker))> exec {
        yorch::prev_slot(worker)};

    auto task = yorch::bind_member(
        &move_only_member_worker::bump,
        yorch::consume_prev<move_only_member_worker>(),
        yorch::value(5));

    const auto result = task.invoke_raw(exec);

    EXPECT_EQ(result, 12);
}

TEST(BindTest, BoundConstMemberTaskSupportsCopyAndConsumeReceivers) {
    member_worker copy_worker;
    copy_worker.state = 8;

    yorch::exec_context<void, decltype(yorch::prev_slot(copy_worker))> copy_exec {
        yorch::prev_slot(copy_worker)};

    auto copy_task = yorch::bind_member(
        &member_worker::read_state,
        yorch::copy_prev<member_worker>(),
        yorch::value(3));

    EXPECT_EQ(copy_task.invoke_raw(copy_exec), 11);

    move_only_member_worker consume_worker {10};
    yorch::exec_context<void, decltype(yorch::prev_slot(consume_worker))> consume_exec {
        yorch::prev_slot(consume_worker)};

    auto consume_task = yorch::bind_into_member<int>(
        &move_only_member_worker::emit,
        yorch::consume_prev<move_only_member_worker>());

    yorch::detail::typed_slot<int> slot;
    const auto step = consume_task.invoke_into(consume_exec, yorch::direct_out<int> {slot});

    EXPECT_TRUE(step.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), 10);
}

TEST(BindTest, BoundMemberOutputTaskResolvesReceiverAndWritesToOutputSink) {
    member_worker worker;
    worker.state = 2;

    yorch::context<member_worker, int> ctx(worker, 6);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind_into_member<std::string>(
        &member_worker::emit,
        yorch::from_ctx<member_worker>(),
        yorch::from_ctx<int>());

    yorch::detail::typed_slot<std::string> slot;
    const auto result = task.invoke_into(exec, yorch::direct_out<std::string> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), "8");
    EXPECT_EQ(ctx.get<int>(), 8);
    EXPECT_EQ(ctx.get<member_worker>().seen_value, 6);
}

TEST(BindTest, BoundMemberOutputTaskSupportsCopyAndConsumeReceivers) {
    member_worker copy_worker;
    copy_worker.state = 5;

    yorch::exec_context<void, decltype(yorch::prev_slot(copy_worker))> copy_exec {
        yorch::prev_slot(copy_worker)};

    auto copy_task = yorch::bind_into_member<int>(
        &member_worker::advance_and_emit,
        yorch::copy_prev<member_worker>(),
        yorch::value(4));

    yorch::detail::typed_slot<int> copy_slot;
    const auto copy_result = copy_task.invoke_into(copy_exec, yorch::direct_out<int> {copy_slot});

    EXPECT_TRUE(copy_result.ok());
    EXPECT_TRUE(copy_slot.has_value());
    EXPECT_EQ(copy_slot.get(), 9);
    EXPECT_EQ(copy_worker.state, 5);

    move_only_member_worker consume_worker {13};
    yorch::exec_context<void, decltype(yorch::prev_slot(consume_worker))> consume_exec {
        yorch::prev_slot(consume_worker)};

    auto consume_task = yorch::bind_into_member<int>(
        &move_only_member_worker::emit,
        yorch::consume_prev<move_only_member_worker>());

    yorch::detail::typed_slot<int> consume_slot;
    const auto consume_result =
        consume_task.invoke_into(consume_exec, yorch::direct_out<int> {consume_slot});

    EXPECT_TRUE(consume_result.ok());
    EXPECT_TRUE(consume_slot.has_value());
    EXPECT_EQ(consume_slot.get(), 13);
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
